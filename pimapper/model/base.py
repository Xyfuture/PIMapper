

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal

@dataclass
class InferenceConfig:
    batch_size: int = 1
    past_seq_len: int = 1024

@dataclass
class ModelConfig:
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 0
    vocab_size: int = 0
    model_type: Literal['llama','gpt','mixtral','phi','mistral','phi3','qwen2'] = 'llama' # use different ffn design

    @property
    def use_gqa(self):
        if self.num_key_value_heads == 0 or self.num_key_value_heads == self.num_attention_heads:
            return False # use MHA not GQA
        elif self.num_attention_heads != self.num_key_value_heads:
            return True # use GQA

    @property
    def per_head_size(self):
        return self.hidden_size // self.num_attention_heads


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0, dtype=torch.float16):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype

    def forward(self, x):
        """
        Apply Rotary Position Embedding to input tensor x.

        Args:
            x: input tensor of shape (bsz, n_heads, seqlen, head_dim)

        Returns:
            Tensor with RoPE applied
        """
        _, _, seqlen, head_dim = x.shape

        # Create position indices [0, 1, 2, ..., seqlen-1]
        position = torch.arange(seqlen, device=x.device, dtype=self.dtype).unsqueeze(1)  # (seqlen, 1)

        # Create frequency matrix
        freqs = torch.pow(self.base, -torch.arange(0, head_dim, 2, device=x.device, dtype=self.dtype) / head_dim)  # (head_dim/2,)

        # Compute rotation angles
        angles = position * freqs  # (seqlen, head_dim/2)

        # Apply cos and sin to create rotation matrix
        cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, head_dim/2)
        sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, head_dim/2)

        # Split x into even and odd dimensions
        x_even = x[..., ::2]  # (bsz, n_heads, seqlen, head_dim/2)
        x_odd = x[..., 1::2]  # (bsz, n_heads, seqlen, head_dim/2)

        # Apply rotation: [cos*sin; -sin*cos] * [x_even; x_odd]
        x_rotated = torch.cat([
            x_even * cos_vals - x_odd * sin_vals,
            x_even * sin_vals + x_odd * cos_vals
        ], dim=-1)  # (bsz, n_heads, seqlen, head_dim)

        return x_rotated


class BatchedMatMulWithPast(nn.Module):
    """Batched matrix multiplication with past KV cache.

    This module performs batched matmul between input and past_matrix.
    Used for attention computation: Q @ K^T or scores @ V.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        is_qk_matmul: bool = True,
        dtype=torch.float16
    ):
        super().__init__()
        self.model_config = model_config
        self.inference_config = inference_config
        self.is_qk_matmul = is_qk_matmul
        self.dtype = dtype

        # Determine number of heads (handle GQA)
        self.n_heads = model_config.num_attention_heads
        if model_config.use_gqa:
            self.n_kv_heads = model_config.num_key_value_heads
        else:
            self.n_kv_heads = self.n_heads

        self.head_dim = model_config.per_head_size
        batch_size = inference_config.batch_size
        past_seq_len = inference_config.past_seq_len

        # For Q @ K^T: input is (bsz, n_heads, seqlen, head_dim), past is (bsz, n_kv_heads, past_seq_len, head_dim)
        # Output: (bsz, n_heads, seqlen, past_seq_len)
        # For scores @ V: input is (bsz, n_heads, seqlen, past_seq_len), past is (bsz, n_kv_heads, past_seq_len, head_dim)
        # Output: (bsz, n_heads, seqlen, head_dim)

        if is_qk_matmul:
            # Q @ K^T: past_matrix is K with shape (bsz, n_kv_heads, past_seq_len, head_dim)
            past_shape = (batch_size, self.n_kv_heads, past_seq_len, self.head_dim)
        else:
            # scores @ V: past_matrix is V with shape (bsz, n_kv_heads, past_seq_len, head_dim)
            past_shape = (batch_size, self.n_kv_heads, past_seq_len, self.head_dim)

        # Register past_matrix as a buffer (not a parameter, won't be trained)
        self.register_buffer('past_matrix', torch.zeros(past_shape, dtype=dtype))

    def forward(self, input, coming_kv):
        """
        Args:
            input: Query tensor (bsz, n_heads, seqlen, head_dim) for QK matmul
                   or scores tensor (bsz, n_heads, seqlen, past_seq_len) for score-V matmul
            coming_kv: New KV tensor (not used in computation, just passed through)

        Returns:
            Result of batched matmul between input and past_matrix
        """
        # coming_kv is not used in this operation, it's just for graph tracing

        if self.is_qk_matmul:
            # Q @ K^T: input (bsz, n_heads, seqlen, head_dim) @ past_matrix^T (bsz, n_kv_heads, head_dim, past_seq_len)
            # Handle GQA by repeating KV heads if needed
            past = self.past_matrix
            if self.n_heads != self.n_kv_heads:
                # Repeat KV heads to match Q heads
                repeat_factor = self.n_heads // self.n_kv_heads
                past = past.repeat_interleave(repeat_factor, dim=1)

            # Transpose last two dims of past for K^T
            result = torch.matmul(input, past.transpose(-2, -1))
        else:
            # scores @ V: input (bsz, n_heads, seqlen, past_seq_len) @ past_matrix (bsz, n_kv_heads, past_seq_len, head_dim)
            past = self.past_matrix
            if self.n_heads != self.n_kv_heads:
                repeat_factor = self.n_heads // self.n_kv_heads
                past = past.repeat_interleave(repeat_factor, dim=1)

            result = torch.matmul(input, past)

        return result






class FFNLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig
    ):
        super().__init__()

        self.config = model_config

        self.dim = model_config.hidden_size
        self.ffn_dim = model_config.intermediate_size
        self.dtype = torch.float16

        # FFN Block
        self.w1 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )
        self.w2 = nn.Linear(
            self.ffn_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )
        self.w3 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )

    def forward(self, x):
        # FFN: w2(w1(x) * silu(w3(x)))
        x1 = self.w1(x)
        x3 = F.silu(self.w3(x))
        x2 = self.w2(x1 * x3)
        return x2


class LLaMALayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig = None
    ):
        super().__init__()

        self.config = model_config
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()

        self.dim = model_config.hidden_size
        self.ffn_dim = model_config.intermediate_size
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.n_heads = model_config.num_attention_heads
        self.n_kv_heads = model_config.num_key_value_heads
        self.dtype = torch.float16

        # Attention Block
        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )

        # FFN Block
        self.w1 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )
        self.w2 = nn.Linear(
            self.ffn_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )
        self.w3 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )

        # RMSNorm layers
        self.attention_norm = nn.RMSNorm(self.dim, dtype=self.dtype)
        self.ffn_norm = nn.RMSNorm(self.dim, dtype=self.dtype)

        # RoPE module
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, base=10000.0, dtype=self.dtype)

        # BatchedMatMulWithPast modules for attention
        self.qk_matmul = BatchedMatMulWithPast(
            model_config=model_config,
            inference_config=self.inference_config,
            is_qk_matmul=True,
            dtype=self.dtype
        )
        self.score_v_matmul = BatchedMatMulWithPast(
            model_config=model_config,
            inference_config=self.inference_config,
            is_qk_matmul=False,
            dtype=self.dtype
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        # Attention with pre-normalization
        h = self.attention_norm(x)

        xq = self.wq(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xk = self.wk(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        xq = self.rotary_emb(xq)
        xk = self.rotary_emb(xk)

        # Use BatchedMatMulWithPast for Q @ K^T
        scores_raw = self.qk_matmul(xq, xk)  # (bsz, n_heads, seqlen, past_seq_len)
        scores = torch.softmax(scores_raw, dim=-1)

        # Use BatchedMatMulWithPast for scores @ V
        output = self.score_v_matmul(scores, xv).transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        xo = self.wo(output)

        # FFN with pre-normalization
        h = self.ffn_norm(xo)
        x1 = self.w1(h)
        x3 = F.silu(self.w3(h))

        x2 = self.w2(x1*x3)

        return x2


def load_model_config(card_path: str | Path) -> ModelConfig:
    """Load a ModelConfig from a JSON model card."""
    path = Path(card_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model card not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    config_kwargs = {
        "hidden_size": data.get("hidden_size"),
        "intermediate_size": data.get("intermediate_size"),
        "num_hidden_layers": data.get("num_hidden_layers"),
        "num_attention_heads": data.get("num_attention_heads"),
        "num_key_value_heads": data.get("num_key_value_heads", 0),
        "vocab_size": data.get("vocab_size", 0),
        "model_type": data.get("model_type", "llama"),
    }
    missing = [k for k, v in config_kwargs.items() if v is None]
    if missing:
        raise ValueError(f"Missing fields in model card {path}: {missing}")

    return ModelConfig(**config_kwargs)


def initialize_module(
    config: ModelConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Instantiate the LLaMA layer with the supplied configuration."""
    module = LLaMALayer(config)
    module.eval()
    module.to(device=device, dtype=dtype)

    # Ensure helper modules that cache dtype (e.g. RotaryPositionEmbedding) stay in sync.
    for submodule in module.modules():
        if hasattr(submodule, "dtype") and isinstance(getattr(submodule, "dtype"), torch.dtype):
            setattr(submodule, "dtype", dtype)

    return module
