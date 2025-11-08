

from dataclasses import dataclass, field
from typing import Optional


@dataclass(eq=False)
class CommandBase:
    command_id: int = field(init=False)

    # DAG structure
    input_commands: dict['CommandBase', None] = field(default_factory=dict)
    output_commands: dict['CommandBase', None] = field(default_factory=dict)

    prev_commands: dict['CommandBase', None] = field(default_factory=dict)
    next_commands: dict['CommandBase', None] = field(default_factory=dict)

    # Class variable to track the next ID
    _next_id: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        # Assign the current ID and increment the class-level counter
        self.command_id = CommandBase._next_id
        CommandBase._next_id += 1

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other 

@dataclass(eq=False)
class HostWriteBufferCommand(CommandBase):
    op_name:str =  "host_write_buffer"

    # 忽略 batch size 的信息
    batch_size: int = 0 
    vector_length: int = 0 
    
    src_host_buffer_addr:int  = 0 

    dst_channel_id: int = 0
    dst_packet_id: int = 0 

    


@dataclass(eq=False)
class HostReadBufferCommand(CommandBase):
    op_name: str = "host_read_buffer" 
    
    batch_size:int = 0 
    vector_length: int = 0

    src_channel_id: int = 0
    src_packet_id: int = 0 

    dst_host_buffer_addr: int = 0 # 


@dataclass(eq=False)
class PIMComputeCommand(CommandBase):
    op_name:str = "pim_compute"

    batch_size:int = 0 
    tile_shape:tuple[int,int] = (0,0)  # (K,N)
    
    channel_id: int = 0

    input_packet_id:int = 0 # 用于检测正确性, 实际上的 fifo 也没有操作的空间
    output_packet_id:int = 0






@dataclass(eq=False)
class HostVectorCommand(CommandBase):
    # 最复杂的部分 

    pass 

