

from dataclasses import dataclass, field


@dataclass
class CommandBase:
    command_id: int = field(init=False)

    # Class variable to track the next ID
    _next_id: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        # Assign the current ID and increment the class-level counter
        self.command_id = CommandBase._next_id
        CommandBase._next_id += 1 

@dataclass
class HostWriteBufferCommand(CommandBase):
    op_name:str =  "host_write_buffer"

    # 忽略 batch size 的信息
    batch_size: int = 0 
    vector_length: int = 0 
    
    src_host_buffer_addr:int  = 0 

    dst_channel_id: int = 0
    dst_packet_id: int = 0 

    


@dataclass
class HostReadBufferCommand(CommandBase):
    op_name: str = "host_read_buffer" 
    
    batch_size:int = 0 
    vector_length: int = 0

    src_channel_id: int = 0
    src_packet_id: int = 0 

    dst_host_buffer_addr: int = 0 # 


@dataclass
class PIMComputeCommand(CommandBase):
    op_name:str = "pim_compute"

    batch_size:int = 0 
    tile_shape:tuple[int,int] = (0,0)  # (K,N)
    
    channel_id: int = 0

    input_packet_id:int = 0 # 用于检测正确性, 实际上的 fifo 也没有操作的空间
    output_packet_id:int = 0






@dataclass
class HostVectorCommand(CommandBase):
    # 最复杂的部分 

    pass 

