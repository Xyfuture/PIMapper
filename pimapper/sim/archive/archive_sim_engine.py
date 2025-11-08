import math
from Desim.Core import SimModule,SimCoroutine,SimSession, SimTime,Event
from Desim.Sync import EventQueue
from Desim.module.FIFO import FIFO
from perf_tracer import PerfettoTracer
from typing import Dict, List, Optional
import logging

from ..core.hwspec import ComputeDie, Chip
from ..core.matrixspec import Mapping, Tile

# Set up logging
logger = logging.getLogger(__name__)


class SharedLinkEngine(SimModule):
    def __init__(self,bandwidth):
        super().__init__()

        self.bandwidth:float =  bandwidth
        self.transfer_id:int = 0

        self.running_transfers:int = 0
        self.transfer_status_table:dict[int,tuple[int,int,int]] = {}
        # transfer_id -> (start_time, left_bytes, end_time)

        self.pending_transfer_table:dict[int,int] = {}
        # transfer_id -> bytes

        self.transfer_event_table:dict[int,Event] = {}
        # transfer_id -> event

        # self.main_process_event_queue:EventQueue = EventQueue()
        self.main_process_event = Event()

        # logger.debug(f"SharedLinkEngine initialized with bandwidth: {bandwidth}")
        self.register_coroutine(self.process)


    def process(self,):
        # logger.debug("SharedLinkEngine process started")
        while True:
            # SimModule.wait(self.main_process_event_queue.event)
            SimModule.wait(self.main_process_event)
            cur_time = SimSession.sim_time.cycle
            # logger.debug(f"Processing transfers at cycle {cur_time}, running transfers: {self.running_transfers}")

            # 更新剩余的 bytes
            for transfer_id in list(self.transfer_status_table.keys()):
                cur_start_time,cur_left_bytes,end_time = self.transfer_status_table[transfer_id]
                new_left_bytes = int(cur_left_bytes - self.current_bandwidth*(cur_time-cur_start_time))
                self.transfer_status_table[transfer_id] = (cur_time,new_left_bytes,-1)
                          # logger.debug(f"Transfer {transfer_id}: {cur_left_bytes} -> {new_left_bytes} bytes remaining")

            # 删除已完成的传输
            completed_transfers = []
            for transfer_id in list(self.transfer_status_table.keys()):
                _,cur_left_bytes,_ = self.transfer_status_table[transfer_id]
                if cur_left_bytes <= 0:
                    completed_transfers.append(transfer_id)
                    self.transfer_status_table.pop(transfer_id)
                    event = self.transfer_event_table.pop(transfer_id)
                    event.notify(SimTime(0))
                    self.running_transfers -= 1

            if completed_transfers:
                pass
                # logger.debug(f"Completed transfers: {completed_transfers} at cycle {cur_time}")

            # 加入新开始的传输
            new_transfers = []
            for transfer_id in list(self.pending_transfer_table.keys()):
                if transfer_id in self.transfer_status_table:
                    raise ValueError("Transfer ID already exists")
                data_bytes = self.pending_transfer_table.pop(transfer_id)
                self.transfer_status_table[transfer_id] = (SimSession.sim_time.cycle,data_bytes,-1)
                self.running_transfers += 1
                new_transfers.append((transfer_id, data_bytes))

            if new_transfers:
                pass
                # logger.debug(f"Started new transfers: {[(tid, bytes_) for tid, bytes_ in new_transfers]} at cycle {cur_time}")
                # logger.debug(f"Current bandwidth per transfer: {self.current_bandwidth:.2f} bytes/cycle")

            # 计算下一个事件时间
            next_time = -1
            for transfer_id in list(self.transfer_status_table.keys()):
                start_time,left_bytes,_ = self.transfer_status_table[transfer_id]
                assert start_time == SimSession.sim_time.cycle
                end_time = start_time + math.ceil(left_bytes / self.current_bandwidth)
                assert end_time >= 0
                self.transfer_status_table[transfer_id] = (start_time,left_bytes,end_time)
                if next_time == -1:
                    next_time = end_time
                elif end_time < next_time:
                    next_time = end_time
                # self.main_process_event_queue.next_notify(SimTime(end_time - start_time))
                
                # logger.debug(f"Transfer {transfer_id} scheduled to complete at cycle {end_time}")
            if next_time != -1:
                assert next_time != start_time
                self.main_process_event.notify(SimTime(next_time - start_time))

    def transfer(self,data_bytes:int)->Event:
        self.transfer_id += 1
        transfer_id = self.transfer_id
        self.pending_transfer_table[transfer_id] = data_bytes
        event = Event()
        self.transfer_event_table[transfer_id] = event

        # logger.debug(f"Transfer request {transfer_id}: {data_bytes} bytes at cycle {SimSession.sim_time.cycle}")
        # logger.debug(f"Pending transfers: {list(self.pending_transfer_table.keys())}")

        # self.main_process_event_queue.next_notify(SimTime(0))
        self.main_process_event.notify(SimTime(0))
        return event



    @property
    def current_bandwidth(self)->float:
        if self.running_transfers == 0:
            return self.bandwidth
        bandwidth_per_transfer = self.bandwidth/self.running_transfers
        # logger.debug(f"Bandwidth allocation: {self.bandwidth} total / {self.running_transfers} transfers = {bandwidth_per_transfer:.2f} per transfer")
        return bandwidth_per_transfer



class BufferController:
    def __init__(self, buffer_size:int):
        self.buffer_size = buffer_size
        self.fifo = FIFO(buffer_size)   
    
    def lock_buffer(self,size:int):
        for _ in range(size):
            self.fifo.write(None)

    def release_buffer(self,size:int):
        for _ in range(size):
            _ = self.fifo.read()




class SimComputeDie(SimModule):
    def __init__(self, compute_die: ComputeDie, tracer:PerfettoTracer, fifo_size: int = 2):
        super().__init__()

        self.fifo_size = fifo_size
        self.tracer = tracer
        self.compute_die: ComputeDie = compute_die
        self.mapping: Optional[Mapping] = None
        self.task_queue:List[Tile] = []


        self.input_fifo: FIFO = FIFO(self.fifo_size)
        self.output_fifo: FIFO = FIFO(self.fifo_size)

        self.tracer_unit_name = f"ComputeDie-{self.compute_die.die_id}"
        self.track_info = self.tracer.register_module(self.tracer_unit_name)

        self.shared_link_engine:Optional[SharedLinkEngine] = None 

        if self.compute_die.spec.shared_bandwidth:
            # 使用共享的带宽
            self.shared_link_engine = SharedLinkEngine(self.compute_die.spec.shared_bandwidth)

        self.input_buffer_controller = BufferController(self.fifo_size)
        self.output_buffer_controller = BufferController(self.fifo_size)

        self.register_coroutine(self.input_process)
        self.register_coroutine(self.compute_process)
        self.register_coroutine(self.output_process)
    
    def set_mapping(self, mapping: Mapping):
        self.mapping = mapping
        self.task_queue = self.mapping.placement[self.compute_die.die_id]

    def get_sim_time(self):
        return float(SimSession.sim_time.cycle)

    def input_process(self):
        input_track_info = self.tracer.register_track("Input",self.track_info)

        for i,task in enumerate(self.task_queue):
            self.input_buffer_controller.lock_buffer(1)
            with self.tracer.record_event(input_track_info, self.get_sim_time, f"Input-Task-{i}"):
                                
                data_bytes = int (task.batches * task.rows  * task.data_format.input_dtype.bytes_per_element)
                if self.shared_link_engine:
                    event = self.shared_link_engine.transfer(data_bytes)
                    SimModule.wait(event)
                else:
                    assert self.compute_die.spec.get_input_bandwidth()
                    latency = int(data_bytes // self.compute_die.spec.get_input_bandwidth())
                    SimModule.wait_time(SimTime(latency))
            self.input_fifo.write(i)
    def compute_process(self):
        compute_track_info = self.tracer.register_track("Compute",self.track_info)

        for i,task in enumerate(self.task_queue):
            _ = self.input_fifo.read()

            # 运算开始前, 获取 output buffer
            self.output_buffer_controller.lock_buffer(1)

            with self.tracer.record_event(compute_track_info, self.get_sim_time, f"Compute-Task-{i}"):
                # 运算时间
                compute_ops = task.batches * task.rows * task.cols
                compute_latency = int(compute_ops // (self.compute_die.spec.compute_power * 10**3))

                memory_ops = task.rows * task.cols * task.data_format.weight_dtype.bytes_per_element
                memory_latency = int(memory_ops // (self.compute_die.spec.memory_bandwidth * 10**3))

    
                latency = max(compute_latency, memory_latency)

                SimModule.wait_time(SimTime(latency))
                self.output_fifo.write(i)
            # 运算结束之后释放 buffer
            self.input_buffer_controller.release_buffer(1)

    def output_process(self):
        output_track_info = self.tracer.register_track("Output",self.track_info)

        for i,task in enumerate(self.task_queue):
            _ = self.output_fifo.read()
            with self.tracer.record_event(output_track_info, self.get_sim_time, f"Output-Task-{i}"):
                data_bytes = int(task.batches * task.cols * task.data_format.output_dtype.bytes_per_element)
                if self.shared_link_engine:
                    event = self.shared_link_engine.transfer(data_bytes)
                    SimModule.wait(event)   
                else:
                    assert self.compute_die.spec.get_output_bandwidth()
                    latency = int(data_bytes // self.compute_die.spec.get_output_bandwidth())
                    SimModule.wait_time(SimTime(latency))
            self.output_buffer_controller.release_buffer(1)

class SimChip:
    def __init__(self, chip: Chip, fifo_size: int = 2):
        
        SimSession.reset()
        SimSession.init()

        self.running_cycles = 0

        self.tracer = PerfettoTracer(100)
        self.chip: Chip = chip
        self.mapping: Optional[Mapping] = None

        self.sim_compute_dies: Dict[str, SimComputeDie] = {}
        for die_id, compute_die in self.chip.compute_dies.items():
            self.sim_compute_dies[die_id] = SimComputeDie(
                compute_die=compute_die,
                tracer=self.tracer,
                fifo_size=fifo_size
            )

    def set_mapping(self, mapping: Mapping):
        self.mapping = mapping
        for die_id, sim_compute_die in self.sim_compute_dies.items():
            sim_compute_die.set_mapping(mapping)

    def run_sim(self):
        SimSession.scheduler.run()
        # self.tracer.save("trace.json")

        self.running_cycles = int(SimSession.sim_time.cycle)
        
    def get_running_cycles(self):
        return self.running_cycles  

    def save_trace_file(self, filename: str = "trace.json"):
        self.tracer.save(filename)



def simulate(chip:Chip, mapping: Mapping,save_trace:bool=False, trace_filename:str="main_trace.json")->int:
    sim_chip = SimChip(chip)
    sim_chip.set_mapping(mapping)
    sim_chip.run_sim()
    if save_trace:
        sim_chip.save_trace_file(trace_filename)
    return sim_chip.get_running_cycles()