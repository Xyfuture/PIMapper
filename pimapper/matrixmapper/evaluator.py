"""Matrix mapping evaluator for performance estimation.

This module provides a simplified discrete-event simulator for evaluating
matrix mapping strategies. It is used internally by matrixmapper strategies
to estimate the performance of different tile placements.

Note: This is a lightweight evaluator for mapping optimization, not the
full system simulator.
"""

import math
from Desim.Core import SimModule, SimCoroutine, SimSession, SimTime, Event
from Desim.Sync import EventQueue
from Desim.module.FIFO import FIFO
from perf_tracer import PerfettoTracer
from typing import Dict, List, Optional
import logging

from ..core.hwspec import PIMChannel, Accelerator
from ..core.matrixspec import Mapping, Tile

# Set up logging
logger = logging.getLogger(__name__)


class SharedLinkEngine(SimModule):
    def __init__(self, bandwidth):
        super().__init__()

        self.bandwidth: float = bandwidth
        self.transfer_id: int = 0

        self.running_transfers: int = 0
        self.transfer_status_table: dict[int, tuple[int, int, int]] = {}
        # transfer_id -> (start_time, left_bytes, end_time)

        self.pending_transfer_table: dict[int, int] = {}
        # transfer_id -> bytes

        self.transfer_event_table: dict[int, Event] = {}
        # transfer_id -> event

        self.main_process_event = Event()

        self.register_coroutine(self.process)

    def process(self):
        while True:
            SimModule.wait(self.main_process_event)
            cur_time = SimSession.sim_time.cycle

            # Update remaining bytes
            for transfer_id in list(self.transfer_status_table.keys()):
                cur_start_time, cur_left_bytes, end_time = self.transfer_status_table[transfer_id]
                new_left_bytes = int(cur_left_bytes - self.current_bandwidth * (cur_time - cur_start_time))
                self.transfer_status_table[transfer_id] = (cur_time, new_left_bytes, -1)

            # Remove completed transfers
            completed_transfers = []
            for transfer_id in list(self.transfer_status_table.keys()):
                _, cur_left_bytes, _ = self.transfer_status_table[transfer_id]
                if cur_left_bytes <= 0:
                    completed_transfers.append(transfer_id)
                    self.transfer_status_table.pop(transfer_id)
                    event = self.transfer_event_table.pop(transfer_id)
                    event.notify(SimTime(0))
                    self.running_transfers -= 1

            # Add new transfers
            new_transfers = []
            for transfer_id in list(self.pending_transfer_table.keys()):
                if transfer_id in self.transfer_status_table:
                    raise ValueError("Transfer ID already exists")
                data_bytes = self.pending_transfer_table.pop(transfer_id)
                self.transfer_status_table[transfer_id] = (SimSession.sim_time.cycle, data_bytes, -1)
                self.running_transfers += 1
                new_transfers.append((transfer_id, data_bytes))

            # Calculate next event time
            next_time = -1
            for transfer_id in list(self.transfer_status_table.keys()):
                start_time, left_bytes, _ = self.transfer_status_table[transfer_id]
                assert start_time == SimSession.sim_time.cycle
                end_time = start_time + math.ceil(left_bytes / self.current_bandwidth)
                assert end_time >= 0
                self.transfer_status_table[transfer_id] = (start_time, left_bytes, end_time)
                if next_time == -1:
                    next_time = end_time
                elif end_time < next_time:
                    next_time = end_time

            if next_time != -1:
                assert next_time != start_time
                self.main_process_event.notify(SimTime(next_time - start_time))

    def transfer(self, data_bytes: int) -> Event:
        self.transfer_id += 1
        transfer_id = self.transfer_id
        self.pending_transfer_table[transfer_id] = data_bytes
        event = Event()
        self.transfer_event_table[transfer_id] = event

        self.main_process_event.notify(SimTime(0))
        return event

    @property
    def current_bandwidth(self) -> float:
        if self.running_transfers == 0:
            return self.bandwidth
        bandwidth_per_transfer = self.bandwidth / self.running_transfers
        return bandwidth_per_transfer


class BufferController:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.fifo = FIFO(buffer_size)

    def lock_buffer(self, size: int):
        for _ in range(size):
            self.fifo.write(None)

    def release_buffer(self, size: int):
        for _ in range(size):
            _ = self.fifo.read()


class PIMChannelModel(SimModule):
    """Simulated PIM channel for matrix mapping evaluation."""

    def __init__(self, pim_channel: PIMChannel, tracer: PerfettoTracer, fifo_size: int = 10):
        super().__init__()

        self.fifo_size = 100
        self.tracer = tracer
        self.pim_channel: PIMChannel = pim_channel
        self.mapping: Optional[Mapping] = None
        self.task_queue: List[Tile] = []

        self.input_fifo: FIFO = FIFO(self.fifo_size)
        self.output_fifo: FIFO = FIFO(self.fifo_size)

        self.tracer_unit_name = f"PIMChannel-{self.pim_channel.channel_id}"
        self.track_info = self.tracer.register_module(self.tracer_unit_name)

        self.shared_link_engine: Optional[SharedLinkEngine] = None

        # if self.pim_channel.spec.shared_bandwidth:
            # Use shared bandwidth
        self.shared_link_engine = SharedLinkEngine(self.pim_channel.spec.get_input_bandwidth())

        self.input_buffer_controller = BufferController(self.fifo_size)
        self.output_buffer_controller = BufferController(self.fifo_size)

        self.register_coroutine(self.input_process)
        self.register_coroutine(self.compute_process)
        self.register_coroutine(self.output_process)

    def set_mapping(self, mapping: Mapping):
        self.mapping = mapping
        self.task_queue = self.mapping.placement[self.pim_channel.channel_id]

    def get_sim_time(self):
        return float(SimSession.sim_time.cycle)

    def input_process(self):
        input_track_info = self.tracer.register_track("Input", self.track_info)

        for i, task in enumerate(self.task_queue):
            self.input_buffer_controller.lock_buffer(1)
            with self.tracer.record_event(input_track_info, self.get_sim_time, f"Input-Task-{i}"):

                data_bytes = int(task.batches * task.rows * task.data_format.input_dtype.bytes_per_element)
                if self.shared_link_engine:
                    event = self.shared_link_engine.transfer(data_bytes)
                    SimModule.wait(event)
                else:
                    assert self.pim_channel.spec.get_input_bandwidth()
                    latency = int(data_bytes // self.pim_channel.spec.get_input_bandwidth())
                    SimModule.wait_time(SimTime(latency))
            self.input_fifo.write(i)

    def compute_process(self):
        compute_track_info = self.tracer.register_track("Compute", self.track_info)

        for i, task in enumerate(self.task_queue):
            _ = self.input_fifo.read()

            # Acquire output buffer before computation
            self.output_buffer_controller.lock_buffer(1)

            with self.tracer.record_event(compute_track_info, self.get_sim_time, f"Compute-Task-{i}"):
                # Compute time
                compute_ops = task.batches * task.rows * task.cols
                compute_latency = int(compute_ops // (self.pim_channel.spec.compute_power * 10**3))

                memory_ops = task.rows * task.cols * task.data_format.weight_dtype.bytes_per_element
                memory_latency = int(memory_ops // (self.pim_channel.spec.memory_bandwidth * 10**3))

                latency = max(compute_latency, memory_latency)

                SimModule.wait_time(SimTime(latency))
                self.output_fifo.write(i)
            # Release input buffer after computation
            self.input_buffer_controller.release_buffer(1)

    def output_process(self):
        output_track_info = self.tracer.register_track("Output", self.track_info)

        for i, task in enumerate(self.task_queue):
            _ = self.output_fifo.read()
            with self.tracer.record_event(output_track_info, self.get_sim_time, f"Output-Task-{i}"):
                data_bytes = int(task.batches * task.cols * task.data_format.output_dtype.bytes_per_element)
                if self.shared_link_engine:
                    event = self.shared_link_engine.transfer(data_bytes)
                    SimModule.wait(event)
                else:
                    assert self.pim_channel.spec.get_output_bandwidth()
                    latency = int(data_bytes // self.pim_channel.spec.get_output_bandwidth())
                    SimModule.wait_time(SimTime(latency))
            self.output_buffer_controller.release_buffer(1)


class AcceleratorModel:
    """Simulated accelerator for matrix mapping evaluation."""

    def __init__(self, accelerator: Accelerator, fifo_size: int = 2):

        SimSession.reset()
        SimSession.init()

        self.running_cycles = 0

        self.tracer = PerfettoTracer(100)
        self.accelerator: Accelerator = accelerator
        self.mapping: Optional[Mapping] = None

        self.sim_channels: Dict[str, PIMChannelModel] = {}
        for channel_id, pim_channel in self.accelerator.channels.items():
            self.sim_channels[channel_id] = PIMChannelModel(
                pim_channel=pim_channel,
                tracer=self.tracer,
                fifo_size=fifo_size
            )

    def set_mapping(self, mapping: Mapping):
        self.mapping = mapping
        for channel_id, sim_channel in self.sim_channels.items():
            sim_channel.set_mapping(mapping)

    def run_sim(self):
        SimSession.scheduler.run()

        self.running_cycles = int(SimSession.sim_time.cycle)

    def get_running_cycles(self):
        return self.running_cycles

    def save_trace_file(self, filename: str = "trace.json"):
        self.tracer.save(filename)


def evaluate(accelerator: Accelerator, mapping: Mapping, save_trace: bool = False, trace_filename: str = "mapping_trace.json") -> int:
    """Evaluate a matrix mapping and return the estimated latency in cycles.

    Args:
        accelerator: The accelerator hardware configuration
        mapping: The matrix-to-channel mapping to evaluate
        save_trace: Whether to save a Perfetto trace file
        trace_filename: Name of the trace file to save

    Returns:
        Estimated latency in cycles
    """
    sim_accelerator = AcceleratorModel(accelerator)
    sim_accelerator.set_mapping(mapping)
    sim_accelerator.run_sim()
    if save_trace:
        sim_accelerator.save_trace_file(trace_filename)
    return sim_accelerator.get_running_cycles()


# Backwards compatibility alias
simulate = evaluate
