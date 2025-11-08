from typing import Optional
from Desim.Core import SimModule, SimTime
from Desim.module.FIFO import FIFO

from pimapper.core.instruction import (
    CommandBase,
    HostWriteBufferCommand,
    HostReadBufferCommand,
    PIMComputeCommand
)
from pimapper.sim.graph import CommandGraph
from pimapper.sim.resource import SimHost, SimPIMChannel


class ExecutorBase(SimModule):
    def __init__(self):
        super().__init__()


class HostWriteBufferExecutor(ExecutorBase):
    def __init__(self, command: HostWriteBufferCommand, host: SimHost, channels: list[SimPIMChannel]):
        super().__init__()
        self.command = command
        self.host = host
        self.channels = channels
        self.register_coroutine(self.process)

    def process(self):
        # Acquire resources
        self.channels[self.command.dst_channel_id].host_channel_link.wait()

        # Notify graph engine
        GraphExecuteEngine.current_engine.upcoming_fifo.write(self.command)

        # Execute
        SimModule.wait_time(SimTime(self.command.batch_size * self.command.vector_length))

        # Release resources
        self.channels[self.command.dst_channel_id].host_channel_link.post()


class HostReadBufferExecutor(ExecutorBase):
    def __init__(self, command: HostReadBufferCommand, host: SimHost, channels: list[SimPIMChannel]):
        super().__init__()
        self.command = command
        self.host = host
        self.channels = channels
        self.register_coroutine(self.process)

    def process(self):
        # Acquire resources
        self.channels[self.command.src_channel_id].host_channel_link.wait()

        # Notify graph engine
        GraphExecuteEngine.current_engine.upcoming_fifo.write(self.command)

        # Execute
        SimModule.wait_time(SimTime(self.command.batch_size * self.command.vector_length))

        # Release resources
        self.channels[self.command.src_channel_id].host_channel_link.post()


class PIMComputeExecutor(ExecutorBase):
    def __init__(self, command: PIMComputeCommand, channels: list[SimPIMChannel]):
        super().__init__()
        self.command = command
        self.channels = channels
        self.register_coroutine(self.process)

    def process(self):
        # Acquire resources
        self.channels[self.command.channel_id].channel_compute.wait()

        # Notify graph engine
        GraphExecuteEngine.current_engine.upcoming_fifo.write(self.command)

        # Execute
        K, N = self.command.tile_shape
        SimModule.wait_time(SimTime(self.command.batch_size * K * N))

        # Release resources
        self.channels[self.command.channel_id].channel_compute.post()


class GraphExecuteEngine(SimModule):
    current_engine: Optional['GraphExecuteEngine'] = None

    def __init__(self, graph: CommandGraph, host: SimHost, channels: list[SimPIMChannel]):
        super().__init__()
        self.graph = graph
        self.host = host
        self.channels = channels

        self.upcoming_fifo: FIFO[CommandBase] = FIFO(100)
        self.pending_fifo: FIFO[CommandBase] = FIFO(100)

        self.dep_map = graph.gen_dep_map()

        if graph.root_command:
            self.pending_fifo.write(graph.root_command)

        GraphExecuteEngine.current_engine = self

        self.register_coroutine(self.graph_topo_process)
        self.register_coroutine(self.issue_command)

    def graph_topo_process(self):
        while True:
            upcoming_command = self.upcoming_fifo.read()

            for next_command in upcoming_command.next_commands.keys():
                self.dep_map[next_command] -= 1
                if self.dep_map[next_command] == 0:
                    self.pending_fifo.write(next_command)

    def issue_command(self):
        while True:
            pending_command = self.pending_fifo.read()

            if isinstance(pending_command, HostWriteBufferCommand):
                HostWriteBufferExecutor(pending_command, self.host, self.channels)
            elif isinstance(pending_command, HostReadBufferCommand):
                HostReadBufferExecutor(pending_command, self.host, self.channels)
            elif isinstance(pending_command, PIMComputeCommand):
                PIMComputeExecutor(pending_command, self.channels)
            elif isinstance(pending_command, CommandBase):
                # Root command: add children to pending
                for child_command in pending_command.output_commands.keys():
                    self.pending_fifo.write(child_command)
