from Desim.Core import SimSession

from pimapper.core.instruction import CommandBase, HostWriteBufferCommand, PIMComputeCommand, HostReadBufferCommand
from pimapper.sim.graph import CommandGraph
from pimapper.sim.resource import SimHost, SimPIMChannel
from pimapper.sim.executor import GraphExecuteEngine


SimSession.reset()
SimSession.init()

# Create root command
root_command = CommandBase()

# Create 5 channels of commands
channels_commands = []
for channel_id in range(5):
    write_cmd = HostWriteBufferCommand(
        batch_size=16,
        vector_length=1024,
        dst_channel_id=channel_id
    )

    compute_cmd = PIMComputeCommand(
        batch_size=16,
        tile_shape=(128, 128),
        channel_id=channel_id
    )

    read_cmd = HostReadBufferCommand(
        batch_size=16,
        vector_length=2048,
        src_channel_id=channel_id
    )

    # Build DAG: root -> write -> compute -> read
    write_cmd.input_commands[root_command] = None
    compute_cmd.input_commands[write_cmd] = None
    read_cmd.input_commands[compute_cmd] = None

    channels_commands.append((write_cmd, compute_cmd, read_cmd))

# Build command graph
graph = CommandGraph()
graph.add_command(root_command)
for write_cmd, compute_cmd, read_cmd in channels_commands:
    graph.add_command(write_cmd)
    graph.add_command(compute_cmd)
    graph.add_command(read_cmd)
graph.root_command = root_command
graph.build_graph()

# Create hardware resources
host = SimHost()
channels = [SimPIMChannel() for _ in range(5)]

# Create executor and run
executor = GraphExecuteEngine(graph, host, channels)

SimSession.scheduler.run()

print(f"Simulation Finished at time {SimSession.sim_time}")
