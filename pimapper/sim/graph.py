from typing import Optional
from pimapper.core.instruction import CommandBase


class CommandGraph:
    def __init__(self):
        self.root_command: Optional[CommandBase] = None
        self.command_list: dict[CommandBase, None] = {}

    def add_command(self, command: CommandBase):
        self.command_list[command] = None

    def build_graph(self):
        # Auto-complete graph connections
        # By default only configure input_commands, this function auto-configures output relationships
        for command in self.command_list.keys():
            for producer_command in command.input_commands.keys():
                if producer_command.output_commands is None:
                    producer_command.output_commands = {}
                producer_command.output_commands[command] = None

        # Auto-complete prev and next relationships from input/output commands
        for command in self.command_list.keys():
            for producer_command in command.input_commands.keys():
                command.prev_commands[producer_command] = None

            for consumer_command in command.output_commands.keys():
                command.next_commands[consumer_command] = None

    def gen_dep_map(self) -> dict[CommandBase, int]:
        dep_map: dict[CommandBase, int] = {}
        for command in self.command_list.keys():
            dep_map[command] = len(command.prev_commands)
        return dep_map
