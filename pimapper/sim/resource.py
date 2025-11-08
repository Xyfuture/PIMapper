from typing import Optional
from Desim.Sync import SimOrderedSemaphore


class ResourceBase:
    pass


class SimHost(ResourceBase):
    def __init__(self):
        self.vector_compute: Optional[SimOrderedSemaphore] = SimOrderedSemaphore(1)


class SimPIMChannel(ResourceBase):
    def __init__(self):
        self.host_channel_link: Optional[SimOrderedSemaphore] = SimOrderedSemaphore(1)
        self.channel_compute: Optional[SimOrderedSemaphore] = SimOrderedSemaphore(1)
