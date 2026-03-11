from .mc import MCCollection
from .off import OffCollection


class CollectionFactory:
    @classmethod
    def get_collection(cls, file_mask: str, emission_type: str):
        if emission_type.lower() in ('off', 'off-data'):
            return OffCollection(file_mask)
        else:
            return MCCollection(file_mask)