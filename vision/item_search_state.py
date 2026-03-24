from enum import Enum


class ItemSearchState(str, Enum):
    SEGMENT = "SEGMENT"
    CENTER_GUIDE = "CENTER_GUIDE"
    TRACK = "TRACK"
