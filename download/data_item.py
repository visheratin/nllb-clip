from dataclasses import dataclass


@dataclass
class DataItem:
    id: str
    url: str
    caption: str
    score: float
