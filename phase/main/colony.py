from dataclasses import dataclass

@dataclass
class Colony:
    label: int | None
    centroid :tuple[int, int]
    radius: int
    growth_rate: float


