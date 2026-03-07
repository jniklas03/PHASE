from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from enum import Enum, auto

@dataclass
class Colony:
    centroid: tuple[int, int]
    radius: float
    label: int
    state: str = "temp"
    age: int = 1
    expansion_rate: float = 0.0

    # Kalman filter fields
    kf_radius: float = field(init=False)
    P: float = 1.0      # uncertainty in prediction
    Q: float = 0.05      # process noise
    R: float = 0.6      # measurment noise

    def __post_init__(self):
        # after detection sets predicted radius = detected radius
        self.kf_radius = self.radius


    def kalman_predict(self):
        # predicts change in radius
        self.kf_radius += self.expansion_rate
        # uncertainty grows by process noise before seeing the measurment
        self.P += self.Q 

    def kalman_update(self, measurement: float):
        # K determines how much we trust the measurement vs prediction
        # if P > R, we trust measurement more -> K ~ 1
        # if P < R, we trust prediction more -> K ~ 0
        K = self.P / (self.P + self.R)

        # predicted radius is updated by measurement, weighted by K
        self.kf_radius += K * (measurement - self.kf_radius)

        # uncertainty reduced after incorporating measurement (opposite of kalman_predict)
        self.P *= (1 - K)

    @staticmethod
    def cost_function_iou_circle(prev_colony: Colony, curr_blob) -> float:
        c1 = prev_colony.centroid
        r1 = prev_colony.radius

        c2 = curr_blob.pt
        r2 = curr_blob.size / 2

        x1, y1 = c1
        x2, y2 = c2
        d = math.hypot(x2 - x1, y2 - y1)

        if d >= r1 + r2:
            inter_area = 0  # no overlap
        elif d <= abs(r1 - r2):
            inter_area = math.pi * min(r1, r2)**2  # one circle fully inside the other
        else:
            r1_sq = r1**2
            r2_sq = r2**2

            alpha = math.acos((d**2 + r1_sq - r2_sq) / (2 * d * r1))
            beta  = math.acos((d**2 + r2_sq - r1_sq) / (2 * d * r2))

            inter_area = r1_sq * alpha + r2_sq * beta - 0.5 * math.sqrt(
                (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
            )

        union_area = math.pi * r1**2 + math.pi * r2**2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        cost = 1 - iou

        return cost
    
    @staticmethod
    def cost_function_iou_box(prev_colony: Colony, curr_blob) -> float:
        x1_min, y1_min = prev_colony.centroid[0] - prev_colony.radius, prev_colony.centroid[1] - prev_colony.radius
        x1_max, y1_max = prev_colony.centroid[0] + prev_colony.radius, prev_colony.centroid[1] + prev_colony.radius

        x2_min, y2_min = int(curr_blob.pt[0]) - int(curr_blob.size/2), int(curr_blob.pt[1]) - int(curr_blob.size/2)
        x2_max, y2_max = int(curr_blob.pt[0]) + int(curr_blob.size/2), int(curr_blob.pt[1]) + int(curr_blob.size/2)

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area1 + area2 - inter_area

        iou = inter_area / union if union > 0 else 0

        cost = 1 - iou

        return cost
    
    @staticmethod
    def cost_function_distance(prev_colony: Colony, curr_blob) -> float:
        px, py = prev_colony.centroid
        cx, cy = int(curr_blob.pt[0]), int(curr_blob.pt[1])

        cost = np.sqrt((px - cx)**2 + (py - cy)**2)

        return cost

class CostFunction(Enum):
    IOU_CIRCLE = staticmethod(Colony.cost_function_iou_circle)
    IOU_BOX = staticmethod(Colony.cost_function_iou_box)
    DISTANCE = staticmethod(Colony.cost_function_distance)

    def __call__(self, prev_colony, curr_blob):
        return self.value(prev_colony, curr_blob)