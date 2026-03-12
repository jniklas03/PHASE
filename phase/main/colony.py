from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from enum import Enum

@dataclass
class Colony:
    centroid: tuple[int, int]
    radius: float
    label: int
    state: str = "temp"
    age: int = 1
    expansion_rate: float = 0.0

    # Kalman filter fields
    kf_centroid: np.ndarray = field(init=False)
    kf_radius: float = field(init=False)
    kf_expansion_rate: float = field(init=False)
    P: np.ndarray = field(default_factory=lambda: np.eye(4))  # covariance for [x, y, radius, expansion_rate]
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.05, 0.2, 0.02]))  # process noise
    R: np.ndarray = field(default_factory=lambda: np.diag([0.4, 0.4, 0.02, 0.05]))  # measurement noise

    def __post_init__(self):
        self.kf_centroid = np.array(self.centroid, dtype=float)
        self.kf_radius = self.radius
        self.kf_expansion_rate = self.expansion_rate

    def kalman_predict(self):
        """
        Predict next state:
        - radius grows by expansion_rate
        - centroid assumed stationary
        - uncertainty grows by process noise Q
        """
        self.kf_radius += self.kf_expansion_rate
        self.P += self.Q

    def kalman_update(self, measured_radius: float, measured_centroid: tuple[int, int]):
        """
        Update state using measurements:
        - measured_radius: detected radius
        - measured_centroid: detected centroid
        - expansion_rate is smoothed implicitly
        """
        # measurement vector: [x, y, radius, dr]
        dr_measure = measured_radius - self.kf_radius
        z = np.array([measured_centroid[0], measured_centroid[1], measured_radius, dr_measure])

        # predicted state vector
        x_pred = np.array([self.kf_centroid[0], self.kf_centroid[1], self.kf_radius, self.kf_expansion_rate])

        # Kalman gain
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # state update
        x_new = x_pred + K @ (z - x_pred)

        self.kf_centroid = x_new[:2]
        self.kf_radius = x_new[2]
        self.kf_expansion_rate = x_new[3]

        # covariance update
        self.P = (np.eye(4) - K) @ self.P

        # sync standard attributes
        self.centroid = tuple(self.kf_centroid)
        self.radius = self.kf_radius
        self.expansion_rate = self.kf_expansion_rate

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