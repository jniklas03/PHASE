from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
from enum import Enum
from scipy import ndimage as ndi

@dataclass
class Colony:
    centroid: tuple[int, int]
    radius: float
    label: int
    expansion_rate: float = 0.0
    state: str = "temp"
    age: int = 1
    missed_frames: int = 0

    # Kalman state vector [x, y, vx, vy, r, vr]
    x: np.ndarray = field(init=False)

    # Covariance
    P: np.ndarray = field(default_factory=lambda: np.diag([1, 1, 5, 5, 1, 2]))

    # Process noise
    Q: np.ndarray = field(default_factory=lambda: np.diag([0.05, 0.05, 0.2, 0.2, 0.1, 0.05]))

    # Measurement noise
    R: np.ndarray = field(default_factory=lambda: np.diag([0.5, 0.5, 0.3]))

    def __post_init__(self):

        # state vector
        self.x = np.array([
            self.centroid[0],
            self.centroid[1],
            0.0,                     # vx
            0.0,                     # vy
            self.radius,
            self.expansion_rate      # vr
        ], dtype=float)


    def predict(self, dt: float = 1.0) -> Colony:
        # state transition matrix
        F = np.array([
            [1,0,dt,0,0,0],
            [0,1,0,dt,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,dt],
            [0,0,0,0,0,1]
        ])

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self.Q

        new_col = Colony(
            centroid=(int(x_pred[0]), int(x_pred[1])),
            radius=float(x_pred[4]),
            label=self.label,
            expansion_rate=float(x_pred[5]),
            state=self.state,
            age=self.age+1,
            missed_frames=self.missed_frames
        )

        new_col.x = x_pred
        new_col.P = P_pred
        new_col.Q = self.Q
        new_col.R = self.R

        return new_col

    def update(self, measured_centroid, measured_radius):

        z = np.array([
            measured_centroid[0],
            measured_centroid[1],
            measured_radius
        ])

        # measurement matrix
        H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0]
        ])

        y = z - H @ self.x

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

        # sync public attributes
        self.centroid = (int(self.x[0]), int(self.x[1]))
        self.radius = float(self.x[4])
        self.expansion_rate = float(self.x[5])


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
    
    @staticmethod
    def convert_segments_to_colonies(labels):
        centroids = []
        radii = []

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        for lab in unique_labels:
            mask = labels == lab
            y, x = ndi.center_of_mass(mask)
            area = np.sum(mask)
            radius = np.sqrt(area / np.pi)

            centroids.append((x,y))
            radii.append(radius)

        return centroids, radii

class CostFunction(Enum):
    IOU_CIRCLE = staticmethod(Colony.cost_function_iou_circle)
    IOU_BOX = staticmethod(Colony.cost_function_iou_box)
    DISTANCE = staticmethod(Colony.cost_function_distance)

    def __call__(self, prev_colony, curr_blob):
        return self.value(prev_colony, curr_blob)