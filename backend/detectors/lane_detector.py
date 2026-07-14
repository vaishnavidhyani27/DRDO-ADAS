from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parents[1]
UFLD_DIR = BACKEND_DIR / "third_party" / "ufldv2"
CONFIG = UFLD_DIR / "configs" / "culane_res18.py"
WEIGHTS = UFLD_DIR / "weights" / "culane_res18.pth"

sys.path.insert(0, str(UFLD_DIR))


def initialize_weights(*models):
    for model in models:
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu",
                )
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


common_stub = types.ModuleType("utils.common")
common_stub.initialize_weights = initialize_weights
sys.modules["utils.common"] = common_stub

from model.model_culane import parsingNet
from utils.config import Config


class LaneDetector:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.cfg = Config.fromfile(str(CONFIG))
        self.row_anchor = np.linspace(
            0.42,
            1.0,
            self.cfg.num_row,
        )

        self.model = parsingNet(
            pretrained=False,
            backbone=self.cfg.backbone,
            num_grid_row=self.cfg.num_cell_row,
            num_cls_row=self.cfg.num_row,
            num_grid_col=self.cfg.num_cell_col,
            num_cls_col=self.cfg.num_col,
            num_lane_on_row=self.cfg.num_lanes,
            num_lane_on_col=self.cfg.num_lanes,
            use_aux=self.cfg.use_aux,
            input_height=self.cfg.train_height,
            input_width=self.cfg.train_width,
            fc_norm=self.cfg.fc_norm,
        ).to(self.device)

        checkpoint = torch.load(
            WEIGHTS,
            map_location=self.device,
        )

        state_dict = {
            key.removeprefix("module."): value
            for key, value in checkpoint["model"].items()
        }

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        resized_height = int(
            self.cfg.train_height / self.cfg.crop_ratio
        )

        self.transform = T.Compose(
            [
                T.Resize(
                    (
                        resized_height,
                        self.cfg.train_width,
                    )
                ),
                T.ToTensor(),
                T.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

        print(f"UFLDv2 lane detector loaded on {self.device}")

    def preprocess(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb))

        tensor = tensor[
            :,
            -self.cfg.train_height :,
            :,
        ]

        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def smooth(points):
        if len(points) < 4:
            return sorted(points, key=lambda point: point[1])

        points = np.asarray(points, dtype=np.float32)
        points = points[np.argsort(points[:, 1])]

        y = points[:, 1]
        x = points[:, 0]

        try:
            curve = np.polyfit(y, x, 2)
            smooth_y = np.linspace(y.min(), y.max(), 25)
            smooth_x = np.polyval(curve, smooth_y)

            return [
                [int(px), int(py)]
                for px, py in zip(smooth_x, smooth_y)
            ]
        except (ValueError, np.linalg.LinAlgError):
            return points.astype(int).tolist()

    def decode_ufld_lane(
        self,
        prediction,
        lane_index,
        width,
        height,
    ):
        locations = prediction["loc_row"].detach().cpu()
        existence = (
            prediction["exist_row"]
            .argmax(1)
            .detach()
            .cpu()
        )
        maxima = locations.argmax(1)

        _, grid_count, anchor_count, _ = locations.shape
        points = []

        for anchor_index in range(anchor_count):
            if not existence[0, anchor_index, lane_index]:
                continue

            maximum = int(
                maxima[0, anchor_index, lane_index]
            )

            start = max(0, maximum - 1)
            end = min(grid_count - 1, maximum + 1)

            indices = torch.arange(
                start,
                end + 1,
                dtype=torch.float32,
            )

            probabilities = locations[
                0,
                start : end + 1,
                anchor_index,
                lane_index,
            ].softmax(0)

            position = float(
                (probabilities * indices).sum() + 0.5
            )

            x = int(
                position
                / (grid_count - 1)
                * width
            )

            y = int(
                self.row_anchor[anchor_index]
                * height
            )

            if 0 <= x < width and 0 <= y < height:
                points.append([x, y])

        return self.smooth(points) if len(points) >= 3 else []

    @staticmethod
    def opencv_fallback(frame):
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        mask = np.zeros_like(edges)

        roi = np.array(
            [
                [
                    (int(width * 0.08), height),
                    (int(width * 0.43), int(height * 0.55)),
                    (int(width * 0.57), int(height * 0.55)),
                    (int(width * 0.92), height),
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, roi, 255)
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=35,
            minLineLength=35,
            maxLineGap=80,
        )

        left_segments = []
        right_segments = []

        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line

                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)

                if abs(slope) < 0.35:
                    continue

                if slope < 0:
                    left_segments.append(line)
                else:
                    right_segments.append(line)

        def build_lane(segments):
            if not segments:
                return []

            points = np.asarray(segments).reshape(-1, 2)
            x = points[:, 0]
            y = points[:, 1]

            try:
                fit = np.polyfit(y, x, 1)
            except (ValueError, np.linalg.LinAlgError):
                return []

            y_values = np.linspace(
                int(height * 0.58),
                height - 1,
                20,
            )

            x_values = np.polyval(fit, y_values)

            return [
                [
                    int(np.clip(px, 0, width - 1)),
                    int(py),
                ]
                for px, py in zip(x_values, y_values)
            ]

        return (
            build_lane(left_segments),
            build_lane(right_segments),
        )

    @staticmethod
    def x_at_y(points, target_y):
        if len(points) < 2:
            return None

        points = np.asarray(points, dtype=np.float32)
        points = points[np.argsort(points[:, 1])]

        y = points[:, 1]
        x = points[:, 0]

        target_y = np.clip(
            target_y,
            y.min(),
            y.max(),
        )

        return float(np.interp(target_y, y, x))

    def get_status(
        self,
        left_lane,
        right_lane,
        width,
        height,
    ):
        check_y = int(height * 0.88)

        left_x = self.x_at_y(left_lane, check_y)
        right_x = self.x_at_y(right_lane, check_y)

        if (
            left_x is None
            or right_x is None
            or left_x >= right_x
        ):
            return {
                "lane_detected": False,
                "lane_status": "Lane Not Detected",
                "lane_departure": False,
                "departure_direction": None,
                "lane_center": None,
                "offset_ratio": None,
            }

        lane_width = right_x - left_x
        lane_center = (left_x + right_x) / 2
        offset = (width / 2 - lane_center) / lane_width

        departure = abs(offset) > 0.22

        if not departure:
            status = "Lane Safe"
            direction = None
        elif offset < 0:
            status = "Lane Departure Left"
            direction = "left"
        else:
            status = "Lane Departure Right"
            direction = "right"

        return {
            "lane_detected": True,
            "lane_status": status,
            "lane_departure": departure,
            "departure_direction": direction,
            "lane_center": round(lane_center, 1),
            "offset_ratio": round(offset, 3),
        }

    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Empty lane frame")

        height, width = frame.shape[:2]

        with torch.inference_mode():
            prediction = self.model(
                self.preprocess(frame)
            )

        left_lane = self.decode_ufld_lane(
            prediction,
            lane_index=1,
            width=width,
            height=height,
        )

        right_lane = self.decode_ufld_lane(
            prediction,
            lane_index=2,
            width=width,
            height=height,
        )

        method = "UFLDv2"

        if not left_lane or not right_lane:
            fallback_left, fallback_right = (
                self.opencv_fallback(frame)
            )

            left_lane = left_lane or fallback_left
            right_lane = right_lane or fallback_right
            method = "UFLDv2 + OpenCV fallback"

        polygon = (
            left_lane + list(reversed(right_lane))
            if left_lane and right_lane
            else []
        )

        return {
            "left_lane": left_lane,
            "right_lane": right_lane,
            "lane_polygon": polygon,
            "lane_method": method,
            **self.get_status(
                left_lane,
                right_lane,
                width,
                height,
            ),
        }