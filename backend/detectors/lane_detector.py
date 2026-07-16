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
    """
    Hybrid lane detector.
 
    Primary path:  UFLDv2 (row-anchor based deep model)
    Fallback path: classical OpenCV Canny + Hough pipeline
 
    The fallback is used automatically whenever UFLDv2 throws, or
    whenever its output for a side fails a basic sanity check (too few
    points, points don't span enough of the frame, etc). Whichever side
    succeeds is kept; the API contract (return keys) is unchanged so
    app.py and the frontend drawLanes()/RoadCamera.jsx need no changes.
    """
 
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
 
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "UFLDv2 checkpoint/model key mismatch - "
                f"missing={len(load_result.missing_keys)} "
                f"unexpected={len(load_result.unexpected_keys)}. "
                "If these numbers are large, UFLDv2 predictions will be "
                "unreliable and the OpenCV fallback will carry most frames."
            )
 
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
 
        # ---- temporal smoothing state (persists across detect() calls) ----
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.smoothing_alpha = 0.55   # weight given to the newest frame
        self.max_jump_px = 120        # snap instead of blend past this jump
        self._left_miss = 0
        self._right_miss = 0
        self.max_miss_frames = 20      # frames before a stale lane is dropped
 
        # ---- lane-departure hysteresis state ----
        self._departure_streak = 0
        self._safe_streak = 0
        self._last_departure_state = False
        self._last_direction = None
        self.confirm_frames = 3       # consecutive frames needed to flip
 
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
    def _valid_lane(points, height, width):
        """Sanity-check a lane before trusting it (from either source)."""
        if len(points) < 3:
            return False
 
        pts = np.asarray(points, dtype=np.float32)
        y_span = pts[:, 1].max() - pts[:, 1].min()
 
        # A handful of points bunched together vertically is almost always
        # noise (a shadow edge, a zebra-crossing stripe, a stray Hough
        # segment) rather than a real lane boundary.
        if y_span < 0.12 * height:
            return False
 
        # Reject wild extrapolation outside the frame (can happen with a
        # bad polyfit on a near-vertical or sparse point cluster).
        if np.any(pts[:, 0] < -0.25 * width) or np.any(pts[:, 0] > 1.25 * width):
            return False
 
        return True
 
    @staticmethod
    def opencv_fallback(frame):
        height, width = frame.shape[:2]
 
        # Shadow/lighting-robust preprocessing: work in the L channel of
        # HLS and equalize contrast locally (CLAHE) so shadow boundaries
        # don't produce edges as strong as real lane markings.
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
 
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
 
        blur = cv2.GaussianBlur(l_eq, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
 
        mask = np.zeros_like(edges)
 
        # Narrower trapezoid: excludes sky at the top and the far
        # left/right shoulder area at the bottom, which is where curb
        # edges and zebra-crossing stripes tend to leak into the ROI.
        roi = np.array(
            [
                [
                    (int(width * 0.06), height),
                    (int(width * 0.34), int(height * 0.52)),
                    (int(width * 0.66), int(height * 0.52)),
                    (int(width * 0.96), height),
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
            threshold=24,
            minLineLength=max(20, int(height * 0.04)),
            maxLineGap=120,
        )
 
        left_segments = []
        right_segments = []
 
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                if x2 == x1:
                    continue
 
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < max(25.0, height * 0.035):
                    # Reject tiny fragments - Hough noise, not real markings.
                    continue
 
                slope = (y2 - y1) / (x2 - x1)
 
                # Zebra-crossing stripes and curb/road edges run close to
                # horizontal in the image; real lane markings don't.
                if abs(slope) < 0.25:
                    continue
 
                # Near-vertical artifacts (lamp posts, signage caught
                # inside the ROI) aren't lane markings either.
                if abs(slope) > 8:
                    continue
 
                if slope < 0:
                    left_segments.append([x1, y1, x2, y2, length])
                else:
                    right_segments.append([x1, y1, x2, y2, length])
 
        def build_lane(segments):
            if not segments:
                return []
 
            segments = np.asarray(segments, dtype=np.float32)
            weights = segments[:, 4]
            points = segments[:, :4].reshape(-1, 2)
            point_weights = np.repeat(weights, 2)
 
            x = points[:, 0]
            y = points[:, 1]
 
            # Use a quadratic fit (supports slightly curved roads) when
            # there's enough vertical spread of points to constrain it,
            # otherwise fall back to a straight line.
            degree = 2 if len(np.unique(y)) >= 6 else 1
 
            try:
                fit = np.polyfit(y, x, degree, w=point_weights)
            except (ValueError, np.linalg.LinAlgError, TypeError):
                try:
                    fit = np.polyfit(y, x, 1)
                except (ValueError, np.linalg.LinAlgError):
                    return []
 
            y_values = np.linspace(
                int(height * 0.52),
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
 
    def _smooth_lane(self, lane, previous):
        """Blend this frame's lane toward the previous frame's lane so a
        video/live stream doesn't flicker between detections."""
        if not lane:
            return list(previous) if previous else []
 
        if not previous:
            return lane
 
        lane_arr = np.asarray(lane, dtype=np.float32)
        prev_arr = np.asarray(previous, dtype=np.float32)
 
        blended = []
        for x, y in lane_arr:
            prev_x = float(np.interp(y, prev_arr[:, 1], prev_arr[:, 0]))
 
            if abs(prev_x - x) > self.max_jump_px:
                # Big jump (lane change, sharp curve, re-acquisition after
                # a miss) - trust the new detection instead of dragging it
                # toward a stale position.
                blended.append([int(x), int(y)])
            else:
                new_x = (
                    self.smoothing_alpha * x
                    + (1 - self.smoothing_alpha) * prev_x
                )
                blended.append([int(new_x), int(y)])
 
        return blended
 
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
            self._departure_streak = 0
            self._safe_streak = 0
            return {
                "lane_detected": False,
                "lane_status": "Lane Not Detected",
                "lane_departure": False,
                "departure_direction": None,
                "lane_center": None,
                "offset_ratio": None,
            }
 
        lane_width = right_x - left_x
 
        if lane_width < width * 0.05:
            # Left/right boundaries have effectively collapsed onto each
            # other - an unreliable reading, not a real lane departure.
            self._departure_streak = 0
            self._safe_streak = 0
            return {
                "lane_detected": False,
                "lane_status": "Lane Not Detected",
                "lane_departure": False,
                "departure_direction": None,
                "lane_center": None,
                "offset_ratio": None,
            }
 
        lane_center = (left_x + right_x) / 2
        offset = (width / 2 - lane_center) / lane_width
 
        raw_departure = abs(offset) > 0.22
        raw_direction = "left" if offset < 0 else "right"
 
        # Hysteresis: require several consecutive frames of agreement
        # before flipping the alert, so a single noisy frame doesn't fire
        # a false departure warning.
        if raw_departure:
            self._departure_streak += 1
            self._safe_streak = 0
        else:
            self._safe_streak += 1
            self._departure_streak = 0
 
        if self._departure_streak >= self.confirm_frames:
            departure = True
            direction = raw_direction
            self._last_departure_state = True
            self._last_direction = direction
        elif self._safe_streak >= self.confirm_frames:
            departure = False
            direction = None
            self._last_departure_state = False
            self._last_direction = None
        else:
            # Not enough consecutive agreement yet - hold the last
            # confirmed state instead of flapping frame to frame.
            departure = self._last_departure_state
            direction = self._last_direction
 
        if not departure:
            status = "Lane Safe"
        elif direction == "left":
            status = "Lane Departure Left"
        else:
            status = "Lane Departure Right"
 
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
 
        ufld_left, ufld_right = [], []
        try:
            with torch.inference_mode():
                prediction = self.model(self.preprocess(frame))
 
            ufld_left = self.decode_ufld_lane(
                prediction,
                lane_index=1,
                width=width,
                height=height,
            )
            ufld_right = self.decode_ufld_lane(
                prediction,
                lane_index=2,
                width=width,
                height=height,
            )
        except Exception as error:
            # UFLDv2 failed outright (bad frame, device error, etc) - the
            # OpenCV fallback below will carry this frame instead.
            print(f"UFLDv2 inference failed, using OpenCV fallback: {error}")
            ufld_left, ufld_right = [], []
 
        left_ok = self._valid_lane(ufld_left, height, width)
        right_ok = self._valid_lane(ufld_right, height, width)
 
        raw_left, raw_right = ufld_left, ufld_right
        method = "UFLDv2"
 
        if not (left_ok and right_ok):
            fallback_left, fallback_right = self.opencv_fallback(frame)
 
            fb_left_ok = self._valid_lane(fallback_left, height, width)
            fb_right_ok = self._valid_lane(fallback_right, height, width)
 
            if not left_ok:
                raw_left = fallback_left if fb_left_ok else []
            if not right_ok:
                raw_right = fallback_right if fb_right_ok else []
 
            used_fallback = (not left_ok and fb_left_ok) or (
                not right_ok and fb_right_ok
            )
            if used_fallback:
                method = "UFLDv2+OpenCV" if (left_ok or right_ok) else "OpenCV"
 
        # --- miss tracking: drop a stale smoothed lane if it hasn't been
        # freshly detected (by either method) for several frames in a row,
        # so a ghost lane doesn't linger on screen indefinitely.
        if raw_left:
            self._left_miss = 0
        else:
            self._left_miss += 1
            if self._left_miss > self.max_miss_frames:
                self.prev_left_lane = None
 
        if raw_right:
            self._right_miss = 0
        else:
            self._right_miss += 1
            if self._right_miss > self.max_miss_frames:
                self.prev_right_lane = None
 
        # --- temporal smoothing across frames (reduces flicker on video) ---
        left_lane = self._smooth_lane(raw_left, self.prev_left_lane)
        right_lane = self._smooth_lane(raw_right, self.prev_right_lane)
 
        self.prev_left_lane = left_lane if left_lane else self.prev_left_lane
        self.prev_right_lane = right_lane if right_lane else self.prev_right_lane
 
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