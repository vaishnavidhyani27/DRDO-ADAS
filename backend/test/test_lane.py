from pathlib import Path
import sys

import cv2
import numpy as np


# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from detectors.lane_detector import LaneDetector


# ---------------------------------------------------------
# INPUT AND OUTPUT PATHS
# ---------------------------------------------------------

INPUT_VIDEO = (
    BACKEND_DIR
    / "third_party"
    / "ufldv2"
    / "example.mp4"
)

OUTPUT_VIDEO = (
    TEST_DIR
    / "outputs"
    / "lane_test.mp4"
)


# ---------------------------------------------------------
# DRAWING FUNCTION
# ---------------------------------------------------------

def draw_lane_result(
    frame: np.ndarray,
    result: dict,
) -> np.ndarray:
    output_frame = frame.copy()

    lanes = result.get("lanes", [])

    # Draw continuous lane boundary lines.
    for lane in lanes:
        if len(lane) < 2:
            continue

        points = np.asarray(
            lane,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            output_frame,
            [points],
            isClosed=False,
            color=(0, 255, 0),
            thickness=5,
        )

    # Fill detected driving-lane region.
    lane_polygon = result.get(
        "lane_polygon",
        [],
    )

    if len(lane_polygon) >= 4:
        polygon_points = np.asarray(
            lane_polygon,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        lane_overlay = output_frame.copy()

        cv2.fillPoly(
            lane_overlay,
            [polygon_points],
            color=(0, 180, 0),
        )

        output_frame = cv2.addWeighted(
            lane_overlay,
            0.25,
            output_frame,
            0.75,
            0,
        )

    # Draw camera centre.
    frame_height, frame_width = (
        output_frame.shape[:2]
    )

    camera_center_x = frame_width // 2

    cv2.line(
        output_frame,
        (camera_center_x, frame_height),
        (
            camera_center_x,
            int(frame_height * 0.55),
        ),
        color=(255, 255, 0),
        thickness=2,
    )

    # Draw calculated lane centre.
    lane_center = result.get("lane_center")

    if lane_center is not None:
        lane_center_x = int(lane_center)

        cv2.line(
            output_frame,
            (lane_center_x, frame_height),
            (
                lane_center_x,
                int(frame_height * 0.55),
            ),
            color=(255, 0, 255),
            thickness=3,
        )

    lane_status = result.get(
        "lane_status",
        "Lane Not Detected",
    )

    lane_departure = result.get(
        "lane_departure",
        False,
    )

    status_color = (
        (0, 0, 255)
        if lane_departure
        else (0, 255, 0)
    )

    cv2.putText(
        output_frame,
        lane_status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3,
        cv2.LINE_AA,
    )

    offset_ratio = result.get(
        "offset_ratio"
    )

    if offset_ratio is not None:
        cv2.putText(
            output_frame,
            f"Lane offset: {offset_ratio:.3f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


# ---------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------

def main() -> None:
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found: {INPUT_VIDEO}"
        )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading UFLDv2 lane detector...")

    detector = LaneDetector()

    print(f"Opening video: {INPUT_VIDEO}")

    capture = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {INPUT_VIDEO}"
        )

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 20.0

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Unable to create output video: "
            f"{OUTPUT_VIDEO}"
        )

    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            lane_result = detector.detect(
                frame
            )

            output_frame = draw_lane_result(
                frame,
                lane_result,
            )

            writer.write(output_frame)

            frame_number += 1

            if frame_number % 10 == 0:
                print(
                    f"Processed "
                    f"{frame_number} frames"
                )

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print()
    print("Lane detection test completed.")
    print(
        f"Total processed frames: "
        f"{frame_number}"
    )
    print(
        f"Output saved to: "
        f"{OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()