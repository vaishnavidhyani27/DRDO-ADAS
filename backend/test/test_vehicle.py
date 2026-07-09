import cv2
import sys
import os


# Add backend folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectors.vehicle_detector import VehicleDetector


def main():
    detector = VehicleDetector()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✅ Webcam started")
    print("Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to read frame")
            break

        frame_count += 1

        # Run YOLO every 2nd frame for better speed + accuracy balance
        if frame_count % 2 == 0:
            frame, detections = detector.detect(frame)

        cv2.imshow("ADAS Vehicle + Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
