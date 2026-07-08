import cv2
import sys
import os

# Add backend folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detectors.vehicle_detector import VehicleDetector


def main():

    detector = VehicleDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("✅ Webcam started")
    print("Press 'q' to quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame, detections = detector.detect(frame)

        cv2.imshow("ADAS Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()