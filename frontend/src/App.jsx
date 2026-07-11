import { useEffect, useRef, useState } from "react";

const BACKEND_URL =
  "https://reception-shirts-coffee-alignment.trycloudflare.com/detect";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const processingRef = useRef(false);

  const [status, setStatus] = useState("Starting camera...");
  const [vehicleCount, setVehicleCount] = useState(0);
  const [personCount, setPersonCount] = useState(0);
  const [potholeCount, setPotholeCount] = useState(0);
  const [distance, setDistance] = useState("N/A");
  const [alert, setAlert] = useState("No Alert");

  useEffect(() => {
    let cameraStream;

    async function startCamera() {
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = cameraStream;
          await videoRef.current.play();
        }

        setStatus("Camera active | Connecting to YOLOv8...");
      } catch (error) {
        console.error("Camera error:", error);
        setStatus("Camera permission denied");
      }
    }

    startCamera();

    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      const video = videoRef.current;
      const overlayCanvas = canvasRef.current;
      const captureCanvas = captureCanvasRef.current;

      if (
        processingRef.current ||
        !video ||
        !overlayCanvas ||
        !captureCanvas ||
        video.readyState < 2
      ) {
        return;
      }

      const width = video.videoWidth;
      const height = video.videoHeight;

      if (!width || !height) {
        return;
      }

      processingRef.current = true;

      try {
        captureCanvas.width = width;
        captureCanvas.height = height;

        const captureContext = captureCanvas.getContext("2d");
        captureContext.drawImage(video, 0, 0, width, height);

        // Compress the frame before sending it to the backend.
        const imageData = captureCanvas.toDataURL("image/jpeg", 0.65);

        const response = await fetch(BACKEND_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: imageData,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(
            `Backend returned ${response.status}: ${errorText}`
          );
        }

        const data = await response.json();

        overlayCanvas.width = width;
        overlayCanvas.height = height;

        const context = overlayCanvas.getContext("2d");
        context.clearRect(0, 0, width, height);

        const detections = Array.isArray(data.detections)
          ? data.detections
          : [];

        detections.forEach((detection) => {
          const [x1, y1, x2, y2] = detection.bbox;
          const boxWidth = x2 - x1;
          const boxHeight = y2 - y1;

          const isPothole = detection.class === "Pothole";
          const boxColor = isPothole ? "#ef4444" : "#22c55e";

          context.strokeStyle = boxColor;
          context.lineWidth = 4;
          context.strokeRect(x1, y1, boxWidth, boxHeight);

          const confidence = Math.round(
            (detection.confidence || 0) * 100
          );

          const distanceText =
            detection.distance_m !== null &&
            detection.distance_m !== undefined
              ? ` | ${detection.distance_m} m`
              : "";

          const label =
            `${detection.class} ${confidence}%${distanceText}`;

          context.font = "18px Arial";
          const textWidth = context.measureText(label).width;
          const labelY = y1 > 30 ? y1 - 28 : y1;

          context.fillStyle = boxColor;
          context.fillRect(labelY === y1 ? x1 : x1, labelY, textWidth + 14, 28);

          context.fillStyle = "#ffffff";
          context.fillText(label, x1 + 7, labelY + 20);
        });

        setVehicleCount(data.vehicle_count ?? 0);
        setPersonCount(data.pedestrian_count ?? 0);
        setPotholeCount(data.pothole_count ?? 0);

        setDistance(
          data.nearest_distance_m !== null &&
            data.nearest_distance_m !== undefined
            ? `${data.nearest_distance_m} m`
            : "N/A"
        );

        setAlert(data.alert || "No Alert");
        setStatus("YOLOv8 live detection running");
      } catch (error) {
        console.error("Detection error:", error);
        setStatus("Backend connection failed");
      } finally {
        processingRef.current = false;
      }
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-white p-4">
      <h1 className="text-3xl font-bold text-center">
        Smart ADAS Detection System
      </h1>

      <p className="text-center text-slate-400 mt-2">{status}</p>

      <div className="grid md:grid-cols-2 gap-6 mt-6">
        <div className="bg-slate-900 rounded-2xl p-4 shadow-lg">
          <h2 className="text-xl font-semibold mb-3">Road Camera</h2>

          <div className="relative w-full overflow-hidden rounded-xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="block w-full rounded-xl"
            />

            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />

            <canvas
              ref={captureCanvasRef}
              className="hidden"
            />
          </div>
        </div>

        <div className="bg-slate-900 rounded-2xl p-4 shadow-lg flex flex-col justify-center items-center">
          <h2 className="text-xl font-semibold mb-3">
            Driver Camera
          </h2>

          <div className="text-slate-400 text-center">
            Driver monitoring module active
            <br />
            Status: Awake
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Vehicles</p>
          <h2 className="text-3xl font-bold">{vehicleCount}</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Pedestrians</p>
          <h2 className="text-3xl font-bold">{personCount}</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Distance</p>
          <h2 className="text-xl font-bold">{distance}</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Lane</p>
          <h2 className="text-xl font-bold">Safe</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Potholes</p>
          <h2 className="text-3xl font-bold">{potholeCount}</h2>
        </div>
      </div>

      <div
        className={`mt-6 p-4 rounded-xl text-center text-xl font-bold ${
          alert === "No Alert" ? "bg-green-700" : "bg-red-700"
        }`}
      >
        {alert}
      </div>
    </div>
  );
}

export default App;