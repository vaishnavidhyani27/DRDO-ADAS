import { useEffect, useRef, useState } from "react";

const BACKEND_URL = "https://drdo-adas-backend.onrender.com/detect";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const [status, setStatus] = useState("Starting camera...");
  const [vehicleCount, setVehicleCount] = useState(0);
  const [personCount, setPersonCount] = useState(0);
  const [potholeCount, setPotholeCount] = useState(0);
  const [distance, setDistance] = useState("N/A");
  const [alert, setAlert] = useState("No Alert");

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });

        videoRef.current.srcObject = stream;
        setStatus("Camera active | YOLOv8 backend connected");
      } catch (error) {
        setStatus("Camera permission denied");
        console.error(error);
      }
    }

    startCamera();
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (
        !videoRef.current ||
        videoRef.current.readyState !== 4 ||
        !canvasRef.current ||
        !hiddenCanvasRef.current
      ) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const hiddenCanvas = hiddenCanvasRef.current;

      const width = video.videoWidth;
      const height = video.videoHeight;

      if (!width || !height) return;

      canvas.width = width;
      canvas.height = height;
      hiddenCanvas.width = width;
      hiddenCanvas.height = height;

      const hiddenCtx = hiddenCanvas.getContext("2d");
      hiddenCtx.drawImage(video, 0, 0, width, height);

      const imageData = hiddenCanvas.toDataURL("image/jpeg", 0.6);

      try {
        const response = await fetch(BACKEND_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: imageData,
          }),
        });

        const data = await response.json();

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, width, height);

        let vehicles = 0;
        let persons = 0;
        let potholes = 0;

        if (data.detections) {
          data.detections.forEach((det) => {
            const [x1, y1, x2, y2] = det.bbox;
            const boxWidth = x2 - x1;
            const boxHeight = y2 - y1;

            if (det.class === "Person") {
              persons++;
            } else if (det.class === "Pothole") {
              potholes++;
            } else {
              vehicles++;
            }

            const color = det.class === "Pothole" ? "#ef4444" : "#22c55e";

            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, boxWidth, boxHeight);

            ctx.fillStyle = color;
            ctx.font = "18px Arial";
            ctx.fillText(
              `${det.class} ${(det.confidence * 100).toFixed(0)}% | ${det.distance}`,
              x1,
              y1 > 20 ? y1 - 8 : y1 + 20
            );
          });
        }

        setVehicleCount(vehicles);
        setPersonCount(persons);
        setPotholeCount(potholes);
        setDistance(data.distance || "N/A");
        setAlert(data.alert || "No Alert");
        setStatus("YOLOv8 detection running");
      } catch (error) {
        console.error(error);
        setStatus("Backend connection failed");
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

          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full rounded-xl object-contain"
            />

            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
            />

            <canvas ref={hiddenCanvasRef} className="hidden" />
          </div>
        </div>

        <div className="bg-slate-900 rounded-2xl p-4 shadow-lg flex flex-col justify-center items-center">
          <h2 className="text-xl font-semibold mb-3">Driver Camera</h2>
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
          <p className="text-slate-400">Potholes</p>
          <h2 className="text-3xl font-bold">{potholeCount}</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Distance</p>
          <h2 className="text-xl font-bold">{distance}</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Lane</p>
          <h2 className="text-xl font-bold">Safe</h2>
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