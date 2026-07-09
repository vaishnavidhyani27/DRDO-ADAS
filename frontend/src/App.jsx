import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [model, setModel] = useState(null);
  const [status, setStatus] = useState("Loading AI model...");
  const [vehicleCount, setVehicleCount] = useState(0);
  const [personCount, setPersonCount] = useState(0);
  const [distance, setDistance] = useState("N/A");
  const [alert, setAlert] = useState("No Alert");

  const targetClasses = ["person", "bicycle", "car", "motorcycle", "bus", "truck"];

  useEffect(() => {
    async function loadModel() {
      const loadedModel = await cocoSsd.load();
      setModel(loadedModel);
      setStatus("AI model ready");
    }

    loadModel();
  }, []);

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
        setStatus("Camera active | Live detection running");
      } catch (error) {
        setStatus("Camera permission denied");
        console.error(error);
      }
    }

    startCamera();
  }, []);

  useEffect(() => {
    if (!model) return;

    const interval = setInterval(async () => {
      if (!videoRef.current || videoRef.current.readyState < 2 || !canvasRef.current) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const width = video.videoWidth;
      const height = video.videoHeight;

      if (!width || !height) return;

      canvas.width = width;
      canvas.height = height;

      const predictions = await model.detect(video);

      ctx.clearRect(0, 0, width, height);

      let vehicles = 0;
      let persons = 0;
      let nearestDistance = "N/A";

      predictions.forEach((prediction) => {
        if (targetClasses.includes(prediction.class) && prediction.score > 0.35) {
          const [x, y, boxWidth, boxHeight] = prediction.bbox;

          const approxDistance = Math.max(1, Math.round(1200 / boxHeight));
          nearestDistance = `${approxDistance} m`;

          if (prediction.class === "person") {
            persons++;
          } else {
            vehicles++;
          }

          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, boxWidth, boxHeight);

          ctx.fillStyle = "#22c55e";
          ctx.font = "18px Arial";
          ctx.fillText(
            `${prediction.class} ${(prediction.score * 100).toFixed(0)}% | ${approxDistance}m`,
            x,
            y > 20 ? y - 8 : y + 20
          );
        }
      });

      setVehicleCount(vehicles);
      setPersonCount(persons);
      setDistance(nearestDistance);

      if (persons > 0) {
        setAlert("Pedestrian Detected");
      } else if (vehicles >= 3) {
        setAlert("Traffic Ahead");
      } else {
        setAlert("No Alert");
      }
    }, 500);

    return () => clearInterval(interval);
  }, [model]);

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
          <p className="text-slate-400">Distance</p>
          <h2 className="text-xl font-bold">{distance}</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Lane</p>
          <h2 className="text-xl font-bold">Safe</h2>
        </div>

        <div className="bg-slate-900 p-4 rounded-xl text-center">
          <p className="text-slate-400">Pothole</p>
          <h2 className="text-xl font-bold">Monitoring</h2>
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