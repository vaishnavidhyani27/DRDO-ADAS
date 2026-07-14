import { useEffect, useRef, useState } from "react";

const BACKEND_URL =
  "https://newport-vendors-favorite-trek.trycloudflare.com/detect";

const REQUEST_INTERVAL_MS = 2500;
const AUDIO_COOLDOWN_MS = 6000;

function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const processingRef = useRef(false);
  const lastAlertRef = useRef("");
  const lastSpokenTimeRef = useRef(0);

  const [status, setStatus] = useState("Starting camera...");
  const [vehicleCount, setVehicleCount] = useState(0);
  const [personCount, setPersonCount] = useState(0);
  const [potholeCount, setPotholeCount] = useState(0);
  const [distance, setDistance] = useState("N/A");
  const [laneStatus, setLaneStatus] = useState("Lane Not Detected");
  const [wrongWay, setWrongWay] = useState(false);
  const [alert, setAlert] = useState("No Alert");
  const [audioEnabled, setAudioEnabled] = useState(false);

  function enableAudio() {
    if (!window.speechSynthesis) {
      setStatus("Audio alerts are not supported");
      return;
    }

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(
      "Audio alerts enabled"
    );

    speech.rate = 1;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);
    setAudioEnabled(true);
    setStatus("Audio alerts enabled");
  }

  function drawPath(context, points, color, width = 5) {
    if (!Array.isArray(points) || points.length < 2) return;

    context.beginPath();
    context.moveTo(points[0][0], points[0][1]);

    for (let i = 1; i < points.length; i += 1) {
      context.lineTo(points[i][0], points[i][1]);
    }

    context.strokeStyle = color;
    context.lineWidth = width;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.stroke();
  }

  function drawLaneOverlay(context, data) {
    const leftLane = Array.isArray(data.left_lane)
      ? data.left_lane
      : [];

    const rightLane = Array.isArray(data.right_lane)
      ? data.right_lane
      : [];

    const polygon = Array.isArray(data.lane_polygon)
      ? data.lane_polygon
      : [];

    const laneColor = data.lane_departure
      ? "#ef4444"
      : "#facc15";

    if (polygon.length >= 4) {
      context.save();
      context.beginPath();
      context.moveTo(polygon[0][0], polygon[0][1]);

      for (let i = 1; i < polygon.length; i += 1) {
        context.lineTo(polygon[i][0], polygon[i][1]);
      }

      context.closePath();
      context.fillStyle = data.lane_departure
        ? "rgba(239, 68, 68, 0.18)"
        : "rgba(34, 197, 94, 0.16)";
      context.fill();
      context.restore();
    }

    drawPath(context, leftLane, laneColor);
    drawPath(context, rightLane, laneColor);
  }

  function drawDetections(context, detections) {
    detections.forEach((detection) => {
      if (
        !Array.isArray(detection.bbox) ||
        detection.bbox.length !== 4
      ) {
        return;
      }

      const [x1, y1, x2, y2] = detection.bbox;
      const isPothole = detection.class === "Pothole";
      const color = isPothole ? "#ef4444" : "#22c55e";

      context.strokeStyle = color;
      context.lineWidth = 4;
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const confidence = Math.round(
        (detection.confidence || 0) * 100
      );

      const distanceText =
        !isPothole &&
        detection.distance_m !== null &&
        detection.distance_m !== undefined
          ? ` | ${detection.distance_m} m`
          : "";

      const label =
        `${detection.class} ${confidence}%${distanceText}`;

      context.font = "18px Arial";

      const textWidth = context.measureText(label).width;
      const labelY = y1 > 32 ? y1 - 30 : y1;

      context.fillStyle = color;
      context.fillRect(x1, labelY, textWidth + 14, 28);

      context.fillStyle = "#ffffff";
      context.fillText(label, x1 + 7, labelY + 20);
    });
  }

  function speakAlert(currentAlert) {
    if (
      !audioEnabled ||
      !window.speechSynthesis ||
      currentAlert === "No Alert"
    ) {
      return;
    }

    const now = Date.now();
    const alertChanged =
      currentAlert !== lastAlertRef.current;

    const cooldownExpired =
      now - lastSpokenTimeRef.current >= AUDIO_COOLDOWN_MS;

    if (!alertChanged && !cooldownExpired) return;

    window.speechSynthesis.cancel();

    const speech = new SpeechSynthesisUtterance(
      currentAlert
    );

    speech.rate = 1;
    speech.volume = 1;

    window.speechSynthesis.speak(speech);

    lastAlertRef.current = currentAlert;
    lastSpokenTimeRef.current = now;
  }

  useEffect(() => {
    let stream;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        setStatus("Camera active | Connecting to ADAS...");
      } catch (error) {
        console.error("Camera error:", error);
        setStatus("Camera permission denied");
      }
    }

    startCamera();

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
      window.speechSynthesis?.cancel();
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      const capture = captureRef.current;

      if (
        processingRef.current ||
        !video ||
        !overlay ||
        !capture ||
        video.readyState < 2
      ) {
        return;
      }

      const width = video.videoWidth;
      const height = video.videoHeight;

      if (!width || !height) return;

      processingRef.current = true;

      try {
        const outputWidth = 960;
        const outputHeight = 540;
        const targetRatio = outputWidth / outputHeight;
        const sourceRatio = width / height;

        let sourceX = 0;      
        let sourceY = 0;
        let sourceWidth = width;
        let sourceHeight = height;

        if (sourceRatio > targetRatio) {
          sourceWidth = height * targetRatio;
          sourceX = (width - sourceWidth) / 2;
        } else {
          sourceHeight = width / targetRatio;
          sourceY = (height - sourceHeight) / 2;
          }

        capture.width = outputWidth;
        capture.height = outputHeight;

        const captureContext = capture.getContext("2d");

if (!captureContext) {
  throw new Error("Capture canvas unavailable");
}

captureContext.drawImage(
  video,
  sourceX,
  sourceY,
  sourceWidth,
  sourceHeight,
  0,
  0,
  outputWidth,
  outputHeight
);

const image = capture.toDataURL("image/jpeg", 0.82);


        const response = await fetch(BACKEND_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ image }),
        });

        if (!response.ok) {
          const message = await response.text();

          throw new Error(
            `Backend ${response.status}: ${message}`
          );
        }

        const data = await response.json();

        overlay.width = width;
        overlay.height = height;

        const context = overlay.getContext("2d");

        if (!context) {
          throw new Error("Overlay canvas unavailable");
        }

        context.clearRect(0, 0, width, height);

        drawLaneOverlay(context, data);

        drawDetections(
          context,
          Array.isArray(data.detections)
            ? data.detections
            : []
        );

        setVehicleCount(data.vehicle_count ?? 0);
        setPersonCount(data.pedestrian_count ?? 0);
        setPotholeCount(data.pothole_count ?? 0);

        const nearestVehicle =
          data.nearest_vehicle_distance_m ??
          data.nearest_distance_m;

        setDistance(
          nearestVehicle !== null &&
          nearestVehicle !== undefined
            ? `${nearestVehicle} m`
            : "N/A"
        );

        setLaneStatus(
          data.lane_status || "Lane Not Detected"
        );

        setWrongWay(Boolean(data.wrong_way));

        const currentAlert = data.alert || "No Alert";

        setAlert(currentAlert);
        speakAlert(currentAlert);

        if (currentAlert === "No Alert") {
          lastAlertRef.current = "";
        }

        setStatus(
          audioEnabled
            ? "Integrated ADAS running | Audio enabled"
            : "Integrated ADAS running"
        );
      } catch (error) {
        console.error("Detection error:", error);
        setStatus("Backend connection failed");
      } finally {
        processingRef.current = false;
      }
    }, REQUEST_INTERVAL_MS);

    return () => clearInterval(interval);
        }, [audioEnabled]);

  const laneClass = laneStatus.includes("Departure")
    ? "text-red-400"
    : laneStatus.includes("Safe")
      ? "text-green-400"
      : "text-yellow-400";

  return (
    <div className="min-h-screen bg-slate-950 p-4 text-white">
      <h1 className="text-center text-3xl font-bold">
        Smart ADAS Detection System
      </h1>

      <p className="mt-2 text-center text-slate-400">
        {status}
      </p>

      <div className="mt-3 flex justify-center">
        <button
          type="button"
          onClick={enableAudio}
          className={`rounded-xl px-5 py-2 font-semibold ${
            audioEnabled
              ? "bg-green-700"
              : "bg-blue-600"
          }`}
        >
          {audioEnabled
            ? "Audio Alerts Enabled"
            : "Enable Audio Alerts"}
        </button>
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <div className="rounded-2xl bg-slate-900 p-4 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold">
            Road Camera
          </h2>

          <div className="relative w-full overflow-hidden rounded-xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="block w-full rounded-xl"
            />

            <canvas
              ref={overlayRef}
              className="pointer-events-none absolute inset-0 h-full w-full"
            />

            <canvas
              ref={captureRef}
              className="hidden"
            />
          </div>
        </div>

        <div className="flex flex-col items-center justify-center rounded-2xl bg-slate-900 p-4 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold">
            Driver Camera
          </h2>

          <div className="text-center text-slate-400">
            Driver-monitoring module pending
            <br />
            Status: Not connected
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-6">
        <StatusCard label="Vehicles" value={vehicleCount} />
        <StatusCard label="Pedestrians" value={personCount} />
        <StatusCard label="Nearest Vehicle" value={distance} />

        <StatusCard
          label="Lane"
          value={laneStatus}
          valueClass={laneClass}
        />

        <StatusCard label="Potholes" value={potholeCount} />

        <StatusCard
          label="Wrong Way"
          value={wrongWay ? "Detected" : "Clear"}
          valueClass={
            wrongWay
              ? "text-red-400"
              : "text-green-400"
          }
        />
      </div>

      <div
        className={`mt-6 rounded-xl p-4 text-center text-xl font-bold ${
          alert === "No Alert"
            ? "bg-green-700"
            : "bg-red-700"
        }`}
      >
        {alert}
      </div>
    </div>
  );
}

function StatusCard({
  label,
  value,
  valueClass = "",
}) {
  return (
    <div className="rounded-xl bg-slate-900 p-4 text-center">
      <p className="text-slate-400">{label}</p>

      <h2
        className={`mt-1 text-xl font-bold ${valueClass}`}
      >
        {value}
      </h2>
    </div>
  );
}

export default App;