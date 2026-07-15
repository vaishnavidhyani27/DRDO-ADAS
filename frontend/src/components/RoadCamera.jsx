import { useEffect, useRef, useState } from "react";

import { detectRoad } from "../utils/api";
import { drawDetections } from "../utils/drawDetections";
import { drawLanes } from "../utils/drawLanes";
import { resetSpeechAlert, speakAlert } from "../utils/speech";
import InfoCard from "./InfoCard";

const INTERVAL = 2200;

const INITIAL_DATA = {
  vehicles: 0,
  pedestrians: 0,
  potholes: 0,
  distance: "N/A",
  lane: "Lane Not Detected",
  wrongWay: false,
  alert: "No Alert",
};

export default function RoadCamera({
  audioEnabled,
  onAlert,
}) {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const videoUrlRef = useRef(null);
  const processingRef = useRef(false);

  const [mode, setMode] = useState("camera");
  const [status, setStatus] = useState(
    "Road camera not started"
  );
  const [data, setData] = useState(INITIAL_DATA);

  function stopSource() {
    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current);
      videoUrlRef.current = null;
    }

    const video = videoRef.current;

    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.load();
    }
  }

  async function startCamera() {
    stopSource();
    setMode("camera");
    setStatus("Starting road camera...");

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: "environment",
            },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

      streamRef.current = stream;

      const video = videoRef.current;

      if (video) {
        video.srcObject = stream;
        video.muted = true;
        await video.play();
      }

      setStatus("Live road camera active");
    } catch (error) {
      console.error(error);
      setStatus("Unable to start road camera");
    }
  }

  async function uploadVideo(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("video/")) {
      setStatus("Select a valid video file");
      return;
    }

    stopSource();
    setMode("video");

    const url = URL.createObjectURL(file);
    videoUrlRef.current = url;

    const video = videoRef.current;

    if (video) {
      video.src = url;
      video.loop = true;
      video.muted = true;

      try {
        await video.play();
        setStatus(`Processing ${file.name}`);
      } catch (error) {
        console.error(error);
        setStatus("Tap video to start playback");
      }
    }

    event.target.value = "";
  }

  useEffect(() => {
    startCamera();

    return () => {
      stopSource();
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(async () => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      const capture = captureRef.current;

      if (
        processingRef.current ||
        !video ||
        !overlay ||
        !capture ||
        video.readyState < 2 ||
        video.paused
      ) {
        return;
      }

      const width = video.videoWidth;
      const height = video.videoHeight;

      if (!width || !height) return;

      processingRef.current = true;

      try {
        capture.width = width;
        capture.height = height;

        const captureContext =
          capture.getContext("2d");

        captureContext.drawImage(
          video,
          0,
          0,
          width,
          height
        );

        const image = capture.toDataURL(
          "image/jpeg",
          0.8
        );

        const result = await detectRoad(image);

        overlay.width = width;
        overlay.height = height;

        const overlayContext =
          overlay.getContext("2d");

        overlayContext.clearRect(
          0,
          0,
          width,
          height
        );

        drawLanes(overlayContext, result);
        drawDetections(
          overlayContext,
          result.detections || []
        );

        const nearest =
          result.nearest_vehicle_distance_m ??
          result.nearest_distance_m;

        const alert =
          result.alert || "No Alert";

        setData({
          vehicles: result.vehicle_count ?? 0,
          pedestrians:
            result.pedestrian_count ?? 0,
          potholes: result.pothole_count ?? 0,
          distance:
            nearest != null
              ? `${nearest} m`
              : "N/A",
          lane:
            result.lane_status ||
            "Lane Not Detected",
          wrongWay: Boolean(
            result.wrong_way
          ),
          alert,
        });

        speakAlert(alert, audioEnabled);
        onAlert(alert);

        if (alert === "No Alert") {
          resetSpeechAlert();
        }

        setStatus(
          mode === "video"
            ? "Uploaded video analysis running"
            : "Live road analysis running"
        );
      } catch (error) {
        console.error(
          "Road detection failed:",
          error
        );

        setStatus(
          "Road backend connection failed"
        );
      } finally {
        processingRef.current = false;
      }
    }, INTERVAL);

    return () => clearInterval(timer);
  }, [audioEnabled, mode, onAlert]);

  return (
    <section>
      <div className="mb-3 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={startCamera}
          className="rounded-xl bg-green-700 px-4 py-2 font-semibold"
        >
          Road Camera
        </button>

        <label className="cursor-pointer rounded-xl bg-purple-700 px-4 py-2 font-semibold">
          Upload Road Video

          <input
            type="file"
            accept="video/*"
            onChange={uploadVideo}
            className="hidden"
          />
        </label>
      </div>

      <div className="rounded-2xl bg-slate-900 p-4">
        <h2 className="text-xl font-bold">
          Road Camera
        </h2>

        <p className="mb-3 text-sm text-slate-400">
          {status}
        </p>

        <div className="relative overflow-hidden rounded-xl bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            controls={mode === "video"}
            className="block w-full"
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

      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-6">
        <InfoCard
          label="Vehicles"
          value={data.vehicles}
        />

        <InfoCard
          label="Pedestrians"
          value={data.pedestrians}
        />

        <InfoCard
          label="Distance"
          value={data.distance}
        />

        <InfoCard
          label="Lane"
          value={data.lane}
          danger={data.lane.includes(
            "Departure"
          )}
        />

        <InfoCard
          label="Potholes"
          value={data.potholes}
          danger={data.potholes > 0}
        />

        <InfoCard
          label="Wrong Way"
          value={
            data.wrongWay
              ? "Detected"
              : "Clear"
          }
          danger={data.wrongWay}
        />
      </div>
    </section>
  );
}