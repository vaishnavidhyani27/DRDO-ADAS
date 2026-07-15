import { useEffect, useRef, useState } from "react";

import { detectDriver } from "../utils/api";
import { speakAlert } from "../utils/speech";
import InfoCard from "./InfoCard";

const INTERVAL = 1200;

const INITIAL = {
  face: false,
  status: "Not Started",
  direction: "Unknown",
  eyesClosed: false,
  yawning: false,
  drowsy: false,
  alert: "No Alert",
};

export default function DriverCamera({
  audioEnabled,
  onAlert,
}) {
  const videoRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const processingRef = useRef(false);

  const [cameraStatus, setCameraStatus] = useState(
    "Driver camera not started"
  );

  const [driver, setDriver] = useState(INITIAL);

  function stopCamera() {
    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  }

  async function startCamera() {
    stopCamera();

    setCameraStatus("Starting driver camera...");

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: "user",
            },
            width: {
              ideal: 640,
            },
            height: {
              ideal: 480,
            },
          },
          audio: false,
        });

      streamRef.current = stream;

      const video = videoRef.current;

      video.srcObject = stream;
      video.muted = true;

      await video.play();

      setCameraStatus(
        "Driver camera active"
      );
    } catch (error) {
      console.error(error);

      setCameraStatus(
        "Unable to access driver camera"
      );
    }
  }

  useEffect(() => {
    return () => stopCamera();
  }, []);

  useEffect(() => {
    const timer = setInterval(async () => {
      const video = videoRef.current;
      const capture = captureRef.current;

      if (
        processingRef.current ||
        !video ||
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

        const ctx =
          capture.getContext("2d");

        ctx.drawImage(
          video,
          0,
          0,
          width,
          height
        );

        const image =
          capture.toDataURL(
            "image/jpeg",
            0.8
          );

        const result =
          await detectDriver(image);

        const alert =
          result.alert || "No Alert";

        setDriver({
          face: Boolean(
            result.face_detected
          ),
          status:
            result.driver_status ||
            "Unknown",
          direction:
            result.direction ||
            "Unknown",
          eyesClosed: Boolean(
            result.eyes_closed
          ),
          yawning: Boolean(
            result.yawning
          ),
          drowsy: Boolean(
            result.drowsy
          ),
          alert,
        });

        speakAlert(
          alert,
          audioEnabled
        );

        onAlert(alert);

        setCameraStatus(
          "Driver monitoring active"
        );

              } catch (error) {
        console.error(error);

        setCameraStatus(
          "Driver backend connection failed"
        );
      } finally {
        processingRef.current = false;
      }
    }, INTERVAL);

    return () => clearInterval(timer);
  }, [audioEnabled, onAlert]);

  return (
    <section>
      <button
        type="button"
        onClick={startCamera}
        className="mb-3 rounded-xl bg-cyan-700 px-4 py-2 font-semibold"
      >
        Start Driver Camera
      </button>

      <div className="rounded-2xl bg-slate-900 p-4">
        <h2 className="text-xl font-bold">
          Driver Camera
        </h2>

        <p className="mb-3 text-sm text-slate-400">
          {cameraStatus}
        </p>

        <div className="overflow-hidden rounded-xl bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="block w-full"
          />

          <canvas
            ref={captureRef}
            className="hidden"
          />
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-6">
        <InfoCard
          label="Face"
          value={
            driver.face
              ? "Detected"
              : "Not Detected"
          }
        />

        <InfoCard
          label="Status"
          value={driver.status}
          danger={
            driver.status !== "Attentive"
          }
        />

        <InfoCard
          label="Direction"
          value={driver.direction}
        />

        <InfoCard
          label="Eyes"
          value={
            driver.eyesClosed
              ? "Closed"
              : "Open"
          }
          danger={driver.eyesClosed}
        />

        <InfoCard
          label="Yawning"
          value={
            driver.yawning
              ? "Detected"
              : "No"
          }
          danger={driver.yawning}
        />

        <InfoCard
          label="Drowsiness"
          value={
            driver.drowsy
              ? "Detected"
              : "Alert"
          }
          danger={driver.drowsy}
        />
      </div>
    </section>
  );
}