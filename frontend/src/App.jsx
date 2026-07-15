import { useCallback, useState } from "react";

import DriverCamera from "./components/DriverCamera";
import RoadCamera from "./components/RoadCamera";

function App() {
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [roadAlert, setRoadAlert] = useState("No Alert");
  const [driverAlert, setDriverAlert] = useState("No Alert");

  const handleRoadAlert = useCallback((message) => {
    setRoadAlert(message);
  }, []);

  const handleDriverAlert = useCallback((message) => {
    setDriverAlert(message);
  }, []);

  function enableAudio() {
    if (!window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(
      new SpeechSynthesisUtterance("Audio alerts enabled")
    );

    setAudioEnabled(true);
  }

  const alert =
    driverAlert !== "No Alert" ? driverAlert : roadAlert;

  return (
    <main className="min-h-screen bg-slate-950 p-4 text-white">
      <header className="text-center">
        <h1 className="text-3xl font-bold">DRDO Smart ADAS</h1>

        <p className="mt-2 text-slate-400">
          Integrated road and driver monitoring
        </p>

        <button
          type="button"
          onClick={enableAudio}
          className={`mt-4 rounded-xl px-5 py-2 font-semibold ${
            audioEnabled ? "bg-green-700" : "bg-blue-700"
          }`}
        >
          {audioEnabled ? "Audio Enabled" : "Enable Audio"}
        </button>
      </header>

      <div className="mt-6 grid gap-8">
        <RoadCamera
          audioEnabled={audioEnabled}
          onAlert={handleRoadAlert}
        />

        <DriverCamera
          audioEnabled={audioEnabled}
          onAlert={handleDriverAlert}
        />
      </div>

      <div
        className={`mt-6 rounded-xl p-4 text-center text-xl font-bold ${
          alert === "No Alert" ? "bg-green-700" : "bg-red-700"
        }`}
      >
        {alert}
      </div>
    </main>
  );
}

export default App;