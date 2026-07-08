import { useEffect, useState } from "react";
import Header from "./components/Header";
import CameraCard from "./components/CameraCard";
import StatusCard from "./components/StatusCard";
import AlertBox from "./components/AlertBox";

function App() {
  const [status, setStatus] = useState({});

  useEffect(() => {
    fetch("http://127.0.0.1:5000/status")
      .then((response) => response.json())
      .then((data) => setStatus(data))
      .catch((error) => console.error("Error fetching status:", error));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <Header />

      <div className="grid md:grid-cols-2 gap-6 mt-6">
        <CameraCard title="📷 Road Camera" />
        <CameraCard title="😴 Driver Camera" />
      </div>

      <StatusCard status={status} />

      <AlertBox />

      <div className="flex justify-center gap-6 mt-8">
        <button className="bg-green-600 px-8 py-3 rounded-xl hover:bg-green-700">
          ▶ Start AI
        </button>

        <button className="bg-red-600 px-8 py-3 rounded-xl hover:bg-red-700">
          ■ Stop AI
        </button>
      </div>
    </div>
  );
}

export default App;