function CameraCard({ title }) {
  return (
    <div className="bg-slate-800 rounded-2xl p-4 shadow-lg">
      <h2 className="text-xl font-semibold mb-4">{title}</h2>

      <div className="h-72 rounded-xl bg-black flex items-center justify-center border border-slate-700">
        Live Camera Feed
      </div>
    </div>
  );
}

export default CameraCard;