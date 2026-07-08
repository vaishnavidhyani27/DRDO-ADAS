function StatusCard({ status }) {
  const data = [
    ["🚗 Vehicle", status.vehicle],
    ["📏 Distance", status.distance],
    ["🚶 Pedestrian", status.pedestrian],
    ["🕳️ Pothole", status.pothole],
    ["🛣️ Lane", status.lane],
    ["👁️ Driver", status.driver],
    ["📱 Phone", status.phone],
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
      {data.map(([title, value]) => (
        <div
          key={title}
          className="bg-slate-800 rounded-xl p-4 shadow-lg"
        >
          <p className="text-slate-400 text-sm">{title}</p>
          <h2 className="text-xl font-bold mt-2">
            {value || "--"}
          </h2>
        </div>
      ))}
    </div>
  );
}

export default StatusCard;