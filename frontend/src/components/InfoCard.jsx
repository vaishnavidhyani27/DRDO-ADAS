export default function InfoCard({
  label,
  value,
  danger = false,
}) {
  return (
    <div className="bg-slate-900 rounded-xl p-4 shadow-lg text-center">
      <p className="text-slate-400 text-sm">
        {label}
      </p>

      <h2
        className={`mt-2 text-xl font-bold ${
          danger ? "text-red-400" : "text-white"
        }`}
      >
        {value}
      </h2>
    </div>
  );
}