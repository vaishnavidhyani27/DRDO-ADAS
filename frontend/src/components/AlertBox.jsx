function AlertBox({ alert }) {
  const isSafe = alert === "No Alert";

  return (
    <div
      className={`mt-8 rounded-xl p-4 text-center font-bold text-lg ${
        isSafe ? "bg-green-600" : "bg-red-600 animate-pulse"
      }`}
    >
      🚨 {alert || "No Alert"}
    </div>
  );
}

export default AlertBox;