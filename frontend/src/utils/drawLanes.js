function drawLine(ctx, points, color) {
  if (!Array.isArray(points) || points.length < 2) return;

  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);

  points.slice(1).forEach(([x, y]) => {
    ctx.lineTo(x, y);
  });

  ctx.strokeStyle = color;
  ctx.lineWidth = 6;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

export function drawLanes(ctx, data) {
  const left = data.left_lane || [];
  const right = data.right_lane || [];
  const polygon = data.lane_polygon || [];

  if (polygon.length >= 4) {
    ctx.beginPath();
    ctx.moveTo(polygon[0][0], polygon[0][1]);

    polygon.slice(1).forEach(([x, y]) => {
      ctx.lineTo(x, y);
    });

    ctx.closePath();
    ctx.fillStyle = data.lane_departure
      ? "rgba(239, 68, 68, 0.22)"
      : "rgba(34, 197, 94, 0.22)";
    ctx.fill();
  }

  const color = data.lane_departure
    ? "#ef4444"
    : "#facc15";

  drawLine(ctx, left, color);
  drawLine(ctx, right, color);
}