export function drawDetections(ctx, detections = []) {
  detections.forEach((detection) => {
    if (!detection.bbox || detection.bbox.length !== 4) return;

    const [x1, y1, x2, y2] = detection.bbox;

    const pothole = detection.class === "Pothole";

    const color = pothole ? "#ef4444" : "#22c55e";

    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const confidence = Math.round(
      (detection.confidence || 0) * 100
    );

    const distance =
      !pothole &&
      detection.distance_m != null
        ? ` | ${detection.distance_m} m`
        : "";

    const label =
      `${detection.class} ${confidence}%${distance}`;

    ctx.font = "18px Arial";

    const width =
      ctx.measureText(label).width + 14;

    const top =
      y1 > 30 ? y1 - 28 : y1;

    ctx.fillStyle = color;
    ctx.fillRect(x1, top, width, 28);

    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 7, top + 20);
  });
}