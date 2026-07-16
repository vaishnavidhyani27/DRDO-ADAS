const BASE_URL = "http://34.100.209.136:5000";

async function sendFrame(endpoint, image) {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image }),
  });

  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }

  return response.json();
}

export const detectRoad = (image) =>
  sendFrame("/detect", image);

export const detectDriver = (image) =>
  sendFrame("/detect-driver", image);