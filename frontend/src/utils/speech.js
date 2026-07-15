let lastMessage = "";
let lastTime = 0;

export function speakAlert(message, enabled, cooldown = 6000) {
  const blockedAlerts = [
    "Pedestrian detected",
    "Pedestrian Detected",
  ];

  if (
    !enabled ||
    !window.speechSynthesis ||
    !message ||
    message === "No Alert" ||
    blockedAlerts.includes(message)
  ) {
    return;
  }

  const now = Date.now();

  if (message === lastMessage && now - lastTime < cooldown) {
    return;
  }

  window.speechSynthesis.cancel();

  const speech = new SpeechSynthesisUtterance(message);
  speech.rate = 1;
  speech.volume = 1;

  window.speechSynthesis.speak(speech);

  lastMessage = message;
  lastTime = now;
}

export function resetSpeechAlert() {
  lastMessage = "";
}