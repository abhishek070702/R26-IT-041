/**
 * Mock live device status for PP2 demo UI.
 * Replace with: const res = await fetch("/api/live-status"); liveDemoStatus = await res.json();
 */
export const liveDemoStatus = {
  device: {
    name: "VisionRead Wearable",
    status: "Demo Mode / Waiting for Pi",
    connection: "Backend ready",
    currentStep: "Waiting for capture",
  },
  modules: [
    {
      id: "manoj",
      name: "Manoj",
      role: "Camera guidance and auto capture",
      status: "Ready",
    },
    {
      id: "abhishek",
      name: "Abhishek",
      role: "Document type, title, image description",
      status: "Waiting",
    },
    {
      id: "harshaka",
      name: "Harshaka",
      role: "OCR, category, summary/full content",
      status: "Waiting",
    },
    {
      id: "rashmi",
      name: "Rashmi",
      role: "RFID, preferences, voice input, TTS",
      status: "Running",
    },
  ],
  timeline: [
    { id: "rfid", label: "RFID user identification", state: "complete" },
    { id: "prefs", label: "User preferences loaded", state: "complete" },
    { id: "camera", label: "Camera guidance started", state: "active" },
    { id: "capture", label: "Page captured", state: "pending" },
    { id: "abhishek", label: "Abhishek analyzes document type/title/image", state: "pending" },
    { id: "harshaka", label: "Harshaka extracts text/category", state: "pending" },
    { id: "rashmi", label: "Rashmi applies reading level and TTS", state: "pending" },
    { id: "output", label: "Output spoken to user", state: "pending" },
  ],
  output: {
    documentType: "Newspaper",
    title: "Daily",
    category: "Politics",
    readingLevel: "Simple",
    voice: "Female",
    pace: "Slow",
    tone: "Calm",
    spokenText:
      "Politics section. I found one article about the city council vote. The headline mentions new transport funding for the downtown corridor.",
    imageDescription:
      "The selected article image appears to show a man in a suit and tie.",
  },
};

export const MODULE_STATUS_VARIANT = {
  Ready: "ready",
  Running: "running",
  Waiting: "waiting",
};
