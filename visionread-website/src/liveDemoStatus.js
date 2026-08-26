export const LIVE_STATUS_URL =
  import.meta.env.VITE_LIVE_STATUS_URL || "http://127.0.0.1:8000/live-status";

export const MODULE_DEFINITIONS = [
  {
    id: "manoj",
    name: "Manoj",
    role: "Camera guidance and auto capture",
  },
  {
    id: "abhishek",
    name: "Abhishek",
    role: "Document type, title, image description",
  },
  {
    id: "harshaka",
    name: "Harshaka",
    role: "OCR, category, summary/full content",
  },
  {
    id: "rashmi",
    name: "Rashmi",
    role: "RFID, preferences, voice input, TTS",
  },
];

export const FALLBACK_TIMELINE = [
  { id: "rfid", label: "RFID user identification", state: "complete" },
  { id: "prefs", label: "User preferences loaded", state: "complete" },
  { id: "camera", label: "Camera guidance started", state: "active" },
  { id: "capture", label: "Page captured", state: "pending" },
  {
    id: "abhishek",
    label: "Abhishek analyzes document type/title/image",
    state: "pending",
  },
  {
    id: "harshaka",
    label: "Harshaka extracts text/category",
    state: "pending",
  },
  {
    id: "rashmi",
    label: "Rashmi applies reading level and TTS",
    state: "pending",
  },
  { id: "output", label: "Output spoken to user", state: "pending" },
];

export const FALLBACK_API_PAYLOAD = {
  device_status: "waiting",
  current_step: "Waiting for device workflow",
  active_module: "none",
  document_type: "Newspaper",
  title: "Daily",
  category: "Politics",
  reading_level: "simple",
  voice: "female",
  pace: "slow",
  tone: "calm",
  output_text:
    "Politics section. I found one article about the city council vote. The headline mentions new transport funding for the downtown corridor.",
  image_description:
    "The selected article image appears to show a man in a suit and tie.",
  error: "",
  last_updated: "",
  modules: {
    manoj: "ready",
    abhishek: "waiting",
    harshaka: "waiting",
    rashmi: "running",
  },
};

export const MODULE_STATUS_VARIANT = {
  Ready: "ready",
  Running: "running",
  Waiting: "waiting",
  Error: "error",
};

const TIMELINE_MATCHERS = [
  { match: ["session started", "rfid"], index: 0 },
  { match: ["preference"], index: 1 },
  { match: ["camera guidance"], index: 2 },
  { match: ["page captured", "capture"], index: 3 },
  { match: ["abhishek"], index: 4 },
  { match: ["harshaka", "category", "summary", "full text"], index: 5 },
  { match: ["rashmi", "speaking"], index: 6 },
  {
    match: ["output", "image description", "workflow completed", "completed"],
    index: 7,
  },
];

function pickString(value, fallback = "") {
  if (value === null || value === undefined) {
    return fallback;
  }
  return String(value).trim() || fallback;
}

function pickModuleStatus(value) {
  const normalized = pickString(value, "waiting").toLowerCase();
  if (["waiting", "running", "ready", "error"].includes(normalized)) {
    return normalized;
  }
  return "waiting";
}

export function formatModuleStatus(rawStatus) {
  const normalized = pickModuleStatus(rawStatus);
  if (normalized === "ready") return "Ready";
  if (normalized === "running") return "Running";
  if (normalized === "error") return "Error";
  return "Waiting";
}

export function normalizeApiPayload(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const modules = source.modules && typeof source.modules === "object"
    ? source.modules
    : {};

  return {
    device_status: pickString(
      source.device_status,
      FALLBACK_API_PAYLOAD.device_status,
    ),
    current_step: pickString(
      source.current_step,
      FALLBACK_API_PAYLOAD.current_step,
    ),
    active_module: pickString(
      source.active_module,
      FALLBACK_API_PAYLOAD.active_module,
    ),
    document_type: pickString(
      source.document_type,
      FALLBACK_API_PAYLOAD.document_type,
    ),
    title: pickString(source.title, FALLBACK_API_PAYLOAD.title),
    category: pickString(source.category, FALLBACK_API_PAYLOAD.category),
    reading_level: pickString(
      source.reading_level,
      FALLBACK_API_PAYLOAD.reading_level,
    ),
    voice: pickString(source.voice, FALLBACK_API_PAYLOAD.voice),
    pace: pickString(source.pace, FALLBACK_API_PAYLOAD.pace),
    tone: pickString(source.tone, FALLBACK_API_PAYLOAD.tone),
    output_text: pickString(
      source.output_text,
      FALLBACK_API_PAYLOAD.output_text,
    ),
    image_description: pickString(
      source.image_description,
      FALLBACK_API_PAYLOAD.image_description,
    ),
    error: pickString(source.error, ""),
    last_updated: pickString(source.last_updated, ""),
    modules: {
      manoj: pickModuleStatus(modules.manoj ?? FALLBACK_API_PAYLOAD.modules.manoj),
      abhishek: pickModuleStatus(
        modules.abhishek ?? FALLBACK_API_PAYLOAD.modules.abhishek,
      ),
      harshaka: pickModuleStatus(
        modules.harshaka ?? FALLBACK_API_PAYLOAD.modules.harshaka,
      ),
      rashmi: pickModuleStatus(modules.rashmi ?? FALLBACK_API_PAYLOAD.modules.rashmi),
    },
  };
}

export function buildTimeline(currentStep) {
  const step = pickString(currentStep).toLowerCase();
  let activeIndex = -1;

  for (const matcher of TIMELINE_MATCHERS) {
    if (matcher.match.some((token) => step.includes(token))) {
      activeIndex = Math.max(activeIndex, matcher.index);
    }
  }

  if (activeIndex < 0) {
    return FALLBACK_TIMELINE.map((item) => ({ ...item }));
  }

  return FALLBACK_TIMELINE.map((item, index) => ({
    ...item,
    state:
      index < activeIndex
        ? "complete"
        : index === activeIndex
          ? "active"
          : "pending",
  }));
}

export function mapApiToViewModel(raw, connectionState = "waiting") {
  const api = normalizeApiPayload(raw);

  const connectionLabels = {
    connected: "Connected",
    waiting: "Waiting for backend",
    fallback: "Demo fallback mode",
  };

  return {
    device: {
      name: "VisionRead Wearable",
      status: api.device_status,
      connection: connectionLabels[connectionState] || connectionLabels.waiting,
      currentStep: api.current_step,
      activeModule: api.active_module,
    },
    modules: MODULE_DEFINITIONS.map((definition) => ({
      ...definition,
      status: formatModuleStatus(api.modules[definition.id]),
    })),
    timeline: buildTimeline(api.current_step),
    output: {
      documentType: api.document_type,
      title: api.title,
      category: api.category,
      readingLevel: api.reading_level,
      voice: api.voice,
      pace: api.pace,
      tone: api.tone,
      spokenText: api.output_text || "Waiting for spoken output...",
      imageDescription:
        api.image_description || "No image description yet.",
    },
    error: api.error,
    lastUpdated: api.last_updated,
    activeModule: api.active_module,
  };
}

export const CONNECTION_LABELS = {
  connected: "Live API: connected",
  waiting: "Waiting for backend",
  fallback: "Demo fallback: backend not connected",
};

/** @deprecated Use mapApiToViewModel(FALLBACK_API_PAYLOAD) via useLiveStatus instead */
export const liveDemoStatus = mapApiToViewModel(
  FALLBACK_API_PAYLOAD,
  "fallback",
);
