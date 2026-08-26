export const LIVE_STATUS_URL =
  import.meta.env.VITE_LIVE_STATUS_URL || "http://127.0.0.1:8000/live-status";

export const MODULE_LABELS = {
  manoj: "Camera Guidance",
  abhishek: "Vision Analysis",
  harshaka: "Content Processing",
  rashmi: "Voice Output",
  system: "System",
};

export const MODULE_DEFINITIONS = [
  {
    id: "manoj",
    name: MODULE_LABELS.manoj,
    role: "Manoj · Camera guidance and auto capture",
  },
  {
    id: "abhishek",
    name: MODULE_LABELS.abhishek,
    role: "Abhishek · Document type, title, image description",
  },
  {
    id: "harshaka",
    name: MODULE_LABELS.harshaka,
    role: "Harshaka · OCR, category, summary/full content",
  },
  {
    id: "rashmi",
    name: MODULE_LABELS.rashmi,
    role: "Rashmi · RFID, preferences, voice input, TTS",
  },
];

export const FALLBACK_TIMELINE = [
  { id: "rfid", label: "RFID user identification", state: "complete" },
  { id: "prefs", label: "User preferences loaded", state: "complete" },
  { id: "camera", label: "Camera guidance started", state: "complete" },
  { id: "capture", label: "Page captured", state: "complete" },
  {
    id: "abhishek",
    label: "Abhishek analyzes document type/title/image",
    state: "complete",
  },
  {
    id: "harshaka",
    label: "Harshaka extracts text/category",
    state: "pending",
  },
  {
    id: "rashmi",
    label: "Rashmi applies reading level and TTS",
    state: "active",
  },
  { id: "output", label: "Output spoken to user", state: "pending" },
];

const FALLBACK_EVENT_TIMES = [
  "2026-08-26T17:20:02.000Z",
  "2026-08-26T17:20:08.000Z",
  "2026-08-26T17:20:14.000Z",
  "2026-08-26T17:20:28.000Z",
  "2026-08-26T17:20:31.000Z",
  "2026-08-26T17:20:38.000Z",
  "2026-08-26T17:20:39.000Z",
  "2026-08-26T17:20:42.000Z",
  "2026-08-26T17:20:45.000Z",
];

export const FALLBACK_API_PAYLOAD = {
  device_status: "running",
  current_step: "Cover image described",
  active_module: "rashmi",
  document_type: "Novel",
  title: "Peter Pan",
  category: "",
  reading_level: "simple",
  voice: "female",
  pace: "slow",
  tone: "calm",
  output_text: "This is a Novel. The title is Peter Pan.",
  image_description:
    "The cover shows a boy in an illustrated fantasy scene.",
  error: "",
  last_updated: FALLBACK_EVENT_TIMES[8],
  workflow_stage: "cover_described",
  current_page: 1,
  page_type: "cover",
  cover_description:
    "The cover shows a boy in an illustrated fantasy scene.",
  page_image_description: "",
  speech_text: "This is a Novel. The title is Peter Pan.",
  selected_mode: "",
  selected_category: "",
  modules: {
    manoj: "ready",
    abhishek: "ready",
    harshaka: "waiting",
    rashmi: "running",
  },
  events: [
    {
      time: FALLBACK_EVENT_TIMES[0],
      module: "rashmi",
      status: "running",
      message: "Session started",
      details: {},
    },
    {
      time: FALLBACK_EVENT_TIMES[1],
      module: "rashmi",
      status: "ready",
      message: "RFID user preferences loaded",
      details: {
        reading_level: "simple",
        voice: "female",
        pace: "slow",
        tone: "calm",
      },
    },
    {
      time: FALLBACK_EVENT_TIMES[2],
      module: "manoj",
      status: "running",
      message: "Camera guidance started",
      details: { page_type: "cover" },
    },
    {
      time: FALLBACK_EVENT_TIMES[3],
      module: "manoj",
      status: "ready",
      message: "Cover page captured",
      details: { current_page: 1, page_type: "cover" },
    },
    {
      time: FALLBACK_EVENT_TIMES[4],
      module: "abhishek",
      status: "running",
      message: "Abhishek analyzing cover page",
      details: { page_type: "cover" },
    },
    {
      time: FALLBACK_EVENT_TIMES[5],
      module: "abhishek",
      status: "ready",
      message: "Document detected as Novel",
      details: { document_type: "Novel" },
    },
    {
      time: FALLBACK_EVENT_TIMES[6],
      module: "abhishek",
      status: "ready",
      message: "Title detected: Peter Pan",
      details: { title: "Peter Pan" },
    },
    {
      time: FALLBACK_EVENT_TIMES[7],
      module: "abhishek",
      status: "ready",
      message: "Cover image described",
      details: {
        cover_description:
          "The cover shows a boy in an illustrated fantasy scene.",
      },
    },
    {
      time: FALLBACK_EVENT_TIMES[8],
      module: "rashmi",
      status: "running",
      message: "Rashmi speaking title and cover description",
      details: { title: "Peter Pan" },
    },
  ],
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

export function formatModuleLabel(moduleId) {
  const key = pickString(moduleId, "system").toLowerCase();
  return MODULE_LABELS[key] || MODULE_LABELS.system;
}

export function getDocumentKind(documentType) {
  const value = pickString(documentType).toLowerCase().replace(/_/g, " ");
  if (value.includes("newspaper")) return "newspaper";
  if (value.includes("magazine")) return "magazine";
  if (value.includes("novel")) return "novel";
  if (value.includes("letter")) return "printed_letter";
  if (value.includes("report")) return "report";
  return "other";
}

export function usefulEventDetails(details) {
  if (!details || typeof details !== "object" || Array.isArray(details)) {
    return [];
  }

  return Object.entries(details)
    .filter(([, value]) => {
      if (value === null || value === undefined || value === "") {
        return false;
      }
      if (typeof value === "boolean") {
        return value;
      }
      if (Array.isArray(value)) {
        return value.length > 0;
      }
      if (typeof value === "object") {
        return Object.keys(value).length > 0;
      }
      return true;
    })
    .map(([key, value]) => ({
      key: key.replace(/_/g, " "),
      value:
        typeof value === "object" ? JSON.stringify(value) : String(value),
    }));
}

function normalizeEvents(rawEvents) {
  if (!Array.isArray(rawEvents)) {
    return [];
  }

  return rawEvents
    .filter((event) => event && typeof event === "object")
    .slice(-30)
    .map((event, index) => ({
      id: `${pickString(event.time, "event")}-${index}`,
      time: pickString(event.time, ""),
      module: pickString(event.module, "system").toLowerCase(),
      status: pickModuleStatus(event.status),
      message: pickString(event.message, ""),
      details:
        event.details && typeof event.details === "object" && !Array.isArray(event.details)
          ? event.details
          : {},
    }));
}

export function normalizeApiPayload(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const modules = source.modules && typeof source.modules === "object"
    ? source.modules
    : {};

  return {
    device_status: pickString(source.device_status, "waiting"),
    current_step: pickString(
      source.current_step,
      "Waiting for device workflow",
    ),
    active_module: pickString(source.active_module, "none"),
    document_type: pickString(source.document_type, ""),
    title: pickString(source.title, ""),
    category: pickString(source.category, ""),
    reading_level: pickString(source.reading_level, ""),
    voice: pickString(source.voice, ""),
    pace: pickString(source.pace, ""),
    tone: pickString(source.tone, ""),
    output_text: pickString(source.output_text, ""),
    image_description: pickString(source.image_description, ""),
    error: pickString(source.error, ""),
    last_updated: pickString(source.last_updated, ""),
    workflow_stage: pickString(source.workflow_stage, ""),
    current_page: source.current_page === 0 || source.current_page
      ? String(source.current_page)
      : "",
    page_type: pickString(source.page_type, ""),
    cover_description: pickString(source.cover_description, ""),
    page_image_description: pickString(source.page_image_description, ""),
    speech_text: pickString(source.speech_text, ""),
    selected_mode: pickString(source.selected_mode, ""),
    selected_category: pickString(
      source.selected_category,
      pickString(source.category, ""),
    ),
    modules: {
      manoj: pickModuleStatus(modules.manoj ?? "waiting"),
      abhishek: pickModuleStatus(modules.abhishek ?? "waiting"),
      harshaka: pickModuleStatus(modules.harshaka ?? "waiting"),
      rashmi: pickModuleStatus(modules.rashmi ?? "waiting"),
    },
    events: normalizeEvents(source.events),
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
  const documentKind = getDocumentKind(api.document_type);
  const speechText = api.speech_text || api.output_text;

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
      workflowStage: api.workflow_stage,
    },
    modules: MODULE_DEFINITIONS.map((definition) => ({
      ...definition,
      status: formatModuleStatus(api.modules[definition.id]),
    })),
    timeline: buildTimeline(api.current_step),
    events: api.events,
    output: {
      documentKind,
      documentType: api.document_type,
      title: api.title,
      category: api.category,
      readingLevel: api.reading_level,
      voice: api.voice,
      pace: api.pace,
      tone: api.tone,
      spokenText: speechText || "Waiting for spoken output...",
      speechText,
      outputText: api.output_text,
      imageDescription: api.image_description,
      coverDescription: api.cover_description || api.image_description,
      pageImageDescription:
        api.page_image_description || api.image_description,
      currentPage: api.current_page,
      pageType: api.page_type,
      selectedMode: api.selected_mode,
      selectedCategory: api.selected_category || api.category,
      workflowStage: api.workflow_stage,
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
