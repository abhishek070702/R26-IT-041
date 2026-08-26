import {
  BookOpen,
  Brain,
  Camera,
  Eye,
  FileText,
  Mic,
  Newspaper,
  Radio,
  ScanLine,
  Sparkles,
  Volume2,
  Wifi,
} from "lucide-react";
import Reveal from "./Reveal";
import { useLiveStatus } from "../hooks/useLiveStatus";
import {
  MODULE_STATUS_VARIANT,
  formatModuleLabel,
  formatModuleStatus,
  usefulEventDetails,
} from "../liveDemoStatus";

const MODULE_ICONS = {
  manoj: Camera,
  abhishek: Eye,
  harshaka: Brain,
  rashmi: Mic,
  system: Sparkles,
};

function formatTimestamp(value) {
  if (!value) {
    return "Not updated yet";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString();
}

function formatEventTime(value) {
  if (!value) {
    return "";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function displayValue(value) {
  return value ? value : "—";
}

function ResultRow({ label, value }) {
  return (
    <div className="vr-live__result-row">
      <span>{label}</span>
      <strong>{displayValue(value)}</strong>
    </div>
  );
}

function DocumentResultPanel({ output }) {
  const kind = output.documentKind;

  if (kind === "newspaper") {
    return (
      <>
        <h3>Document Result Panel</h3>
        <p className="vr-live__panel-kicker">
          <Newspaper size={14} aria-hidden="true" />
          Newspaper workflow
        </p>
        <div className="vr-live__result-list">
          <ResultRow label="Newspaper Name" value={output.title} />
          <ResultRow label="Selected Category" value={output.selectedCategory} />
          <ResultRow label="Selected Mode" value={output.selectedMode} />
          <ResultRow
            label="Article/Image Description"
            value={output.pageImageDescription}
          />
        </div>
      </>
    );
  }

  if (kind === "report" || kind === "printed_letter") {
    return (
      <>
        <h3>Document Result Panel</h3>
        <p className="vr-live__panel-kicker">
          <FileText size={14} aria-hidden="true" />
          {kind === "report" ? "Report workflow" : "Printed letter workflow"}
        </p>
        <div className="vr-live__result-list">
          <ResultRow label="Document Type" value={output.documentType} />
          <ResultRow
            label="Output Preview"
            value={output.outputText || output.speechText}
          />
          <ResultRow
            label="Image Description"
            value={output.imageDescription || output.pageImageDescription}
          />
        </div>
      </>
    );
  }

  return (
    <>
      <h3>Document Result Panel</h3>
      <p className="vr-live__panel-kicker">
        <BookOpen size={14} aria-hidden="true" />
        {kind === "magazine" ? "Magazine workflow" : "Novel workflow"}
      </p>
      <div className="vr-live__result-list">
        <ResultRow label="Document Type" value={output.documentType} />
        <ResultRow label="Title" value={output.title} />
        <ResultRow label="Cover Description" value={output.coverDescription} />
        <ResultRow
          label="Current Page"
          value={
            output.currentPage
              ? `${output.currentPage}${output.pageType ? ` · ${output.pageType}` : ""}`
              : output.pageType
          }
        />
        <ResultRow
          label="Page Image Description"
          value={output.pageImageDescription}
        />
      </div>
    </>
  );
}

function WorkflowTimeline({ events, timeline }) {
  const eventItems = events.length
    ? events
    : timeline.map((step) => ({
        id: step.id,
        module: "system",
        status:
          step.state === "complete"
            ? "ready"
            : step.state === "active"
              ? "running"
              : "waiting",
        message: step.label,
        time: "",
        details: {},
      }));

  return (
    <>
      <h3>Live Workflow Timeline</h3>
      <p className="vr-live__panel-kicker">
        Step-by-step device events from the live status API
      </p>
      {eventItems.length ? (
        <ol className="vr-live__events">
          {eventItems.map((event) => {
            const Icon = MODULE_ICONS[event.module] || Sparkles;
            const details = usefulEventDetails(event.details);
            const statusLabel = formatModuleStatus(event.status);
            const variant = MODULE_STATUS_VARIANT[statusLabel] || "waiting";

            return (
              <li
                key={event.id}
                className={`vr-live__event vr-live__event--${event.status}`}
              >
                <span className="vr-live__event-rail" aria-hidden="true">
                  <span className="vr-live__event-dot" />
                </span>
                <div className="vr-live__event-body">
                  <div className="vr-live__event-head">
                    <span className="vr-live__event-module">
                      <Icon size={14} aria-hidden="true" />
                      {formatModuleLabel(event.module)}
                    </span>
                    <span className={`vr-live__badge vr-live__badge--${variant}`}>
                      {statusLabel}
                    </span>
                  </div>
                  <p className="vr-live__event-message">{event.message}</p>
                  {event.time ? (
                    <time className="vr-live__event-time" dateTime={event.time}>
                      {formatEventTime(event.time)}
                    </time>
                  ) : null}
                  {details.length ? (
                    <div className="vr-live__event-details">
                      {details.map((item) => (
                        <span key={`${event.id}-${item.key}`}>
                          <strong>{item.key}</strong> {item.value}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="vr-live__empty">Waiting for workflow events…</p>
      )}
    </>
  );
}

export default function LiveDeviceWorkflow() {
  const {
    connectionState,
    connectionLabel,
    device,
    modules,
    timeline,
    events,
    output,
    error,
    lastUpdated,
    activeModule,
  } = useLiveStatus();

  const badgeClass =
    connectionState === "connected"
      ? "vr-live__badge--connected"
      : connectionState === "waiting"
        ? "vr-live__badge--waiting"
        : "vr-live__badge--demo";

  const badgeLabel =
    connectionState === "connected"
      ? "Live"
      : connectionState === "waiting"
        ? "Waiting"
        : "Demo";

  return (
    <section className="vr-live" id="live-workflow">
      <Reveal variant="up" delay={80} className="vr-live__status">
        <div className="vr-live__status-head">
          <div>
            <p className="vr-live__label">Device status</p>
            <h3>{device.name}</h3>
            <p className={`vr-live__api-label vr-live__api-label--${connectionState}`}>
              {connectionLabel}
            </p>
          </div>
          <span className={`vr-live__badge ${badgeClass}`}>
            <Sparkles size={14} aria-hidden="true" />
            {badgeLabel}
          </span>
        </div>
        <div className="vr-live__status-grid">
          <div className="vr-live__stat">
            <Wifi size={18} aria-hidden="true" />
            <div>
              <span>Connection</span>
              <strong>{device.connection}</strong>
            </div>
          </div>
          <div className="vr-live__stat">
            <ScanLine size={18} aria-hidden="true" />
            <div>
              <span>Pipeline</span>
              <strong>{device.status}</strong>
            </div>
          </div>
          <div className="vr-live__stat">
            <Radio size={18} aria-hidden="true" />
            <div>
              <span>Current step</span>
              <strong>{device.currentStep}</strong>
            </div>
          </div>
        </div>
        <p className="vr-live__meta-line">
          <span>
            <strong>Active module</strong> {activeModule || "none"}
          </span>
          <span>
            <strong>Workflow stage</strong> {displayValue(device.workflowStage)}
          </span>
          <span>
            <strong>Last updated</strong> {formatTimestamp(lastUpdated)}
          </span>
        </p>
        {error ? (
          <p className="vr-live__error" role="status">
            {error}
          </p>
        ) : null}
      </Reveal>

      <div className="vr-live__modules">
        {modules.map((mod, i) => {
          const Icon = MODULE_ICONS[mod.id] || Brain;
          const variant = MODULE_STATUS_VARIANT[mod.status] || "waiting";

          return (
            <Reveal key={mod.id} as="article" variant="up" delay={i * 90}>
              <div className="vr-live__module-icon">
                <Icon size={22} aria-hidden="true" />
              </div>
              <div className="vr-live__module-copy">
                <h3>{mod.name}</h3>
                <p>{mod.role}</p>
              </div>
              <span className={`vr-live__badge vr-live__badge--${variant}`}>
                {mod.status}
              </span>
            </Reveal>
          );
        })}
      </div>

      <div className="vr-live__panels">
        <Reveal variant="left" delay={60} className="vr-live__panel vr-live__panel--timeline">
          <WorkflowTimeline events={events} timeline={timeline} />
        </Reveal>

        <Reveal variant="right" delay={120} className="vr-live__panel vr-live__panel--output">
          <DocumentResultPanel output={output} />
          <div className="vr-live__prefs">
            <span>{output.readingLevel || "—"} reading</span>
            <span>{output.voice || "—"} voice</span>
            <span>{output.pace || "—"} pace</span>
            <span>{output.tone || "—"} tone</span>
          </div>
        </Reveal>
      </div>

      <Reveal variant="up" delay={160} className="vr-live__panel vr-live__panel--speech">
        <h3>Speech Output</h3>
        <p className="vr-live__panel-kicker">
          What Rashmi is speaking to the user
        </p>
        <div className="vr-live__spoken">
          <Volume2 size={18} aria-hidden="true" />
          <p>{output.spokenText}</p>
        </div>
      </Reveal>
    </section>
  );
}
