import {
  Brain,
  Camera,
  Eye,
  Mic,
  Radio,
  ScanLine,
  Sparkles,
  Volume2,
  Wifi,
} from "lucide-react";
import Reveal from "./Reveal";
import { liveDemoStatus, MODULE_STATUS_VARIANT } from "../liveDemoStatus";

const MODULE_ICONS = {
  manoj: Camera,
  abhishek: Eye,
  harshaka: Brain,
  rashmi: Mic,
};

export default function LiveDeviceWorkflow() {
  const { device, modules, timeline, output } = liveDemoStatus;

  return (
    <section className="vr-live" id="live-workflow">
      <Reveal variant="up" delay={80} className="vr-live__status">
        <div className="vr-live__status-head">
          <div>
            <p className="vr-live__label">Device status</p>
            <h3>{device.name}</h3>
          </div>
          <span className="vr-live__badge vr-live__badge--demo">
            <Sparkles size={14} aria-hidden="true" />
            Demo mode
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
        <Reveal variant="left" delay={60} className="vr-live__panel">
          <h3>Pipeline timeline</h3>
          <ol className="vr-live__timeline">
            {timeline.map((step) => (
              <li
                key={step.id}
                className={`vr-live__timeline-item vr-live__timeline-item--${step.state}`}
              >
                <span className="vr-live__timeline-dot" aria-hidden="true" />
                <span>{step.label}</span>
              </li>
            ))}
          </ol>
        </Reveal>

        <Reveal variant="right" delay={120} className="vr-live__panel vr-live__panel--output">
          <h3>Live output preview</h3>
          <div className="vr-live__output-meta">
            <span>
              <strong>Type</strong> {output.documentType}
            </span>
            <span>
              <strong>Title</strong> {output.title}
            </span>
            <span>
              <strong>Category</strong> {output.category}
            </span>
          </div>
          <div className="vr-live__prefs">
            <span>{output.readingLevel} reading</span>
            <span>{output.voice} voice</span>
            <span>{output.pace} pace</span>
            <span>{output.tone} tone</span>
          </div>
          <div className="vr-live__spoken">
            <Volume2 size={18} aria-hidden="true" />
            <p>{output.spokenText}</p>
          </div>
          <p className="vr-live__image-desc">
            <Eye size={16} aria-hidden="true" />
            {output.imageDescription}
          </p>
        </Reveal>
      </div>
    </section>
  );
}
