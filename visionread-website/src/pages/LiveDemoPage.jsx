import { ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";
import LiveDeviceWorkflow from "../components/LiveDeviceWorkflow";
import Reveal from "../components/Reveal";
import SiteShell from "../components/SiteShell";

export default function LiveDemoPage() {
  return (
    <SiteShell activePage="live-demo">
      <section className="vr-live-page">
        <Reveal variant="up" className="vr-live-page__hero">
          <Link to="/" className="vr-live-page__back">
            <ArrowLeft size={16} aria-hidden="true" />
            Back to VisionRead
          </Link>
          <p className="vr-eyebrow">Live device workflow</p>
          <h1>Watch the wearable pipeline run.</h1>
          <p className="vr-live-page__intro">
            Mock status from the integrated Rashmi → Manoj → Abhishek → Harshaka flow.
            Connect your Pi backend later — the UI is ready for live updates.
          </p>
        </Reveal>

        <LiveDeviceWorkflow />
      </section>
    </SiteShell>
  );
}
