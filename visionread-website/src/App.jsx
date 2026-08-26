import { Camera, Eye, FileText, Layers, Radio, Volume2 } from "lucide-react";
import "./App.css";

const features = [
  ["RFID User Profiles", "Loads reading level, voice, tone, and speed preferences.", Radio],
  ["Camera Guidance", "Guides the user to correctly position the reading material.", Camera],
  ["Document Detection", "Detects novels, newspapers, magazines, reports, and letters.", FileText],
  ["Image Description", "Describes important visual content for the user.", Eye],
  ["Article Detection", "Detects newspaper articles and categories.", Layers],
  ["Voice Output", "Reads titles, summaries, full text, and image descriptions aloud.", Volume2],
];

function App() {
  return (
    <main className="site">
      <nav className="nav">
        <div className="logo">
          <div className="logoIcon">
            <Eye size={24} />
          </div>
          <div>
            <h2>VisionRead</h2>
            <p>Assistive AI Reader</p>
          </div>
        </div>

        <div className="links">
          <a href="#features">Features</a>
          <a href="#workflow">Workflow</a>
          <a href="#team">Team</a>
        </div>
      </nav>

      <section className="hero">
        <div className="heroText">
          <span className="badge">AI-powered wearable reading assistant</span>
          <h1>Helping visually impaired users read with confidence.</h1>
          <p>
            VisionRead identifies printed materials, guides camera capture,
            describes images, detects articles, and reads selected content aloud.
          </p>

          <div className="buttons">
            <a className="primary" href="#workflow">Explore Workflow</a>
            <a className="secondary" href="#features">View Features</a>
          </div>

          <div className="stats">
            <div><strong>4</strong><span>Modules</span></div>
            <div><strong>5</strong><span>Document Classes</span></div>
            <div><strong>100%</strong><span>Voice-first</span></div>
          </div>
        </div>

        <div className="deviceCard">
          <div className="device">
            <div className="cameraDot"></div>
            <span>Document detected</span>
            <h3>Newspaper</h3>
            <p>Move closer to one article area for clear reading.</p>
          </div>
        </div>
      </section>

      <section id="features" className="section">
        <div className="heading">
          <span>Core Features</span>
          <h2>Designed for a real assistive reading experience</h2>
        </div>

        <div className="grid">
          {features.map(([title, text, Icon]) => (
            <article className="card" key={title}>
              <div className="icon"><Icon size={24} /></div>
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="workflow" className="section workflow">
        <div>
          <span className="badge">How it works</span>
          <h2>From page capture to spoken content</h2>
          <p>
            The system connects RFID user profiles, Manoj camera guidance,
            Abhishek document analysis, Harshaka content processing, and Rashmi TTS.
          </p>
        </div>

        <div className="steps">
          {["RFID Scan", "Camera Guidance", "Page Capture", "AI Analysis", "Content Reading", "Voice Output"].map((step, index) => (
            <div className="step" key={step}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <p>{step}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="team" className="section">
        <div className="heading">
          <span>Team Modules</span>
          <h2>Four modules working as one wearable device</h2>
        </div>

        <div className="team">
          <div><h3>Rashmi</h3><p>RFID, preferences, voice input, and TTS</p></div>
          <div><h3>Manoj</h3><p>Camera positioning guidance and auto capture</p></div>
          <div><h3>Abhishek</h3><p>Document type, title, and image description</p></div>
          <div><h3>Harshaka</h3><p>OCR, article detection, categories, and summary</p></div>
        </div>
      </section>

      <footer>
        <p>VisionRead © 2026 | Smart Wearable Reading Assistant</p>
      </footer>
    </main>
  );
}

export default App;
