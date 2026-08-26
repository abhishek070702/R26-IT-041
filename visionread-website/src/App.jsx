import { useState } from "react";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import {
  ArrowRight,
  BookOpen,
  Brain,
  Camera,
  ChevronRight,
  Eye,
  FileText,
  Layers,
  Mic,
  Newspaper,
  Plus,
  Radio,
  ScanLine,
  Volume2,
} from "lucide-react";
import "./App.css";
import Reveal from "./components/Reveal";
import SiteShell from "./components/SiteShell";
import {
  DEMO_FORBES_MAGAZINE,
  LOGO_ICON,
  LOGO_WORDMARK,
  PRODUCT_HEADSET,
  PRODUCT_WORN,
} from "./constants";
import LiveDemoPage from "./pages/LiveDemoPage";

const STATEMENT = [
  "VisionRead seamlessly blends camera sensing with document intelligence.",
  "Novels, newspapers, magazines, reports, and printed letters — all recognised automatically.",
  "Assistive reading, reimagined for the wearable era.",
];

const DOCUMENT_TYPES = [
  "Novel",
  "Newspaper",
  "Magazine",
  "Report",
  "Printed Letter",
];

const DEVICE_CALLOUTS = [
  {
    title: "RFID Reader",
    desc: "Loads your personal reading profile the moment you scan your card.",
  },
  {
    title: "Pi Camera Module",
    desc: "Captures pages with real-time alignment guidance and auto-capture.",
  },
  {
    title: "On-device AI",
    desc: "Classifies Novel, Newspaper, Magazine, Report, and Printed Letter on every capture.",
  },
  {
    title: "Wearable Enclosure",
    desc: "Raspberry Pi 5 prototype designed for hands-free reading sessions.",
  },
];

const EXPERIENCE = [
  {
    id: "guidance",
    eyebrow: "Capture",
    headline: ["Point. Align.", "Capture perfectly."],
    body: "YOLO-based camera guidance helps visually impaired users position any supported material — novels, newspapers, magazines, reports, or printed letters — with spoken cues until the page is ready to capture.",
    icon: Camera,
    visual: "guidance",
  },
  {
    id: "identification",
    eyebrow: "Identification",
    headline: ["Know what", "you're holding."],
    body: "MobileNetV2 classifies every capture into one of five types: Novel, Newspaper, Magazine, Report, or Printed Letter. EasyOCR and Florence read titles, mastheads, and cover images — so every session starts with clarity.",
    icon: Eye,
    visual: "identify",
    reverse: true,
  },
  {
    id: "content",
    eyebrow: "Content",
    headline: ["Every page.", "Every format."],
    body: "For newspapers and magazines, Tesseract OCR and DocLayout-YOLO split pages into articles with categories like Sports or Politics. For reports and letters, full-page OCR extracts structured text. Novels are read as story content — summary or full text.",
    icon: Layers,
    visual: "content",
  },
  {
    id: "voice",
    eyebrow: "Voice Output",
    headline: ["Your content.", "Your voice."],
    body: "Choose summary or full text. Personalized TTS — tuned to your reading level, speed, and tone — reads selected content aloud through the wearable speaker.",
    icon: Volume2,
    visual: "voice",
    reverse: true,
  },
];

const WORKFLOW = [
  "RFID Scan",
  "Camera Guidance",
  "Page Capture",
  "Document Analysis",
  "Content Processing",
  "Voice Output",
];

const FEATURES = [
  { title: "RFID user profiles", icon: Radio },
  { title: "Camera positioning guidance", icon: Camera },
  { title: "Document type detection", icon: FileText },
  { title: "Title, masthead & cover reading", icon: BookOpen },
  { title: "Image description", icon: Eye },
  { title: "OCR & article detection", icon: ScanLine },
  { title: "Category selection", icon: Layers },
  { title: "Summary or full text", icon: Brain },
  { title: "Personalized TTS", icon: Volume2 },
];

const MATERIALS = [
  {
    name: "Novel",
    icon: BookOpen,
    note: "Cover title, illustration description, and story pages",
  },
  {
    name: "Newspaper",
    icon: Newspaper,
    note: "Masthead name, article detection, and category selection",
  },
  {
    name: "Magazine",
    icon: Layers,
    note: "Cover title, multi-section layout, and article categories",
  },
  {
    name: "Report",
    icon: FileText,
    note: "Document type detection and full-page OCR reading",
  },
  {
    name: "Printed Letter",
    icon: Mic,
    note: "Letter recognition and full text voice output",
  },
];

const TEAM = [
  {
    name: "Rashmi",
    focus: "RFID, preferences, voice input, TTS",
  },
  {
    name: "Manoj",
    focus: "Camera guidance and auto capture",
  },
  {
    name: "Abhishek",
    focus: "Document identification, title detection, image description",
  },
  {
    name: "Harshaka",
    focus: "OCR, article detection, categories, summarization",
  },
];

const TECH = [
  { label: "Edge hardware", value: "Raspberry Pi 5 · Pi Camera · RFID Reader" },
  { label: "Vision & layout", value: "YOLO · OpenCV · MobileNetV2 · DocLayout-YOLO" },
  { label: "Text & description", value: "EasyOCR · Tesseract · Florence" },
  { label: "Backend & AI", value: "Python · FastAPI · Ollama / Llama · SQLite" },
];

const DEMO = [
  {
    title: "Novel",
    steps: [
      "Scan RFID card",
      "Capture book cover with guidance",
      "Hear: Novel detected, title, and cover description",
      "Turn page — capture story content",
      "Story category selected automatically",
      "Listen to summary or full text",
    ],
  },
  {
    title: "Newspaper",
    steps: [
      "Scan RFID card",
      "Capture newspaper page",
      "Hear: Newspaper detected and masthead name",
      "Browse article categories (Sports, Politics, etc.)",
      "Select category by voice",
      "Listen to summary or full article",
    ],
  },
  {
    title: "Magazine",
    steps: [
      "Scan RFID card",
      "Capture magazine cover or inner page",
      "Hear: Magazine detected, title, and cover image",
      "Article regions and categories detected",
      "Select section or category",
      "Listen to summary or full text",
    ],
  },
  {
    title: "Report",
    steps: [
      "Scan RFID card",
      "Capture report page with camera guidance",
      "Hear: Report detected",
      "Full-page OCR extracts document text",
      "Choose summary or full reading",
      "Rashmi reads content with your TTS preferences",
    ],
  },
  {
    title: "Printed Letter",
    steps: [
      "Scan RFID card",
      "Capture letter with alignment guidance",
      "Hear: Printed Letter detected",
      "OCR reads the full letter content",
      "Choose summary or full text",
      "Personalized voice output reads the letter aloud",
    ],
  },
];

function ProductPhoto({ src, alt, className }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className={`vr-photo-fallback ${className || ""}`} aria-label={alt}>
        <img src={LOGO_ICON} alt="" className="vr-logo vr-logo--fallback" />
        <span>Product photo</span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

function DeviceVisual({ variant }) {
  if (variant === "guidance") {
    return (
      <div className="vr-visual vr-visual--guidance vr-visual--animated">
        <div className="vr-visual__scan" aria-hidden="true" />
        <div className="vr-visual__frame">
          <div className="vr-visual__corner vr-visual__corner--tl" />
          <div className="vr-visual__corner vr-visual__corner--tr" />
          <div className="vr-visual__corner vr-visual__corner--bl" />
          <div className="vr-visual__corner vr-visual__corner--br" />
          <p>Align page within frame</p>
        </div>
      </div>
    );
  }
  if (variant === "identify") {
    return (
      <div className="vr-visual vr-visual--identify vr-visual--identify-photo vr-visual--animated">
        <div className="vr-visual__identify-frame">
          <ProductPhoto
            src={DEMO_FORBES_MAGAZINE}
            alt="Forbes magazine cover identified as Magazine by VisionRead"
            className="vr-photo vr-photo--identify"
          />
          <span className="vr-visual__chip vr-visual__chip--pulse">Magazine detected</span>
          <p className="vr-visual__identify-caption">
            Cover shows a portrait with bold headline text.
          </p>
        </div>
      </div>
    );
  }
  if (variant === "content") {
    return (
      <div className="vr-visual vr-visual--content vr-visual--animated">
        {["Sports", "Politics", "Business", "Weather"].map((c, i) => (
          <span key={c} style={{ "--tag-i": i }}>{c}</span>
        ))}
      </div>
    );
  }
  return (
    <div className="vr-visual vr-visual--voice">
      <Volume2 size={32} strokeWidth={1.5} />
      <p>&ldquo;Reading Sports summary&hellip;&rdquo;</p>
      <div className="vr-visual__wave">
        {[1, 2, 3, 4, 5, 6, 7].map((i) => (
          <i key={i} style={{ animationDelay: `${i * 0.08}s` }} />
        ))}
      </div>
    </div>
  );
}

function HomePage() {
  return (
    <SiteShell activePage="home">
        {/* Hero — Meta / Apple product launch style */}
        <section className="vr-hero" id="top">
          <p className="vr-hero__eyebrow vr-enter" style={{ "--enter-i": 0 }}>Assistive AI Reader · Research Prototype</p>
          <h1 className="vr-hero__brand vr-enter" style={{ "--enter-i": 1 }}>
            <img
              src={LOGO_WORDMARK}
              alt="VisionRead — Assistive AI Reader"
              className="vr-logo vr-logo--hero"
            />
          </h1>
          <p className="vr-hero__headline vr-enter" style={{ "--enter-i": 2 }}>
            AI-powered wearable reading for visually impaired users
          </p>
          <p className="vr-hero__sub vr-enter" style={{ "--enter-i": 3 }}>
            VisionRead recognises novels, newspapers, magazines, reports, and
            printed letters — then guides capture, reads titles, describes images,
            detects articles, and speaks selected content through personalized voice output.
          </p>

          <div className="vr-hero__types">
            {DOCUMENT_TYPES.map((type, i) => (
              <span key={type} className="vr-enter" style={{ "--enter-i": 4 + i }}>{type}</span>
            ))}
          </div>

          <div className="vr-hero__actions vr-enter" style={{ "--enter-i": 9 }}>
            <a href="#overview" className="vr-link vr-link--primary">
              Take a closer look
              <ChevronRight size={18} />
            </a>
            <Link to="/live-demo" className="vr-link">
              Open live demo
            </Link>
          </div>

          <div className="vr-hero__product vr-enter vr-enter--float" style={{ "--enter-i": 10 }}>
            <div className="vr-hero__glow" aria-hidden="true" />
            <ProductPhoto
              src={PRODUCT_HEADSET}
              alt="VisionRead wearable reading assistant prototype headset"
              className="vr-photo vr-photo--hero"
            />
            <p className="vr-hero__product-caption">
              Research prototype hardware · camera · RFID · integrated audio
            </p>
          </div>

          <ul className="vr-hero__statement">
            {STATEMENT.map((line, i) => (
              <li key={line} className="vr-enter" style={{ "--enter-i": 11 + i }}>{line}</li>
            ))}
          </ul>
        </section>

        {/* Problem — Apple-style single statement */}
        <Reveal as="section" className="vr-statement" variant="scale" id="problem">
          <p className="vr-statement__eyebrow">The challenge</p>
          <h2>
            Printed content is still difficult
            <br />
            to access independently.
          </h2>
          <p className="vr-statement__body">
            VisionRead is a smart wearable reading assistant that helps visually
            impaired users identify reading materials — whether a novel, newspaper,
            magazine, report, or printed letter — capture pages correctly, understand
            document type and titles, describe images, detect article categories where
            applicable, and listen to selected content through voice output.
          </p>
        </Reveal>

        {/* Take a closer look — Apple Vision Pro style */}
        <section className="vr-closer" id="overview">
          <Reveal variant="up">
            <h2 className="vr-closer__title">Take a closer look.</h2>
          </Reveal>

          <div className="vr-closer__grid">
            <Reveal variant="left" delay={80} className="vr-closer__device">
              <ProductPhoto
                src={PRODUCT_HEADSET}
                alt="VisionRead headset front view with integrated headphones"
                className="vr-photo vr-photo--closer vr-photo--tilt"
              />
            </Reveal>

            <div className="vr-closer__details">
              {DEVICE_CALLOUTS.map((item, i) => (
                <Reveal key={item.title} as="article" className="vr-closer__item" variant="right" delay={i * 90}>
                  <button type="button" className="vr-closer__toggle" aria-hidden="true">
                    <Plus size={16} />
                  </button>
                  <div>
                    <h3>{item.title}</h3>
                    <p>{item.desc}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* Wearable in use — lifestyle product shot */}
        <section className="vr-wear" id="wearable">
          <Reveal variant="left" className="vr-wear__copy">
            <p className="vr-eyebrow">Wearable experience</p>
            <h2>
              Hands-free reading.
              <br />
              Voice-first output.
            </h2>
            <p>
              The prototype combines a head-mounted camera, integrated headphones,
              and on-device AI so users can capture printed material and hear
              document type, titles, categories, and selected content — without
              holding a phone or separate reader.
            </p>
            <ul className="vr-wear__list">
              {[
                [Radio, "RFID profile loads your preferences"],
                [Camera, "Camera guidance for page alignment"],
                [Volume2, "Personalized TTS reads selected content"],
              ].map(([Icon, text], i) => (
                <li key={text} className="vr-stagger-item" style={{ "--stagger-i": i }}>
                  <Icon size={16} /> {text}
                </li>
              ))}
            </ul>
          </Reveal>
          <Reveal variant="right" delay={120} className="vr-wear__visual">
            <div className="vr-wear__glow" aria-hidden="true" />
            <ProductPhoto
              src={PRODUCT_WORN}
              alt="User wearing the VisionRead assistive reading wearable with audio output"
              className="vr-photo vr-photo--wear vr-photo--float"
            />
          </Reveal>
        </section>

        {/* Solution pillars */}
        <section className="vr-pillars" id="solution">
          <Reveal variant="up" className="vr-pillars__intro">
            <p className="vr-eyebrow">The solution</p>
            <h2>
              One pipeline.
              <br />
              Five integrated stages.
            </h2>
          </Reveal>
          <div className="vr-pillars__row">
            {[
              ["RFID profiles", "Personal settings per user card"],
              ["Camera guidance", "Real-time page alignment"],
              ["Document analysis", "5-type detection, title & images"],
              ["Content processing", "OCR, articles, categories"],
              ["Voice output", "Summary or full text TTS"],
            ].map(([title, desc], i) => (
              <Reveal key={title} as="article" variant="up" delay={i * 70}>
                <h3>{title}</h3>
                <p>{desc}</p>
              </Reveal>
            ))}
          </div>
        </section>

        {/* How it works */}
        <Reveal as="section" className="vr-flow" variant="scale" id="how-it-works">
          <p className="vr-eyebrow vr-flow__eyebrow">How it works</p>
          <h2 className="vr-flow__title">
            From scan to spoken content.
          </h2>
          <ol className="vr-flow__steps">
            {WORKFLOW.map((step, i) => (
              <Reveal key={step} as="li" variant="up" delay={i * 60}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                {step}
              </Reveal>
            ))}
          </ol>
        </Reveal>

        {/* Experience sections — Apple feature blocks */}
        <div id="experience">
          {EXPERIENCE.map((block, blockIndex) => {
            const Icon = block.icon;
            return (
              <section
                key={block.id}
                className={`vr-block ${block.reverse ? "vr-block--reverse" : ""}`}
              >
                <Reveal
                  variant={block.reverse ? "right" : "left"}
                  delay={blockIndex * 40}
                  className="vr-block__copy"
                >
                  <p className="vr-eyebrow">{block.eyebrow}</p>
                  <h2>
                    {block.headline[0]}
                    <br />
                    {block.headline[1]}
                  </h2>
                  <p>{block.body}</p>
                  <span className="vr-block__icon">
                    <Icon size={20} />
                  </span>
                </Reveal>
                <Reveal
                  variant={block.reverse ? "left" : "right"}
                  delay={120 + blockIndex * 40}
                >
                  <DeviceVisual variant={block.visual} />
                </Reveal>
              </section>
            );
          })}
        </div>

        {/* Core features — Meta product grid */}
        <section className="vr-features" id="features">
          <Reveal variant="up" className="vr-features__head">
            <p className="vr-eyebrow">Core features</p>
            <h2>Everything built for independent reading.</h2>
          </Reveal>
          <div className="vr-features__grid">
            {FEATURES.map(({ title, icon: Icon }, i) => (
              <Reveal key={title} as="article" variant="up" delay={i * 50}>
                <Icon size={22} strokeWidth={1.75} />
                <h3>{title}</h3>
              </Reveal>
            ))}
          </div>
        </section>

        {/* Materials — Meta product lineup */}
        <Reveal as="section" className="vr-lineup" variant="scale" id="materials">
          <p className="vr-eyebrow">Supported materials</p>
          <h2>Five document classes. Fully supported.</h2>
          <p className="vr-lineup__sub">
            Trained MobileNetV2 classifier recognises all five formats in real time.
          </p>
          <div className="vr-lineup__cards">
            {MATERIALS.map(({ name, icon: Icon, note }, i) => (
              <Reveal key={name} as="article" variant="up" delay={i * 80}>
                <div className="vr-lineup__icon">
                  <Icon size={28} strokeWidth={1.5} />
                </div>
                <h3>{name}</h3>
                <p>{note}</p>
              </Reveal>
            ))}
          </div>
        </Reveal>

        {/* Team */}
        <section className="vr-team" id="team">
          <Reveal variant="up">
            <p className="vr-eyebrow">Team modules</p>
            <h2>Four specialists. One wearable system.</h2>
          </Reveal>
          <div className="vr-team__grid">
            {TEAM.map((m, i) => (
              <Reveal key={m.name} as="article" variant="up" delay={i * 90}>
                <h3>{m.name}</h3>
                <p>{m.focus}</p>
              </Reveal>
            ))}
          </div>
        </section>

        {/* Technology — Apple tech section */}
        <Reveal as="section" className="vr-tech" variant="scale" id="technology">
          <div className="vr-tech__hero">
            <p className="vr-eyebrow">Technology</p>
            <h2>
              Innovation you can
              <br />
              see, read, and hear.
            </h2>
            <p>
              Spatial experiences on VisionRead are only possible through
              edge hardware, computer vision, and local AI — orchestrated
              as a modular research prototype on Raspberry Pi 5.
            </p>
          </div>
          <div className="vr-tech__specs">
            {TECH.map((row, i) => (
              <Reveal key={row.label} as="article" variant="up" delay={i * 100}>
                <h3>{row.label}</h3>
                <p>{row.value}</p>
              </Reveal>
            ))}
          </div>
          <div className="vr-tech__tags">
            {[
              "Raspberry Pi 5",
              "Pi Camera",
              "RFID Reader",
              "Python",
              "FastAPI",
              "YOLO",
              "OpenCV",
              "MobileNetV2",
              "EasyOCR",
              "Florence",
              "Tesseract OCR",
              "DocLayout-YOLO",
              "Ollama / Llama",
              "SQLite",
            ].map((t, i) => (
              <span key={t} className="vr-tag-animate" style={{ "--tag-i": i }}>{t}</span>
            ))}
          </div>
        </Reveal>

        {/* Demo flows */}
        <section className="vr-demo" id="demo">
          <Reveal variant="up">
            <p className="vr-eyebrow">Demo-ready workflow</p>
            <h2>Five reading paths. Panel-ready.</h2>
            <p className="vr-demo__intro">
              End-to-end demo flows for every supported document class — from RFID scan to spoken output.
            </p>
          </Reveal>
          <div className="vr-demo__grid">
            {DEMO.map((flow, i) => (
              <Reveal key={flow.title} as="article" variant="up" delay={i * 100}>
                <h3>{flow.title}</h3>
                <ol>
                  {flow.steps.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ol>
              </Reveal>
            ))}
          </div>
        </section>

        {/* Research — Apple Values style */}
        <Reveal as="section" className="vr-values" variant="scale" id="research">
          <div className="vr-values__inner">
            <img
              src={LOGO_ICON}
              alt=""
              className="vr-logo vr-logo--research vr-logo--pulse"
              aria-hidden="true"
            />
            <p className="vr-eyebrow">Research project</p>
            <h2>
              Designed to make
              <br />
              reading accessible.
            </h2>
            <p>
              <strong>Smart Wearable Reading Assistant for the Visually Impaired</strong>
              {" "}is a final-year research prototype — not a commercial product.
              Project ID: R26-IT-041. It demonstrates how assistive AI modules
              can work together for independent printed reading.
            </p>
            <div className="vr-values__meta">
              <span>Final-year prototype</span>
              <span>R26-IT-041</span>
              <span>2026</span>
            </div>
            <a href="#top" className="vr-link vr-link--primary">
              Back to top
              <ArrowRight size={18} />
            </a>
          </div>
        </Reveal>
    </SiteShell>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/live-demo" element={<LiveDemoPage />} />
      </Routes>
    </BrowserRouter>
  );
}
