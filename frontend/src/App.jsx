import { useEffect, useState } from "react";
import {
  BookOpen,
  Brain,
  Camera,
  ChevronRight,
  ChevronUp,
  Eye,
  FileText,
  Layers,
  Menu,
  Mic,
  Newspaper,
  Plus,
  Radio,
  ScanLine,
  Volume2,
  X,
} from "lucide-react";
import "./App.css";
import Reveal from "./components/Reveal";
import {
  DEMO_FORBES_MAGAZINE,
  DEMO_NEWSPAPER,
  ERYN_NAME,
  FOOTER_NAV,
  LOGO_ICON,
  LOGO_WORDMARK,
  NAV,
  PRODUCT_HERO,
  PRODUCT_SHOTS,
  PRODUCT_WORN,
} from "./constants";

const DOCUMENT_TYPES = [
  "Novel",
  "Newspaper",
  "Magazine",
  "Report",
  "Printed Letter",
];

const STATEMENT = [
  "VisionMate blends camera sensing with document intelligence.",
  "Novels, newspapers, magazines, reports, and printed letters — recognised automatically.",
  "Assistive reading, reimagined for the wearable era.",
];

const DEVICE_CALLOUTS = [
  {
    title: "User profile",
    desc: "Loads your reading voice, pace, and tone the moment your identity is known.",
  },
  {
    title: "Wearable camera",
    desc: "Guides you until the full page is in view, then captures automatically.",
  },
  {
    title: "On-device AI",
    desc: "Classifies Novel, Newspaper, Magazine, Report, and Printed Letter on every capture.",
  },
  {
    title: "Hands-free audio",
    desc: "Raspberry Pi 5 prototype with headphones that speak type, title, and content.",
  },
];

const EXPERIENCE = [
  {
    id: "guidance",
    eyebrow: "Capture",
    headline: ["Point. Align.", "Capture perfectly."],
    body: "Camera guidance helps visually impaired users position any supported material — novels, newspapers, magazines, reports, or printed letters — with spoken cues until the page is ready.",
    icon: Camera,
    visual: "guidance",
  },
  {
    id: "identification",
    eyebrow: "Identification",
    headline: ["Know what", "you're holding."],
    body: "VisionMate classifies every capture into one of five types, then reads titles, mastheads, and cover images — so every session starts with clarity.",
    icon: Eye,
    visual: "identify",
    reverse: true,
  },
  {
    id: "content",
    eyebrow: "Content",
    headline: ["Every page.", "Every format."],
    body: "Newspapers and magazines split into articles and categories. Reports and letters become full-page text. Novels are read as story — summary or complete.",
    icon: Layers,
    visual: "content",
  },
  {
    id: "voice",
    eyebrow: "Voice output",
    headline: ["Your content.", "Your voice."],
    body: "Choose summary or full text. Personalized speech — tuned to your reading level, speed, and tone — reads selected content through the wearable speaker.",
    icon: Volume2,
    visual: "voice",
    reverse: true,
  },
];

const WORKFLOW = [
  "Load profile",
  "Camera guidance",
  "Page capture",
  "Document analysis",
  "Content processing",
  "Voice output",
];

const FEATURES = [
  { title: "Personal reading profiles", icon: Radio },
  { title: "Camera positioning guidance", icon: Camera },
  { title: "Document type detection", icon: FileText },
  { title: "Title, masthead & cover reading", icon: BookOpen },
  { title: "Image description", icon: Eye },
  { title: "OCR & article detection", icon: ScanLine },
  { title: "Category selection", icon: Layers },
  { title: "Summary or full text", icon: Brain },
  { title: "Personalized voice", icon: Volume2 },
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
    note: "Document type detection and full-page reading",
  },
  {
    name: "Printed Letter",
    icon: Mic,
    note: "Letter recognition and full text voice output",
  },
];

const TEAM = [
  {
    id: "harshaka",
    name: "Harshaka",
    role: "Content processing",
    focus: "OCR, article detection, categories, and summarization",
    photo: "/images/team-harshaka.jpg?v=1",
  },
  {
    id: "manoj",
    name: "Manoj",
    role: "Camera guidance",
    focus: "Page alignment, wearable capture, and auto-capture",
    photo: "/images/team-manoj.jpg?v=1",
  },
  {
    id: "rashmi",
    name: "Rashmi",
    role: "Voice & identity",
    focus: "Preferences, voice input, and personalized speech",
    photo: "/images/team-rashmi.jpg?v=1",
  },
  {
    id: "abhishek",
    name: "Abhishek Chithrasena",
    role: "Vision analysis",
    focus: "Document type, title detection, and image description",
    photo: "/images/team-abhishek.jpg?v=1",
  },
];

const TECH = [
  { label: "Edge hardware", value: "Raspberry Pi 5 · camera · wearable audio" },
  { label: "Vision & layout", value: "YOLO · OpenCV · MobileNetV2 · DocLayout-YOLO" },
  { label: "Text & description", value: "EasyOCR · Tesseract · Florence" },
  { label: "Speech & AI", value: "Python · FastAPI · local models · personalized TTS" },
];

const DEMO = [
  {
    title: "Novel",
    steps: [
      "Load your reading profile",
      "Capture the book cover with guidance",
      "Hear: Novel detected, title, and cover description",
      "Turn the page — capture story content",
      "Story category selected automatically",
      "Listen to summary or full text",
    ],
  },
  {
    title: "Newspaper",
    steps: [
      "Load your reading profile",
      "Capture a newspaper page",
      "Hear: Newspaper detected and masthead name",
      "Browse article categories",
      "Select a category by voice",
      "Listen to summary or full article",
    ],
  },
  {
    title: "Magazine",
    steps: [
      "Load your reading profile",
      "Capture the cover or an inner page",
      "Hear: Magazine detected, title, and cover image",
      "Article regions and categories detected",
      "Select a section or category",
      "Listen to summary or full text",
    ],
  },
  {
    title: "Report",
    steps: [
      "Load your reading profile",
      "Capture a report page with camera guidance",
      "Hear: Report detected",
      "Full-page text is extracted",
      "Choose summary or full reading",
      "Hear it in your preferred voice",
    ],
  },
  {
    title: "Printed Letter",
    steps: [
      "Load your reading profile",
      "Capture the letter with alignment guidance",
      "Hear: Printed Letter detected",
      "The full letter is read as text",
      "Choose summary or full text",
      "Personalized voice reads the letter aloud",
    ],
  },
];

const MARQUEE_GROUP = [...DOCUMENT_TYPES, ...DOCUMENT_TYPES, ...DOCUMENT_TYPES];

function setSpotlight(event) {
  const node = event.currentTarget;
  const box = node.getBoundingClientRect();
  node.style.setProperty("--mx", `${event.clientX - box.left}px`);
  node.style.setProperty("--my", `${event.clientY - box.top}px`);
}

function ProductPhoto({ src, alt, className }) {
  return (
    <img
      src={src}
      alt={alt}
      className={className}
      loading="lazy"
    />
  );
}

function CloserLook() {
  const [active, setActive] = useState(0);

  return (
    <section className="vr-closer" id="overview">
      <Reveal variant="up">
        <h2 className="vr-closer__title">Take a closer look.</h2>
      </Reveal>
      <div className="vr-closer__grid">
        <Reveal variant="left" delay={80} className="vr-closer__device">
          <div className="vr-closer__stage">
            {PRODUCT_SHOTS.map((item, index) => (
              <img
                key={item.src}
                src={item.src}
                alt={index === active ? item.alt : ""}
                className={`vr-photo vr-photo--closer ${index === active ? "is-active" : ""}`}
              />
            ))}
          </div>
          <div className="vr-shots" role="tablist" aria-label="Product photos">
            {PRODUCT_SHOTS.map((item, index) => (
              <button
                key={item.src}
                type="button"
                role="tab"
                aria-selected={index === active}
                className={`vr-shots__thumb ${index === active ? "vr-shots__thumb--active" : ""}`}
                onClick={() => setActive(index)}
              >
                <span className="vr-shots__frame">
                  <img src={item.src} alt="" />
                </span>
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </Reveal>
        <div className="vr-closer__details">
          {DEVICE_CALLOUTS.map((item, index) => (
            <Reveal
              key={item.title}
              as="article"
              className="vr-closer__item"
              variant="right"
              delay={index * 90}
            >
              <button type="button" className="vr-closer__toggle" aria-hidden="true" tabIndex={-1}>
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
  );
}

function DeviceVisual({ variant }) {
  if (variant === "guidance") {
    return (
      <div
        className="vr-visual vr-visual--guidance vr-visual--animated vr-visual--spotlight"
        onMouseMove={setSpotlight}
      >
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
      <div
        className="vr-visual vr-visual--identify vr-visual--identify-photo vr-visual--animated vr-visual--spotlight"
        onMouseMove={setSpotlight}
      >
        <div className="vr-visual__identify-frame">
          <img
            src={DEMO_FORBES_MAGAZINE}
            alt="Forbes magazine cover: A More Accessible Tomorrow, identified as Magazine"
            className="vr-photo vr-photo--identify"
          />
          <span className="vr-visual__chip vr-visual__chip--pulse">Magazine detected</span>
          <p className="vr-visual__identify-caption">
            Cover identified. Headline and portrait described aloud.
          </p>
        </div>
      </div>
    );
  }

  if (variant === "content") {
    const regions = [
      { top: "18%", left: "8%", width: "28%", height: "42%" },
      { top: "18%", left: "40%", width: "24%", height: "36%" },
      { top: "58%", left: "8%", width: "56%", height: "28%" },
    ];
    const categories = ["Sports", "Politics", "Business", "Weather"];

    return (
      <div className="vr-visual vr-visual--content vr-visual--articles vr-visual--animated">
        <img
          src={DEMO_NEWSPAPER}
          alt=""
          className="vr-visual__paper"
          aria-hidden="true"
        />
        <div className="vr-visual__paper-veil" aria-hidden="true" />
        <div className="vr-visual__scan" aria-hidden="true" />
        {regions.map((region, index) => (
          <span
            key={index}
            className="vr-visual__region"
            style={{
              top: region.top,
              left: region.left,
              width: region.width,
              height: region.height,
              "--region-i": index,
            }}
          />
        ))}
        <span className="vr-visual__chip vr-visual__chip--pulse">Articles detected</span>
        <div className="vr-visual__cats">
          {categories.map((label, index) => (
            <span key={label} style={{ "--tag-i": index }}>
              {label}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="vr-visual vr-visual--voice">
      <Volume2 size={32} strokeWidth={1.5} />
      <p>&ldquo;Reading Sports summary&hellip;&rdquo;</p>
      <div className="vr-visual__wave">
        {[1, 2, 3, 4, 5, 6, 7].map((index) => (
          <i key={index} style={{ animationDelay: `${index * 0.08}s` }} />
        ))}
      </div>
    </div>
  );
}

function DemoForm() {
  const [status, setStatus] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.checkValidity()) {
      setStatus("Please complete the required fields.");
      return;
    }
    form.reset();
    setStatus(
      "Thank you. This product site does not send data. Contact the VisionMate team to book a live demonstration.",
    );
  }

  return (
    <form className="vr-form" onSubmit={handleSubmit} noValidate={false}>
      <label>
        Name
        <input name="name" type="text" required autoComplete="name" />
      </label>
      <label>
        Email
        <input name="email" type="email" required autoComplete="email" />
      </label>
      <label>
        Organization
        <input name="org" type="text" autoComplete="organization" />
      </label>
      <label>
        Message
        <textarea name="message" required placeholder="Tell us about your demo, panel, or research visit." />
      </label>
      <button className="vr-btn vr-btn--primary" type="submit">
        Request a demo
      </button>
      {status ? <p className="vr-form__status">{status}</p> : (
        <p className="vr-form__note">Display only — nothing is sent to a server.</p>
      )}
    </form>
  );
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [navSolid, setNavSolid] = useState(false);
  const [progress, setProgress] = useState(0);
  const [active, setActive] = useState("#overview");

  useEffect(() => {
    const onScroll = () => {
      const scrolled = window.scrollY;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      setNavSolid(scrolled > 24);
      setProgress(max > 0 ? (scrolled / max) * 100 : 0);

      const ids = ["overview", "experience", "technology", "demo", "request"];
      let current = "#overview";
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top < 140) {
          current = `#${id}`;
        }
      }
      setActive(current);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  const closeMenu = () => setMenuOpen(false);

  const goTo = (event, href) => {
    closeMenu();
    if (!href.startsWith("#")) return;
    event.preventDefault();
    if (window.location.hash !== href) {
      history.pushState(null, "", href);
    }
    const el = document.getElementById(href.slice(1));
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="vr-app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <div className="vr-progress" style={{ "--progress": `${progress}%` }} />

      <div className="vr-bg" aria-hidden="true">
        <span className="vr-bg__orb vr-bg__orb--1" />
        <span className="vr-bg__orb vr-bg__orb--2" />
        <span className="vr-bg__orb vr-bg__orb--3" />
        <span className="vr-bg__grid" />
        <span className="vr-bg__grain" />
      </div>

      <header className={`vr-nav ${navSolid ? "vr-nav--scrolled" : "vr-nav--at-top"}`}>
        <a href="#top" className="vr-nav__logo" onClick={(event) => goTo(event, "#top")}>
          <img src={LOGO_ICON} alt="VisionMate" className="vr-logo vr-logo--nav-icon" />
        </a>

        <nav className="vr-nav__center" aria-label="Main">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className={active === item.href ? "vr-nav__link--active" : undefined}
              onClick={(event) => goTo(event, item.href)}
            >
              {item.label}
            </a>
          ))}
        </nav>

        <div className="vr-nav__right">
          <a
            href="#research"
            className="vr-nav__pill"
            onClick={(event) => goTo(event, "#research")}
          >
            Research Prototype
          </a>
          <a href="#request" className="vr-nav__cta" onClick={(event) => goTo(event, "#request")}>
            Request a demo
          </a>
          <button
            type="button"
            className="vr-nav__menu"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </header>

      <div className={`vr-drawer ${menuOpen ? "vr-drawer--open" : ""}`}>
        {NAV.map((item) => (
          <a key={item.href} href={item.href} onClick={(event) => goTo(event, item.href)}>
            {item.label}
            <ChevronRight size={18} />
          </a>
        ))}
        <a href="#request" onClick={(event) => goTo(event, "#request")}>
          Request a demo
          <ChevronRight size={18} />
        </a>
      </div>

      <div className="vr">
        <main id="main">
          <section className="vr-hero" id="top">
            <p className="vr-hero__eyebrow vr-enter" style={{ "--enter-i": 0 }}>
              Assistive AI Reader · Research Prototype
            </p>
            <h1 className="vr-hero__brand vr-enter" style={{ "--enter-i": 1 }}>
              <img
                src={LOGO_WORDMARK}
                alt="VisionMate — Assistive AI Reader"
                className="vr-logo vr-logo--hero"
              />
            </h1>
            <p className="vr-hero__title vr-enter" style={{ "--enter-i": 2 }}>
              The page, spoken.
            </p>
            <p className="vr-hero__sub vr-enter" style={{ "--enter-i": 3 }}>
              Hold a book, newspaper, magazine, report, or letter. VisionMate finds
              the full page, understands what it is, and reads it in the voice you chose.
            </p>

            <div className="vr-hero__types">
              {DOCUMENT_TYPES.map((type, index) => (
                <span key={type} className="vr-enter" style={{ "--enter-i": 4 + index }}>
                  {type}
                </span>
              ))}
            </div>

            <div className="vr-hero__actions vr-enter" style={{ "--enter-i": 9 }}>
              <a href="#request" className="vr-btn vr-btn--primary">
                Request a demo
                <ChevronRight size={18} />
              </a>
              <a href="#overview" className="vr-btn vr-btn--ghost">
                Take a closer look
              </a>
            </div>

            <div className="vr-hero__product vr-enter vr-enter--float" style={{ "--enter-i": 10 }}>
              <div className="vr-hero__glow" aria-hidden="true" />
              <ProductPhoto
                src={PRODUCT_HERO}
                alt="VisionMate wearable reading headset with glowing sensor bar"
                className="vr-photo vr-photo--hero"
              />
              <p className="vr-hero__product-caption">
                Research prototype · wearable camera · personalized audio
              </p>
            </div>

            <ul className="vr-hero__statement">
              {STATEMENT.map((line, index) => (
                <li key={line} className="vr-enter" style={{ "--enter-i": 12 + index }}>
                  {line}
                </li>
              ))}
            </ul>
          </section>

          <div className="vr-marquee" aria-hidden="true">
            <div className="vr-marquee__track">
              {[0, 1].map((copy) => (
                <div className="vr-marquee__group" key={copy}>
                  {MARQUEE_GROUP.map((type, index) => (
                    <span key={`${copy}-${type}-${index}`}>{type}</span>
                  ))}
                </div>
              ))}
            </div>
          </div>

          <Reveal as="section" className="vr-statement" variant="scale" id="problem">
            <p className="vr-statement__eyebrow">The challenge</p>
            <h2>
              Printed content is still difficult
              <br />
              to access independently.
            </h2>
            <p className="vr-statement__body">
              VisionMate is a smart wearable reading assistant that helps visually
              impaired users identify reading materials, capture pages correctly,
              understand document type and titles, describe images, detect article
              categories, and listen to selected content through voice output.
            </p>
          </Reveal>

          <CloserLook />

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
                holding a phone or a separate reader.
              </p>
              <ul className="vr-wear__list">
                {[
                  [Radio, "Your profile loads personal preferences"],
                  [Camera, "Spoken guidance until the page is complete"],
                  [Volume2, "Personalized voice reads selected content"],
                ].map(([Icon, text], index) => (
                  <li key={text} className="vr-stagger-item" style={{ "--stagger-i": index }}>
                    <Icon size={16} /> {text}
                  </li>
                ))}
              </ul>
            </Reveal>
            <Reveal variant="right" delay={120} className="vr-wear__visual">
              <div className="vr-wear__glow" aria-hidden="true" />
              <ProductPhoto
                src={PRODUCT_WORN}
                alt="Person wearing the VisionMate assistive reading headset"
                className="vr-photo vr-photo--wear vr-photo--float"
              />
            </Reveal>
          </section>

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
                ["Identity", "Personal settings for every user"],
                ["Camera guidance", "Real-time page alignment"],
                ["Document analysis", "Five-type detection, title & images"],
                ["Content processing", "OCR, articles, categories"],
                ["Voice output", "Summary or full text speech"],
              ].map(([title, desc], index) => (
                <Reveal key={title} as="article" variant="up" delay={index * 70}>
                  <h3>{title}</h3>
                  <p>{desc}</p>
                </Reveal>
              ))}
            </div>
          </section>

          <Reveal as="section" className="vr-flow" variant="scale" id="how-it-works">
            <p className="vr-eyebrow vr-flow__eyebrow">How it works</p>
            <h2 className="vr-flow__title">From scan to spoken content.</h2>
            <ol className="vr-flow__steps">
              {WORKFLOW.map((step, index) => (
                <Reveal key={step} as="li" variant="up" delay={index * 60}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {step}
                </Reveal>
              ))}
            </ol>
          </Reveal>

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

          <section className="vr-features" id="features">
            <Reveal variant="up" className="vr-features__head">
              <p className="vr-eyebrow">Core features</p>
              <h2>Everything built for independent reading.</h2>
            </Reveal>
            <div className="vr-features__grid">
              {FEATURES.map(({ title, icon: Icon }, index) => (
                <Reveal key={title} as="article" variant="up" delay={index * 50}>
                  <Icon size={22} strokeWidth={1.75} />
                  <h3>{title}</h3>
                </Reveal>
              ))}
            </div>
          </section>

          <Reveal as="section" className="vr-lineup" variant="scale" id="materials">
            <p className="vr-eyebrow">Supported materials</p>
            <h2>Five document classes. Fully supported.</h2>
            <p className="vr-lineup__sub">
              A trained classifier recognises all five formats in real time.
            </p>
            <div className="vr-lineup__cards">
              {MATERIALS.map(({ name, icon: Icon, note }, index) => (
                <Reveal key={name} as="article" variant="up" delay={index * 80}>
                  <div className="vr-lineup__icon">
                    <Icon size={28} strokeWidth={1.5} />
                  </div>
                  <h3>{name}</h3>
                  <p>{note}</p>
                </Reveal>
              ))}
            </div>
          </Reveal>

          <section className="vr-team" id="team">
            <Reveal variant="up">
              <p className="vr-eyebrow">The team</p>
              <h2>The people who built VisionMate.</h2>
              <p className="vr-team__intro">
                Four teammates designed and integrated the wearable pipeline —
                from voice identity to camera capture, vision analysis, and content reading.
              </p>
            </Reveal>
            <div className="vr-team__grid">
              {TEAM.map((member, index) => (
                <Reveal key={member.id} as="article" variant="up" delay={index * 90}>
                  <div className="vr-team__photo-wrap">
                    <img
                      src={member.photo}
                      alt={member.name}
                      className="vr-team__photo"
                    />
                  </div>
                  <span className="vr-team__role">{member.role}</span>
                  <h3>{member.name}</h3>
                  <p>{member.focus}</p>
                </Reveal>
              ))}
            </div>
          </section>

          <Reveal as="section" className="vr-tech" variant="scale" id="technology">
            <div className="vr-tech__hero">
              <p className="vr-eyebrow">Technology</p>
              <h2>
                Innovation you can
                <br />
                see, read, and hear.
              </h2>
              <p>
                Spatial experiences on VisionMate are only possible through
                edge hardware, computer vision, and local AI — orchestrated
                as a modular research prototype on Raspberry Pi 5.
              </p>
            </div>
            <div className="vr-tech__specs">
              {TECH.map((row, index) => (
                <Reveal key={row.label} as="article" variant="up" delay={index * 100}>
                  <h3>{row.label}</h3>
                  <p>{row.value}</p>
                </Reveal>
              ))}
            </div>
            <div className="vr-tech__tags">
              {[
                "Raspberry Pi 5",
                "Wearable camera",
                "Python",
                "FastAPI",
                "YOLO",
                "OpenCV",
                "MobileNetV2",
                "EasyOCR",
                "Florence",
                "Tesseract OCR",
                "DocLayout-YOLO",
                "Personalized TTS",
              ].map((tag, index) => (
                <span key={tag} className="vr-tag-animate" style={{ "--tag-i": index }}>
                  {tag}
                </span>
              ))}
            </div>
          </Reveal>

          <section className="vr-demo" id="demo">
            <Reveal variant="up">
              <p className="vr-eyebrow">Demo-ready workflow</p>
              <h2>Five reading paths. Panel-ready.</h2>
              <p className="vr-demo__intro">
                End-to-end demo flows for every supported document class — from
                profile load to spoken output.
              </p>
            </Reveal>
            <div className="vr-demo__grid">
              {DEMO.map((flow, index) => (
                <Reveal key={flow.title} as="article" variant="up" delay={index * 100}>
                  <h3>{flow.title}</h3>
                  <ol>
                    {flow.steps.map((step) => (
                      <li key={step}>{step}</li>
                    ))}
                  </ol>
                </Reveal>
              ))}
            </div>
          </section>

          <section className="vr-request" id="request">
            <div className="vr-request__inner">
              <Reveal variant="left" className="vr-request__copy">
                <p className="vr-eyebrow">See it in person</p>
                <h2>
                  Request a
                  <br />
                  product demo.
                </h2>
                <p>
                  This is a display website for the VisionMate research prototype.
                  Use the form to note interest — then contact the team to arrange
                  a live demonstration.
                </p>
              </Reveal>
              <Reveal variant="right" delay={80}>
                <DemoForm />
              </Reveal>
            </div>
          </section>

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
              <a href="#top" className="vr-backtop" onClick={(event) => goTo(event, "#top")}>
                <span className="vr-backtop__orb" aria-hidden="true">
                  <ChevronUp size={22} strokeWidth={1.75} />
                </span>
                <span className="vr-backtop__label">Back to top</span>
              </a>
            </div>
          </Reveal>
        </main>

        <footer className="vr-footer">
          <div className="vr-footer__top">
            <div className="vr-footer__brand">
              <img src={LOGO_ICON} alt="" className="vr-logo vr-logo--footer" />
              <div>
                <strong>VisionMate</strong>
                <span>Assistive AI Reader · Research Prototype</span>
              </div>
            </div>
            <nav aria-label="Footer">
              {FOOTER_NAV.map((item) => (
                <a key={item.href} href={item.href} onClick={(event) => goTo(event, item.href)}>
                  {item.label}
                </a>
              ))}
              <a href="#request" onClick={(event) => goTo(event, "#request")}>
                Request a demo
              </a>
            </nav>
          </div>
          <p>
            © 2026 {ERYN_NAME}. All rights reserved. · VisionMate · Smart Wearable
            Reading Assistant · Research Prototype · R26-IT-041
          </p>
          <p className="vr-footer__note">
            This website presents a final-year research prototype. Not available for commercial purchase.
          </p>
        </footer>
      </div>
    </div>
  );
}
