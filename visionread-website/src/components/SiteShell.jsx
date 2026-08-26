import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { ChevronRight, Menu, X } from "lucide-react";
import {
  ERYN_NAME,
  FOOTER_NAV,
  LOGO_ICON,
  LOGO_WORDMARK,
  NAV,
  resolveNavHref,
} from "../constants";

function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, [pathname]);

  return null;
}

export default function SiteShell({ activePage = "home", children }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [navSolid, setNavSolid] = useState(false);
  const onHomePage = activePage === "home";

  useEffect(() => {
    history.scrollRestoration = "manual";

    const navEntry = performance.getEntriesByType("navigation")[0];
    const isReload = navEntry?.type === "reload";
    const scrollTop = () => window.scrollTo({ top: 0, left: 0, behavior: "instant" });

    if (isReload && onHomePage) {
      const { pathname, search } = window.location;
      if (window.location.hash) {
        history.replaceState(null, "", pathname + search);
      }
      scrollTop();
      requestAnimationFrame(scrollTop);
    }
  }, [onHomePage]);

  useEffect(() => {
    const onScroll = () => setNavSolid(window.scrollY > 24);
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

  const renderNavLink = (item) => {
    const href = resolveNavHref(item.href, onHomePage);

    if (href.startsWith("/") && !href.startsWith("/#")) {
      return (
        <Link key={item.href} to={href} onClick={closeMenu}>
          {item.label}
        </Link>
      );
    }

    return (
      <a key={item.href} href={href} onClick={closeMenu}>
        {item.label}
      </a>
    );
  };

  return (
    <>
      <ScrollToTop />

      <div className="vr-app">
        <div className="vr-bg" aria-hidden="true">
          <span className="vr-bg__orb vr-bg__orb--1" />
          <span className="vr-bg__orb vr-bg__orb--2" />
          <span className="vr-bg__grid" />
        </div>

        <header
          className={`vr-nav ${navSolid ? "vr-nav--scrolled" : "vr-nav--at-top"}`}
        >
          <Link to="/" className="vr-nav__logo" onClick={closeMenu}>
            <img
              src={LOGO_WORDMARK}
              alt="VisionRead"
              className="vr-logo vr-logo--nav-wordmark"
            />
            <img
              src={LOGO_ICON}
              alt="VisionRead"
              className="vr-logo vr-logo--nav-icon"
            />
          </Link>

          <nav className="vr-nav__center" aria-label="Main">
            {NAV.map(renderNavLink)}
            <Link
              to="/live-demo"
              className={activePage === "live-demo" ? "vr-nav__link--active" : undefined}
              onClick={closeMenu}
            >
              Live Demo
            </Link>
          </nav>

          <div className="vr-nav__right">
            <a href={resolveNavHref("#research", onHomePage)} className="vr-nav__pill">
              Research Prototype
            </a>
            {activePage === "live-demo" ? (
              <Link to="/" className="vr-nav__cta">
                Back to site
              </Link>
            ) : (
              <Link to="/live-demo" className="vr-nav__cta">
                Live Demo
              </Link>
            )}
            <button
              type="button"
              className="vr-nav__menu"
              aria-label={menuOpen ? "Close menu" : "Open menu"}
              onClick={() => setMenuOpen((v) => !v)}
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </header>

        <div className={`vr-drawer ${menuOpen ? "vr-drawer--open" : ""}`}>
          {NAV.map((item) => (
            <a
              key={item.href}
              href={resolveNavHref(item.href, onHomePage)}
              onClick={closeMenu}
            >
              {item.label}
              <ChevronRight size={18} />
            </a>
          ))}
          <Link to="/live-demo" onClick={closeMenu}>
            Live Demo
            <ChevronRight size={18} />
          </Link>
          {activePage === "live-demo" && (
            <Link to="/" onClick={closeMenu}>
              Back to site
              <ChevronRight size={18} />
            </Link>
          )}
        </div>

        <div className="vr vr-ready">
          <main>{children}</main>

          <footer className="vr-footer">
            <div className="vr-footer__top">
              <div className="vr-footer__brand">
                <img src={LOGO_ICON} alt="" className="vr-logo vr-logo--footer" />
                <div>
                  <strong>VisionRead</strong>
                  <span>Assistive AI Reader · Research Prototype</span>
                </div>
              </div>
              <nav aria-label="Footer">
                {FOOTER_NAV.map((item) => (
                  <a key={item.href} href={resolveNavHref(item.href, onHomePage)}>
                    {item.label}
                  </a>
                ))}
                <Link to="/live-demo">Live Demo</Link>
              </nav>
            </div>
            <p>
              © 2026 {ERYN_NAME}. All rights reserved. · VisionRead · Smart Wearable Reading Assistant · Research Prototype · R26-IT-041
            </p>
            <p className="vr-footer__note">
              This website presents a final-year research prototype. Not available for commercial purchase.
            </p>
          </footer>
        </div>
      </div>
    </>
  );
}
