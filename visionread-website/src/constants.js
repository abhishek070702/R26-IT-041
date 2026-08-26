export const PRODUCT_HEADSET = "/images/product-headset.png";
export const PRODUCT_WORN = "/images/product-worn.png";
export const DEMO_FORBES_MAGAZINE = "/images/demo-forbes-magazine.png";
export const LOGO_ICON = "/images/logo-icon.png";
export const LOGO_WORDMARK = "/images/logo-wordmark.png";
export const ERYN_NAME = "Eryn Technologies";

export const NAV = [
  { href: "#overview", label: "Overview" },
  { href: "#experience", label: "Experience" },
  { href: "#technology", label: "Technology" },
  { href: "#demo", label: "Demo" },
];

export const FOOTER_NAV = [
  { href: "#overview", label: "Overview" },
  { href: "#experience", label: "Experience" },
  { href: "#technology", label: "Technology" },
  { href: "#research", label: "Research" },
];

export function resolveNavHref(href, onHomePage) {
  if (href.startsWith("#")) {
    return onHomePage ? href : `/${href}`;
  }
  return href;
}
