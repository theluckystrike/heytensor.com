# HeyTensor — v3 Design Migration Report (2026-06-26)

Migrated the live **heytensor.com** (150 static HTML pages + vanilla-JS tools, GitHub Pages)
from the old **blue / Space Mono + IBM Plex Sans** design to the new
**v3 amber / DM Sans + JetBrains Mono** design (`Heytensor design project/HeyTensor v3.dc.html`).

## Strategy (why it is low-risk)
The whole site is **CSS-variable-driven** and `app.js` renders the tool via **CSS class names**
(not inline styles). So the migration is a **design-token swap + component refinement + font swap** —
**zero tool-logic changes**. The new design was applied to all 150 pages by editing the shared
`assets/style.css` once, plus a sitewide font-link swap.

## What changed
- **`assets/style.css`** — `:root` tokens (amber `#F59E0B` accent, refined dark palette
  `#0B0B0D`/`#131315`/`#1F1F23`, DM Sans + JetBrains Mono), and component refinements:
  - Gradient "T" logo badge injected via `.logo::before` (no per-page markup edit)
  - Underline tabs (v3 signature) instead of filled pills
  - 12px card radius, 16px mono inputs (iOS-zoom-safe) with uppercase labels + amber focus
  - Glowing green/red result card; fluid `clamp()` type for h1/h2/result
  - Native `<details>` FAQ styled as v3 cards
  - **Fixed pre-existing mobile-nav bug**: a later `nav{display:flex}` (V5) overrode the
    responsive `display:none`, leaving the nav always-on on mobile → horizontal overflow.
    Added higher-specificity `header nav` collapse so the hamburger works.
- **`assets/tools.css`** — `.tool-tag` blue → amber.
- **`assets/js/components/reference.js`** — ReLU plot color blue→amber; canvas fonts → new families.
- **116 HTML files** — Google Fonts link swapped to DM Sans + JetBrains Mono.
- **3 HTML files** that linked `style.css` but had no font link — font link added.

## SEO / traffic safety (critical — site has live traffic)
- **HTML diffs are provably font-link-only.** `git diff` confirms: every removed line is the old
  font link; every added line is a font/preconnect link. **No meta, title, canonical, OG, schema,
  heading, or body content was touched on any page.** URLs unchanged. Content unchanged.

## Validation (all passed)
| Check | Result |
|---|---|
| External JS `node --check` | 0 failures |
| Inline `<script>` parse-gate (67 blocks) | 0 failures |
| Homepage calculator computes (interactive) | ✓ `[1,64,222,222]` green result + viz |
| Conv2d tool page computes | ✓ green result on load |
| Horizontal overflow @ 360 / 390 / 768 | 0 overflowing elements |
| Mobile nav collapses to hamburger | ✓ |
| JSON-LD valid (live pages) | 319/319 (only `templates/_template.html` placeholder excluded) |
| CSS brace balance | OK |
| Core page weight (HTML+CSS+JS) | 85 KB (< 200 KB target) |

## Not done (deliberately — would add risk to a traffic site; optional future enhancements)
- v3 JS-only flourishes: sticky mini-result-on-scroll, vertical-timeline chain mode,
  single bordered "calculator box" wrap. The current result faithfully applies the v3 design
  *language*; these are additive niceties that require `app.js`/markup changes.
- Nav was kept richer than the v3 mockup (Calculator/Tools/Answers/Research/Guides/About/Blog)
  to preserve SEO-valuable internal links.

## Deploy
Repo is GitHub Pages source `theluckystrike/heytensor.com` (branch `main`). Changes are committed
locally in the working clone but **not pushed** — pushing `main` deploys live. Awaiting go-ahead.
