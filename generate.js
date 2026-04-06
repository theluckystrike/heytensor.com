#!/usr/bin/env node
/**
 * generate.js — Static page generator for HeyTensor tool pages.
 * Reads data/_tools.json + templates/_template.html
 * Generates HTML per tool into tools/
 * Updates sitemap.xml with new tool URLs
 */

const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const TOOLS_DIR = path.join(ROOT, 'tools');
const DATA_FILE = path.join(ROOT, 'data', '_tools.json');
const TEMPLATE_FILE = path.join(ROOT, 'templates', '_template.html');
const SITEMAP_FILE = path.join(ROOT, 'sitemap.xml');

// Read inputs
const tools = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
const template = fs.readFileSync(TEMPLATE_FILE, 'utf8');

// Ensure tools directory exists
if (!fs.existsSync(TOOLS_DIR)) fs.mkdirSync(TOOLS_DIR, { recursive: true });

// Read all tools to build a slug->title map for related tools links
const slugMap = {};
tools.forEach(t => { slugMap[t.slug] = { title: t.h1, description: t.description }; });

let generated = 0;

tools.forEach(tool => {
  let html = template;

  // Basic replacements
  html = html.replace(/\{\{TITLE\}\}/g, escHtml(tool.title));
  html = html.replace(/\{\{H1\}\}/g, escHtml(tool.h1));
  html = html.replace(/\{\{DESCRIPTION\}\}/g, escHtml(tool.description));
  html = html.replace(/\{\{SLUG\}\}/g, tool.slug);
  html = html.replace(/\{\{COMPONENT\}\}/g, tool.component);
  html = html.replace(/\{\{CONFIG_JSON\}\}/g, escHtml(JSON.stringify(tool.config)));

  // FAQ — HTML (details/summary)
  const faqHtml = tool.faq.map(faq =>
    `      <details>\n        <summary>${escHtml(faq.q)}</summary>\n        <p>${escHtml(faq.a)}</p>\n      </details>`
  ).join('\n\n');
  html = html.replace('{{FAQ_HTML}}', faqHtml);

  // FAQ — JSON-LD
  const faqJsonLd = tool.faq.map(faq =>
    `      {\n        "@type": "Question",\n        "name": ${JSON.stringify(faq.q)},\n        "acceptedAnswer": {\n          "@type": "Answer",\n          "text": ${JSON.stringify(faq.a)}\n        }\n      }`
  ).join(',\n');
  html = html.replace('{{FAQ_JSONLD}}', '\n' + faqJsonLd + '\n    ');

  // Related tools
  const relatedHtml = (tool.related || []).map(slug => {
    const rel = slugMap[slug];
    if (!rel) return '';
    return `        <a href="/tools/${slug}.html" class="related-card">\n          <h3>${escHtml(rel.title)}</h3>\n          <p>${escHtml(truncate(rel.description, 100))}</p>\n        </a>`;
  }).filter(Boolean).join('\n');
  html = html.replace('{{RELATED_HTML}}', relatedHtml);

  // Write file
  const outPath = path.join(TOOLS_DIR, tool.slug + '.html');
  fs.writeFileSync(outPath, html, 'utf8');
  generated++;
  console.log(`  ✓ tools/${tool.slug}.html`);
});

// Update sitemap.xml
updateSitemap(tools);

console.log(`\nGenerated ${generated} tool pages.`);
console.log('Sitemap updated.');

/* ── Helpers ── */

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function truncate(str, len) {
  if (str.length <= len) return str;
  return str.substring(0, len).replace(/\s+\S*$/, '') + '...';
}

function updateSitemap(tools) {
  let sitemap = fs.readFileSync(SITEMAP_FILE, 'utf8');
  const today = new Date().toISOString().split('T')[0];

  // Remove any existing tool URLs
  sitemap = sitemap.replace(/\s*<url>\s*<loc>https:\/\/heytensor\.com\/tools\/[^<]+<\/loc>[\s\S]*?<\/url>/g, '');

  // Remove tools index URL if exists
  sitemap = sitemap.replace(/\s*<url>\s*<loc>https:\/\/heytensor\.com\/tools\/<\/loc>[\s\S]*?<\/url>/g, '');

  // Build new tool URLs
  let newUrls = `\n  <url>\n<loc>https://heytensor.com/tools/</loc>\n<lastmod>${today}</lastmod>\n<changefreq>weekly</changefreq>\n<priority>0.9</priority>\n</url>`;

  tools.forEach(tool => {
    newUrls += `\n  <url>\n<loc>https://heytensor.com/tools/${tool.slug}.html</loc>\n<lastmod>${today}</lastmod>\n<changefreq>monthly</changefreq>\n<priority>0.7</priority>\n</url>`;
  });

  // Insert before closing </urlset>
  sitemap = sitemap.replace('</urlset>', newUrls + '\n</urlset>');

  fs.writeFileSync(SITEMAP_FILE, sitemap, 'utf8');
}
