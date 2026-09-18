#!/usr/bin/env node
// Fan-in for 2026-09-18 sprint: add new answer pages to sitemap.xml + answers/index.html
const fs = require('fs');
const path = require('path');
const ROOT = '/Users/mike/Desktop/heytensor-repo';
const TODAY = new Date().toISOString().slice(0, 10);
const NEW = [
  'bert-base-pooler-output-dimension.html',
  'bert-large-340m-parameters.html',
  'bert-base-size-on-disk.html',
  'convtranspose2d-output-size-formula.html',
  'dtype-mismatch-weight-argument-float64.html',
];
const URL = f => `https://heytensor.com/answers/${f}`;

// --- sitemap ---
let sm = fs.readFileSync(path.join(ROOT, 'sitemap.xml'), 'utf8');
let added = 0;
for (const f of NEW) {
  if (sm.includes(URL(f))) continue;
  const entry = `  <url><loc>${URL(f)}</loc><lastmod>${TODAY}</lastmod><changefreq>monthly</changefreq><priority>0.6</priority></url>`;
  sm = sm.replace('</urlset>', entry + '\n</urlset>');
  added++;
}
fs.writeFileSync(path.join(ROOT, 'sitemap.xml'), sm);
console.log(`sitemap: +${added} urls`);

// --- answers index ---
const idxPath = path.join(ROOT, 'answers', 'index.html');
let idx = fs.readFileSync(idxPath, 'utf8');
let idxAdded = 0;
for (const f of NEW) {
  if (idx.includes(`href="/answers/${f}"`)) continue;
  const name = f.replace('.html', '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const li = ` <li><a href="/answers/${f}">${name}</a></li>`;
  // insert before </ul> of the answer list
  idx = idx.replace('</ul>', li + '\n</ul>');
  idxAdded++;
}
fs.writeFileSync(idxPath, idx);
console.log(`answers/index.html: +${idxAdded} links`);

// --- sanity: JSON parse of JSON-LD in each new page, canonical present ---
for (const f of NEW) {
  const html = fs.readFileSync(path.join(ROOT, 'answers', f), 'utf8');
  const canon = html.includes(`rel="canonical" href="${URL(f)}"`);
  const ld = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)];
  let ldOk = true;
  for (const m of ld) { try { JSON.parse(m[1]); } catch (e) { ldOk = false; console.log(`  JSON-LD PARSE FAIL in ${f}: ${e.message}`); } }
  const h1 = (html.match(/<h1[^>]*>([\s\S]*?)<\/h1>/) || [])[1] || '';
  console.log(`${f}: canonical=${canon} jsonldBlocks=${ld.length} jsonldOk=${ldOk} h1="${h1.trim().slice(0, 70)}" colonInH1=${h1.includes(':')}`);
}
