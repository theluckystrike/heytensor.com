#!/usr/bin/env node
'use strict';

var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var ledgerPath = path.join(root, 'tools', 'attention-memory-ledger.html');
var maskPath = path.join(root, 'tools', 'causal-attention-mask-lab.html');
var files = [ledgerPath, maskPath];
var failures = [];
var passed = 0;

function check(label, condition) {
 if (condition) {
  passed += 1;
  console.log('PASS - ' + label);
 } else {
  failures.push(label);
  console.log('FAIL - ' + label);
 }
}

function read(file) { return fs.readFileSync(file, 'utf8'); }
function strippedWords(html) {
 return html.replace(/<script[\s\S]*?<\/script>/gi, ' ')
  .replace(/<style[\s\S]*?<\/style>/gi, ' ')
  .replace(/<[^>]+>/g, ' ').replace(/&[^;]+;/g, ' ')
  .replace(/\s+/g, ' ').trim().split(' ').filter(Boolean).length;
}
function inlineScripts(html) {
 var scripts = [];
 var re = /<script([^>]*)>([\s\S]*?)<\/script>/gi;
 var match;
 while ((match = re.exec(html))) {
  if (!/\bsrc\s*=/.test(match[1]) && !/application\/ld\+json/.test(match[1])) scripts.push(match[2]);
 }
 return scripts;
}

files.forEach(function (file) {
 var html = read(file);
 var name = path.basename(file);
 check(name + ' exists and is substantive', html.length > 15000 && strippedWords(html) > 450);
 check(name + ' has one H1', (html.match(/<h1[\s>]/g) || []).length === 1);
 check(name + ' has canonical, description, Open Graph, Twitter, and JSON-LD', /rel="canonical"/.test(html) && /name="description"/.test(html) && /property="og:title"/.test(html) && /name="twitter:card"/.test(html) && /application\/ld\+json/.test(html));
 check(name + ' has breadcrumb, source link, related tools, and updated date', /class="breadcrumb"/.test(html) && /docs\.pytorch\.org/.test(html) && /Related Tools/i.test(html) && /2026-08-30/.test(html));
 check(name + ' has no template artifacts or banned claims', !/(\{\{|\}\}|TODO|FIXME|PLACEHOLDER|undefined|NaN|guaranteed performance|exact runtime|production[- ]ready|best model|zero overhead)/i.test(html));
 check(name + ' avoids innerHTML', !/\.innerHTML\b/.test(html));
 var ids = Array.from(html.matchAll(/\sid="([^"]+)"/g), function (m) { return m[1]; });
 check(name + ' has unique element ids', ids.length === new Set(ids).size);
 inlineScripts(html).forEach(function (script, index) {
  try { Function(script); check(name + ' inline script ' + (index + 1) + ' parses', true); }
  catch (error) { console.log(error.message); check(name + ' inline script ' + (index + 1) + ' parses', false); }
 });
});

var ledger = read(ledgerPath);
var mask = read(maskPath);
var hub = read(path.join(root, 'tools', 'index.html'));
var sitemap = read(path.join(root, 'sitemap.xml'));
check('acquisition ledger is indexable', /<meta name="robots" content="index, follow">/.test(ledger));
check('product-lane mask lab is noindex, follow', /<meta name="robots" content="noindex, follow">/.test(mask));
check('hub links both sprint pages exactly once', (hub.match(/href="\/tools\/attention-memory-ledger\.html"/g) || []).length === 1 && (hub.match(/href="\/tools\/causal-attention-mask-lab\.html"/g) || []).length === 1);
check('sitemap includes only the indexable sprint page', (sitemap.match(/https:\/\/heytensor\.com\/tools\/attention-memory-ledger\.html/g) || []).length === 1 && !/https:\/\/heytensor\.com\/tools\/causal-attention-mask-lab\.html/.test(sitemap));
check('new pages cross-link each other', /\/tools\/causal-attention-mask-lab\.html/.test(ledger) && /\/tools\/attention-memory-ledger\.html/.test(mask));
['tools/multihead-attention-calculator.html', 'tools/kv-cache-calculator.html', 'tools/attention-head-dimension-calculator/index.html', 'guides/transformer-architecture-visual-guide.html'].forEach(function (relative) {
 var html = read(path.join(root, relative));
 check(relative + ' links both sprint pages', /\/tools\/attention-memory-ledger\.html/.test(html) && /\/tools\/causal-attention-mask-lab\.html/.test(html));
});

console.log('\nRESULT: ' + passed + '/' + (passed + failures.length) + ' checks passed');
if (failures.length) process.exit(1);
