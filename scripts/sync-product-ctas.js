#!/usr/bin/env node
// Keep the paid offer present in static HTML, including regenerated pages.
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const stylesheet = '<link rel="stylesheet" href="/assets/css/product-cta.css">';
const menu = '<a class="nav-toolkit" href="https://buy.heytensor.com/">Debug Toolkit <span>· $29</span></a>';
function offer(file) {
  const home = file === 'index.html';
  const error = /error|mat1|dtype|device|cuda|target|mismatch/.test(file);
  const title = home ? 'Turn your next tensor error into a repeatable check.' : error ? 'Keep the fix. Catch the mismatch next time.' : 'Take the next step in your own PyTorch model.';
  return `\n<section class="product-cta${home ? ' product-cta-home' : ''}" aria-labelledby="product-cta-title">
 <div class="product-cta-copy"><p class="product-cta-kicker">TENSOR PREFLIGHT · DOWNLOADABLE PYTORCH TOOLKIT</p>
 <h2 id="product-cta-title">${title}</h2>
 <p>Trace a local forward pass, inspect tensor shapes in an HTML report, and turn a repair into a regression check. Includes eight broken-and-fixed workflows, source code and setup instructions.</p>
 <p class="product-cta-requirements">Python 3.9+ · PyTorch 2.8+ · Runs locally</p></div>
 <div class="product-cta-actions"><a class="product-cta-buy" href="https://buy.heytensor.com/">Get Tensor Preflight — $29 <span aria-hidden="true">→</span></a>
 <span class="product-cta-payment">One-time payment · ZIP download</span>
 <a class="product-cta-sample" href="https://buy.heytensor.com/sample-report.html">See a sample report</a></div>
</section>\n`;
}
function sync() {
  let pages = 0;
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, {withFileTypes:true})) {
      if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
      const full = path.join(dir,entry.name);
      if (entry.isDirectory()) { walk(full); continue; }
      if (!entry.name.endsWith('.html')) continue;
      const rel = path.relative(ROOT,full).replace(/\\/g,'/');
      let html = fs.readFileSync(full,'utf8');
      if (!/<header[\s>]/.test(html)) continue;
      const before = html;
      if (!html.includes('class="nav-toolkit"')) {
        html = html.replace(/(<header[\s\S]*?<nav(?:\s[^>]*)?>)/, '$1\n '+menu);
      }
      if (!html.includes('/assets/css/product-cta.css')) html = html.replace('</head>',stylesheet+'\n</head>');
      if (!html.includes('/assets/js/product-navigation.js')) html = html.replace('</head>', '<script defer src="/assets/js/product-navigation.js"></script>\n</head>');
      // The four existing contextual offers already occupy a useful content position.
      if (!html.includes('id="tensor-preflight-offer"') && !html.includes('id="product-cta-title"')) {
        if (rel === 'index.html') html = html.replace(/(<section class="hero">[\s\S]*?<\/section>)/,'$1\n'+offer(rel));
        else if (html.includes('</main>')) html = html.replace('</main>',offer(rel)+'\n</main>');
        else html = html.replace(/<footer[\s>]/, '<div class="container">'+offer(rel)+'</div>\n$&');
      }
      html = html.replace(/[ \t]+\n(?=<section class="product-cta)/g, "\n");
      if (html !== before) fs.writeFileSync(full,html);
      pages++;
    }
  }
  walk(ROOT);
  console.log(`Product menu and CTA coverage: ${pages} HTML pages/templates`);
}
module.exports = sync;
if (require.main === module) sync();
