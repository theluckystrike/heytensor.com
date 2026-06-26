/* HeyTensor — sticky mini-result (responsive spec 1.5, keyboard solution B).
   Mirrors the latest computed shape into a bar below the header so the result
   stays visible while editing inputs on mobile. Additive only; never touches
   calculator logic. NASA P10: short functions, bounded queries, guarded access. */
(function () {
  'use strict';

  var bar = null, shapeEl = null;

  function header() { return document.querySelector('header'); }

  function ensureBar() {
    if (bar) return bar;
    bar = document.createElement('div');
    bar.className = 'sticky-result';
    bar.setAttribute('aria-live', 'polite');
    bar.innerHTML = '<span class="sr-label">Output</span><span class="sr-shape"></span>';
    document.body.appendChild(bar);
    shapeEl = bar.querySelector('.sr-shape');
    var h = header();
    bar.style.top = (h ? h.offsetHeight : 52) + 'px';
    return bar;
  }

  function latestResult() {
    var boxes = document.querySelectorAll('.result-box');
    if (!boxes.length) return null;
    var box = boxes[boxes.length - 1];
    var s = box.querySelector('.shape');
    if (!s || !s.textContent) return null;
    var text = s.textContent.trim().replace(/^(Final Output|Output|Error)\s*:\s*/i, '');
    return { box: box, text: text, error: box.classList.contains('error') };
  }

  function update() {
    var r = latestResult();
    if (!r) { if (bar) bar.classList.remove('show'); return; }
    ensureBar();
    var h = header();
    var topGuard = (h ? h.offsetHeight : 52) + 4;
    var rect = r.box.getBoundingClientRect();
    if (rect.bottom < topGuard) {
      shapeEl.textContent = r.text;
      bar.classList.toggle('is-error', r.error);
      bar.classList.add('show');
    } else {
      bar.classList.remove('show');
    }
  }

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () { update(); ticking = false; });
  }

  function init() {
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    document.addEventListener('click', function () { window.setTimeout(update, 60); }, true);
    update();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
