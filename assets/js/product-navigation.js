// Shared accessible menu for pages with and without the calculator scripts.
(() => {
  const toggle = document.querySelector('header .mobile-toggle');
  const nav = document.querySelector('header nav');
  if (!toggle || !nav) return;
  nav.id = nav.id || 'site-main-menu';
  toggle.setAttribute('aria-controls', nav.id);
  toggle.setAttribute('aria-expanded', String(nav.classList.contains('open')));
  function setOpen(open) {
    nav.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  }
  // Capture this one control so legacy per-page handlers cannot toggle it twice.
  document.addEventListener('click', event => {
    if (!toggle.contains(event.target)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    setOpen(!nav.classList.contains('open'));
  }, true);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && nav.classList.contains('open')) {
      setOpen(false);
      toggle.focus();
    }
  });
})();
