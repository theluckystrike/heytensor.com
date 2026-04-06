/**
 * HeyTensor Tool Page — Component Loader
 * Reads data-component and data-config from #tool-container,
 * dynamically imports the matching JS component, and calls its init().
 */
(function () {
  'use strict';

  // Mobile nav toggle (shared with main site)
  const toggle = document.querySelector('.mobile-toggle');
  const nav = document.querySelector('nav');
  if (toggle && nav) {
    toggle.addEventListener('click', () => nav.classList.toggle('open'));
  }

  // Tool container
  const container = document.getElementById('tool-container');
  if (!container) return; // Not a tool page

  const componentName = container.dataset.component;
  let config = {};
  try {
    config = JSON.parse(container.dataset.config || '{}');
  } catch (e) {
    console.error('Invalid data-config JSON:', e);
  }

  // Map component names to script paths
  const componentMap = {
    'shape-calculator': '/assets/js/components/shape-calculator.js',
    'error-debugger': '/assets/js/components/error-debugger.js',
    'reference': '/assets/js/components/reference.js',
    'model-tool': '/assets/js/components/model-tool.js'
  };

  const scriptPath = componentMap[componentName];
  if (!scriptPath) {
    container.innerHTML = '<p style="color:var(--red);">Unknown component: ' + componentName + '</p>';
    return;
  }

  // Load component script
  const script = document.createElement('script');
  script.src = scriptPath;
  script.onload = function () {
    // Each component registers itself on window.HeyTensor.components
    if (window.HeyTensor && window.HeyTensor.components && window.HeyTensor.components[componentName]) {
      window.HeyTensor.components[componentName].init(container, config);
    } else {
      console.error('Component did not register:', componentName);
    }
  };
  script.onerror = function () {
    container.innerHTML = '<p style="color:var(--red);">Failed to load component: ' + componentName + '</p>';
  };
  document.head.appendChild(script);
})();
