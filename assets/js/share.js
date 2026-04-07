/**
 * Share Module — Shareable tool outputs via URL encoding
 * Zero dependencies. Pure client-side. No backend needed.
 */
(function() {
  'use strict';

  var MAX_URL_LENGTH = 2000;
  var DEBOUNCE_MS = 150;
  var debounceTimer = null;

  var Share = {
    /** Collect all input values from the tool container */
    collectState: function(container) {
      if (!container) return {};
      var state = {};
      var inputs = container.querySelectorAll('input[type="text"], input[type="number"], textarea, select');
      var count = 0;
      for (var i = 0; i < inputs.length && count < 50; i++) {
        var el = inputs[i];
        var key = el.id || el.name || ('f' + i);
        if (el.value !== '' && el.value !== undefined) {
          state[key] = el.value;
          count++;
        }
      }
      return state;
    },

    /** Generate share URL from state object */
    generate: function(state) {
      if (!state || Object.keys(state).length === 0) return window.location.href;
      var json = JSON.stringify(state);

      // Short states: use query params
      if (json.length < 200) {
        var params = new URLSearchParams();
        var keys = Object.keys(state);
        for (var i = 0; i < keys.length; i++) {
          params.set(keys[i], String(state[keys[i]]));
        }
        var url = window.location.origin + window.location.pathname + '?' + params.toString();
        if (url.length <= MAX_URL_LENGTH) return url;
      }

      // Longer states: base64 hash
      try {
        var b64 = btoa(unescape(encodeURIComponent(json)));
        var url = window.location.origin + window.location.pathname + '#s=' + b64;
        if (url.length <= MAX_URL_LENGTH) return url;
      } catch (e) { /* fall through */ }

      return window.location.href;
    },

    /** Load state from URL (query params or hash) */
    load: function() {
      // Try query params
      var params = new URLSearchParams(window.location.search);
      if (params.toString()) {
        var state = {};
        params.forEach(function(val, key) { state[key] = val; });
        return state;
      }
      // Try base64 hash
      var hash = window.location.hash;
      if (hash && hash.indexOf('#s=') === 0) {
        try {
          var json = decodeURIComponent(escape(atob(hash.slice(3))));
          return JSON.parse(json);
        } catch (e) { return null; }
      }
      return null;
    },

    /** Restore state into the tool's input fields */
    restore: function(container, state) {
      if (!container || !state) return;
      var keys = Object.keys(state);
      for (var i = 0; i < keys.length; i++) {
        var key = keys[i];
        var el = document.getElementById(key);
        if (!el) el = container.querySelector('[name="' + key + '"]');
        if (el) {
          el.value = state[key];
          el.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }
    },

    /** Show the share bar with the generated URL */
    show: function(url) {
      var bar = document.getElementById('share-bar');
      var input = document.getElementById('share-url');
      if (!bar || !input) return;
      input.value = url;
      bar.style.display = 'block';
      history.replaceState(null, '', url);
    },

    /** Copy share URL to clipboard */
    copy: function() {
      var input = document.getElementById('share-url');
      if (!input) return;
      navigator.clipboard.writeText(input.value).then(function() {
        var btn = document.getElementById('copy-share');
        if (btn) {
          btn.textContent = 'Copied!';
          setTimeout(function() { btn.textContent = 'Copy'; }, 2000);
        }
      });
    },

    /** Show "shared result" banner */
    showBanner: function() {
      var main = document.querySelector('main') || document.querySelector('.tool-page') || document.querySelector('.content');
      if (!main) return;
      var banner = document.createElement('div');
      banner.className = 'shared-banner';
      banner.style.cssText = 'background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.3);border-radius:8px;padding:0.75rem 1.25rem;margin-bottom:1.5rem;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;';
      banner.innerHTML = '<span style="font-size:0.9rem;color:#58a6ff;">You\'re viewing a shared result</span><a href="' + window.location.pathname + '" style="color:#8b949e;font-size:0.85rem;text-decoration:none;">Start fresh &rarr;</a>';
      main.prepend(banner);
    },

    /** Social sharing helpers */
    twitter: function(text) {
      var url = document.getElementById('share-url');
      var shareUrl = url ? url.value : window.location.href;
      window.open('https://twitter.com/intent/tweet?text=' + encodeURIComponent(text || document.title) + '&url=' + encodeURIComponent(shareUrl), '_blank', 'width=550,height=420');
    },
    linkedin: function() {
      var url = document.getElementById('share-url');
      var shareUrl = url ? url.value : window.location.href;
      window.open('https://www.linkedin.com/sharing/share-offsite/?url=' + encodeURIComponent(shareUrl), '_blank', 'width=550,height=420');
    },
    reddit: function(title) {
      var url = document.getElementById('share-url');
      var shareUrl = url ? url.value : window.location.href;
      window.open('https://reddit.com/submit?url=' + encodeURIComponent(shareUrl) + '&title=' + encodeURIComponent(title || document.title), '_blank', 'width=550,height=420');
    }
  };

  window.Share = Share;

  /** Initialize share functionality after DOM ready */
  function initShare() {
    var container = document.getElementById('tool-container') || document.querySelector('.tool-container');
    if (!container) return;

    // Inject share bar HTML after tool container
    var shareBar = document.createElement('div');
    shareBar.id = 'share-bar';
    shareBar.style.cssText = 'display:none;background:#161b22;border:1px solid #58a6ff;border-radius:8px;padding:1rem 1.25rem;margin-top:1.5rem;animation:shareSlideUp 0.3s ease;';
    shareBar.innerHTML = '<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">' +
      '<span style="font-size:0.85rem;font-weight:600;color:#c9d1d9;white-space:nowrap;">Share this result</span>' +
      '<div style="flex:1;display:flex;min-width:200px;">' +
        '<input type="text" id="share-url" readonly style="flex:1;background:#0d1117;border:1px solid #30363d;border-right:none;border-radius:6px 0 0 6px;padding:0.5rem 0.75rem;font-family:monospace;font-size:0.75rem;color:#8b949e;">' +
        '<button id="copy-share" onclick="Share.copy()" style="background:#58a6ff;color:#000;border:none;border-radius:0 6px 6px 0;padding:0.5rem 1rem;font-weight:600;font-size:0.8rem;cursor:pointer;">Copy</button>' +
      '</div>' +
      '<div style="display:flex;gap:0.5rem;">' +
        '<button onclick="Share.twitter()" title="Share on X" style="width:36px;height:36px;border-radius:50%;border:1px solid #30363d;background:transparent;color:#8b949e;cursor:pointer;font-size:0.85rem;">&#x1D54F;</button>' +
        '<button onclick="Share.linkedin()" title="Share on LinkedIn" style="width:36px;height:36px;border-radius:50%;border:1px solid #30363d;background:transparent;color:#8b949e;cursor:pointer;font-size:0.85rem;">in</button>' +
        '<button onclick="Share.reddit()" title="Share on Reddit" style="width:36px;height:36px;border-radius:50%;border:1px solid #30363d;background:transparent;color:#8b949e;cursor:pointer;font-size:0.85rem;">&uarr;</button>' +
      '</div>' +
    '</div>';

    // Insert after the tool container's parent section or after the container
    var parent = container.closest('.tool-section') || container.parentNode;
    if (parent) {
      parent.insertBefore(shareBar, container.nextSibling);
    }

    // Add slide-up animation
    var style = document.createElement('style');
    style.textContent = '@keyframes shareSlideUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}';
    document.head.appendChild(style);

    // Hook into button clicks (Calculate, Convert, Encode, etc.)
    container.addEventListener('click', function(e) {
      var btn = e.target.closest('.btn');
      if (!btn) return;
      var text = btn.textContent.toLowerCase().trim();
      // Skip clear, copy, and outline-only buttons
      if (text === 'clear' || text === 'copy' || text === 'copied!') return;
      if (btn.dataset.copy) return; // Skip copy buttons

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function() {
        var state = Share.collectState(container);
        if (Object.keys(state).length > 0) {
          var url = Share.generate(state);
          Share.show(url);
        }
      }, DEBOUNCE_MS);
    });

    // Also trigger on Enter key in inputs
    container.addEventListener('keydown', function(e) {
      if (e.key !== 'Enter') return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function() {
        var state = Share.collectState(container);
        if (Object.keys(state).length > 0) {
          var url = Share.generate(state);
          Share.show(url);
        }
      }, DEBOUNCE_MS);
    });

    // Check for shared state in URL
    var sharedState = Share.load();
    if (sharedState) {
      // Wait for component to initialize, then restore state
      var attempts = 0;
      var maxAttempts = 20;
      var restoreInterval = setInterval(function() {
        attempts++;
        var inputs = container.querySelectorAll('input, textarea, select');
        if (inputs.length > 0 || attempts >= maxAttempts) {
          clearInterval(restoreInterval);
          if (inputs.length > 0) {
            Share.restore(container, sharedState);
            // Find and click the primary action button
            var primaryBtn = container.querySelector('.btn:not(.btn-outline)');
            if (primaryBtn) {
              setTimeout(function() { primaryBtn.click(); }, 50);
            }
            Share.showBanner();
          }
        }
      }, 100);
    }
  }

  // Wait for DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      // Delay slightly to let app.js load components first
      setTimeout(initShare, 300);
    });
  } else {
    setTimeout(initShare, 300);
  }
})();
