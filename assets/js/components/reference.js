/**
 * Reference Component
 * Interactive references for activation functions (with Canvas plots),
 * loss functions, optimizers, and einsum calculator.
 */
(function () {
  'use strict';

  window.HeyTensor = window.HeyTensor || {};
  window.HeyTensor.components = window.HeyTensor.components || {};

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ── Activation Functions ── */
  var ACTIVATIONS = [
    {
      name: 'ReLU',
      formula: 'f(x) = max(0, x)',
      fn: function (x) { return Math.max(0, x); },
      color: '#3B82F6',
      useCase: 'Default for CNNs and hidden layers',
      pros: 'Fast, sparse activation, no vanishing gradient for positive inputs',
      cons: 'Dying ReLU problem (neurons stuck at 0)'
    },
    {
      name: 'LeakyReLU',
      formula: 'f(x) = x if x > 0, else 0.01*x',
      fn: function (x) { return x > 0 ? x : 0.01 * x; },
      color: '#22C55E',
      useCase: 'Alternative to ReLU when dying neurons are a problem',
      pros: 'No dying neuron problem, fast',
      cons: 'Small negative slope is a hyperparameter'
    },
    {
      name: 'GELU',
      formula: 'f(x) = x * \u03A6(x)',
      fn: function (x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))); },
      color: '#A855F7',
      useCase: 'Transformers (BERT, GPT), NLP models',
      pros: 'Smooth, non-monotonic, works well with attention',
      cons: 'Slightly slower to compute than ReLU'
    },
    {
      name: 'SiLU / Swish',
      formula: 'f(x) = x * \u03C3(x)',
      fn: function (x) { return x / (1 + Math.exp(-x)); },
      color: '#F97316',
      useCase: 'EfficientNet, modern architectures',
      pros: 'Smooth, self-gated, outperforms ReLU in many tasks',
      cons: 'Slightly slower than ReLU, unbounded'
    },
    {
      name: 'Sigmoid',
      formula: 'f(x) = 1 / (1 + e^(-x))',
      fn: function (x) { return 1 / (1 + Math.exp(-x)); },
      color: '#EAB308',
      useCase: 'Binary classification output, gates (LSTM/GRU)',
      pros: 'Output bounded (0,1), interpretable as probability',
      cons: 'Vanishing gradient for large |x|, not zero-centered'
    },
    {
      name: 'Tanh',
      formula: 'f(x) = (e^x - e^(-x)) / (e^x + e^(-x))',
      fn: function (x) { return Math.tanh(x); },
      color: '#EF4444',
      useCase: 'RNN hidden states, outputs in [-1, 1]',
      pros: 'Zero-centered, stronger gradients than sigmoid',
      cons: 'Vanishing gradient for large |x|'
    },
    {
      name: 'Softmax',
      formula: 'f(x_i) = e^(x_i) / \u03A3 e^(x_j)',
      fn: function (x) { return 1 / (1 + Math.exp(-x)); }, // approximation for single-value plot
      color: '#EC4899',
      useCase: 'Multi-class classification output layer',
      pros: 'Outputs sum to 1, interpretable as probabilities',
      cons: 'Applied per-vector (not element-wise), sensitive to outliers'
    },
    {
      name: 'ELU',
      formula: 'f(x) = x if x > 0, else \u03B1*(e^x - 1)',
      fn: function (x) { return x > 0 ? x : 1.0 * (Math.exp(x) - 1); },
      color: '#06B6D4',
      useCase: 'Alternative to ReLU with smoother negative region',
      pros: 'No dying neurons, smoother around 0, closer to zero-centered',
      cons: 'Exponential computation for negative values'
    },
    {
      name: 'SELU',
      formula: 'f(x) = \u03BB * (x if x > 0, else \u03B1*(e^x - 1))',
      fn: function (x) {
        var alpha = 1.6732632423543772;
        var scale = 1.0507009873554805;
        return scale * (x > 0 ? x : alpha * (Math.exp(x) - 1));
      },
      color: '#8B5CF6',
      useCase: 'Self-normalizing networks (with lecun_normal init)',
      pros: 'Self-normalizing (pushes outputs toward 0 mean, unit variance)',
      cons: 'Requires specific initialization and architecture constraints'
    }
  ];

  /* ── Loss Functions ── */
  var LOSSES = [
    { name: 'CrossEntropyLoss', formula: '-\u03A3 y_i * log(softmax(x_i))', when: 'Multi-class classification', code: 'loss = nn.CrossEntropyLoss()\nloss(logits, targets)  # targets: class indices', note: 'Includes Softmax — do NOT apply Softmax before this loss' },
    { name: 'MSELoss', formula: 'mean((y - y\u0302)^2)', when: 'Regression', code: 'loss = nn.MSELoss()\nloss(predictions, targets)', note: 'Penalizes large errors heavily (quadratic)' },
    { name: 'L1Loss', formula: 'mean(|y - y\u0302|)', when: 'Regression (robust to outliers)', code: 'loss = nn.L1Loss()\nloss(predictions, targets)', note: 'More robust to outliers than MSE' },
    { name: 'BCEWithLogitsLoss', formula: '-[y*log(\u03C3(x)) + (1-y)*log(1-\u03C3(x))]', when: 'Binary classification, multi-label', code: 'loss = nn.BCEWithLogitsLoss()\nloss(logits, targets)  # targets: 0.0 or 1.0', note: 'Includes Sigmoid — do NOT apply Sigmoid before this' },
    { name: 'HuberLoss', formula: 'L2 if |error|<\u03B4, else L1', when: 'Regression (smooth L1)', code: 'loss = nn.HuberLoss(delta=1.0)\nloss(predictions, targets)', note: 'Combines MSE (small errors) and L1 (large errors)' },
    { name: 'NLLLoss', formula: '-log(p_target)', when: 'After LogSoftmax', code: 'loss = nn.NLLLoss()\nloss(log_softmax_output, targets)', note: 'Requires LogSoftmax input; CrossEntropyLoss = LogSoftmax + NLLLoss' },
    { name: 'KLDivLoss', formula: 'y * log(y/x)', when: 'Distribution matching, VAEs, knowledge distillation', code: 'loss = nn.KLDivLoss(reduction="batchmean")\nloss(log_probs, target_probs)', note: 'Input must be log-probabilities; target must be probabilities' }
  ];

  /* ── Optimizers ── */
  var OPTIMIZERS = [
    { name: 'SGD', desc: 'Stochastic Gradient Descent with optional momentum and weight decay.', params: 'lr (required), momentum=0, weight_decay=0, nesterov=False', when: 'CNNs, final fine-tuning, when best accuracy is needed. Often outperforms Adam on image tasks with proper scheduling.', memory: '1x params (momentum buffer)' },
    { name: 'Adam', desc: 'Adaptive Moment Estimation. Combines momentum (first moment) with RMSprop (second moment).', params: 'lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0', when: 'Default for most tasks. Quick convergence. Good for prototyping.', memory: '2x params (m + v states)' },
    { name: 'AdamW', desc: 'Adam with decoupled weight decay. Fixes the weight decay/L2 regularization issue in Adam.', params: 'lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01', when: 'Transformers, NLP, any task where you use weight decay. Default for fine-tuning LLMs.', memory: '2x params (m + v states)' },
    { name: 'RMSprop', desc: 'Root Mean Square Propagation. Adapts learning rates using a moving average of squared gradients.', params: 'lr=0.01, alpha=0.99, eps=1e-8, momentum=0', when: 'RNNs and non-stationary problems. Also common in reinforcement learning.', memory: '1x-2x params' },
    { name: 'Adagrad', desc: 'Adaptive Gradient. Accumulates all past squared gradients to scale learning rates.', params: 'lr=0.01, lr_decay=0, eps=1e-10', when: 'Sparse features (NLP, recommendation systems). Learning rate decreases over time automatically.', memory: '1x params' },
    { name: 'LBFGS', desc: 'Limited-memory BFGS, a quasi-Newton method. Uses line search for step size.', params: 'lr=1, max_iter=20, history_size=100', when: 'Small models, physics-informed neural networks, full-batch optimization. Not for mini-batch training.', memory: 'O(history_size * params)' }
  ];

  /* ── Einsum common expressions ── */
  var EINSUM_EXAMPLES = [
    { expr: 'ij,jk->ik', desc: 'Matrix multiplication (A @ B)', shapes: '(M,N), (N,K) -> (M,K)' },
    { expr: 'ij,ij->', desc: 'Frobenius inner product (sum of element-wise product)', shapes: '(M,N), (M,N) -> scalar' },
    { expr: 'ij->ji', desc: 'Matrix transpose', shapes: '(M,N) -> (N,M)' },
    { expr: 'ii->', desc: 'Matrix trace (sum of diagonal)', shapes: '(N,N) -> scalar' },
    { expr: 'ij->i', desc: 'Row-wise sum', shapes: '(M,N) -> (M,)' },
    { expr: 'i,j->ij', desc: 'Outer product', shapes: '(M,), (N,) -> (M,N)' },
    { expr: 'bik,bkj->bij', desc: 'Batched matrix multiplication', shapes: '(B,I,K), (B,K,J) -> (B,I,J)' },
    { expr: 'bhqd,bhkd->bhqk', desc: 'Attention scores (Q @ K^T)', shapes: '(B,H,Q,D), (B,H,K,D) -> (B,H,Q,K)' },
    { expr: 'bhqk,bhkd->bhqd', desc: 'Attention output (scores @ V)', shapes: '(B,H,Q,K), (B,H,K,D) -> (B,H,Q,D)' }
  ];

  /* ── Draw activation function on canvas ── */
  function drawActivation(canvas, fn, color, name) {
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    var bg = '#0a0a0a';
    var gridColor = '#1a1a1a';
    var axisColor = '#2a2a2a';
    var textColor = '#666';

    // Background
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Coordinate system: x in [-5, 5], y in [-2, 3]
    var xMin = -5, xMax = 5, yMin = -2, yMax = 3;
    function toCanvasX(x) { return (x - xMin) / (xMax - xMin) * w; }
    function toCanvasY(y) { return h - (y - yMin) / (yMax - yMin) * h; }

    // Grid lines
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 0.5;
    for (var gx = Math.ceil(xMin); gx <= xMax; gx++) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(gx), 0);
      ctx.lineTo(toCanvasX(gx), h);
      ctx.stroke();
    }
    for (var gy = Math.ceil(yMin); gy <= yMax; gy++) {
      ctx.beginPath();
      ctx.moveTo(0, toCanvasY(gy));
      ctx.lineTo(w, toCanvasY(gy));
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    // X axis (y=0)
    ctx.beginPath();
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(w, toCanvasY(0));
    ctx.stroke();
    // Y axis (x=0)
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), h);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = textColor;
    ctx.font = '10px "IBM Plex Sans", sans-serif';
    ctx.textAlign = 'center';
    for (var lx = Math.ceil(xMin); lx <= xMax; lx++) {
      if (lx === 0) continue;
      ctx.fillText(lx, toCanvasX(lx), toCanvasY(0) + 12);
    }
    ctx.textAlign = 'right';
    for (var ly = Math.ceil(yMin); ly <= yMax; ly++) {
      if (ly === 0) continue;
      ctx.fillText(ly, toCanvasX(0) - 4, toCanvasY(ly) + 4);
    }

    // Draw function curve
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    var steps = w * 2;
    for (var i = 0; i <= steps; i++) {
      var x = xMin + (xMax - xMin) * (i / steps);
      var y = fn(x);
      // Clamp y to visible range
      y = Math.max(yMin - 1, Math.min(yMax + 1, y));
      var cx = toCanvasX(x);
      var cy = toCanvasY(y);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Function name label
    ctx.fillStyle = color;
    ctx.font = 'bold 12px "Space Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(name, 8, 16);
  }

  /* ── Build Activation Functions Topic ── */
  function buildActivationFunctions(container) {
    var card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<h3>Activation Function Curves</h3><p class="plot-info">Click on any function below to highlight it. Each curve is plotted on the range x \u2208 [-5, 5].</p>';
    container.appendChild(card);

    var plotGrid = document.createElement('div');
    plotGrid.className = 'plot-grid';
    container.appendChild(plotGrid);

    ACTIVATIONS.forEach(function (act) {
      var plotCard = document.createElement('div');
      plotCard.className = 'plot-card';
      plotCard.innerHTML = '<h3>' + escHtml(act.name) + '</h3>';

      var canvas = document.createElement('canvas');
      canvas.style.cssText = 'width:100%;height:160px;border-radius:4px;';
      plotCard.appendChild(canvas);

      plotCard.innerHTML += '<div class="plot-formula"><code>' + escHtml(act.formula) + '</code></div>' +
        '<div class="plot-info"><strong>Use:</strong> ' + escHtml(act.useCase) + '<br>' +
        '<span style="color:var(--green)">+</span> ' + escHtml(act.pros) + '<br>' +
        '<span style="color:var(--red)">-</span> ' + escHtml(act.cons) + '</div>';
      plotGrid.appendChild(plotCard);

      // Draw after append so clientWidth/Height are available
      requestAnimationFrame(function () {
        drawActivation(canvas, act.fn, act.color, act.name);
      });
    });

    // Comparison table
    var tableCard = document.createElement('div');
    tableCard.className = 'card';
    tableCard.innerHTML = '<h3>Comparison Table</h3>';
    var tableWrap = document.createElement('div');
    tableWrap.className = 'ref-table-wrap';
    var html = '<table class="ref-table"><tr><th>Function</th><th>Formula</th><th>Range</th><th>Best For</th></tr>';
    var ranges = {
      'ReLU': '[0, \u221E)', 'LeakyReLU': '(-\u221E, \u221E)', 'GELU': '(-0.17, \u221E)',
      'SiLU / Swish': '(-0.28, \u221E)', 'Sigmoid': '(0, 1)', 'Tanh': '(-1, 1)',
      'Softmax': '(0, 1)', 'ELU': '(-\u03B1, \u221E)', 'SELU': '(-\u03BB\u03B1, \u221E)'
    };
    ACTIVATIONS.forEach(function (act) {
      html += '<tr><td><strong>' + escHtml(act.name) + '</strong></td><td><code>' + escHtml(act.formula) + '</code></td>' +
        '<td>' + (ranges[act.name] || '') + '</td><td>' + escHtml(act.useCase) + '</td></tr>';
    });
    html += '</table>';
    tableWrap.innerHTML = html;
    tableCard.appendChild(tableWrap);
    container.appendChild(tableCard);
  }

  /* ── Build Loss Functions Topic ── */
  function buildLossFunctions(container) {
    var card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<h3>PyTorch Loss Functions Reference</h3><p class="plot-info">Which loss function to use depends on your task. Classification uses cross-entropy variants; regression uses MSE or L1.</p>';
    container.appendChild(card);

    LOSSES.forEach(function (loss) {
      var lossCard = document.createElement('div');
      lossCard.className = 'card';
      lossCard.innerHTML =
        '<h3>' + escHtml(loss.name) + '</h3>' +
        '<div class="params-grid" style="grid-template-columns:1fr 1fr;margin-bottom:12px;">' +
        '<div class="input-field"><label>Formula</label><div style="font-family:var(--font-display);font-size:0.85rem;color:var(--accent);padding:8px 0;">' + escHtml(loss.formula) + '</div></div>' +
        '<div class="input-field"><label>When to Use</label><div style="font-size:0.9rem;color:var(--text);padding:8px 0;">' + escHtml(loss.when) + '</div></div>' +
        '</div>' +
        '<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;font-family:var(--font-display);font-size:0.8rem;overflow-x:auto;margin-bottom:8px;">' + escHtml(loss.code) + '</pre>' +
        '<p class="plot-info" style="margin:0;"><strong>Note:</strong> ' + escHtml(loss.note) + '</p>';
      container.appendChild(lossCard);
    });
  }

  /* ── Build Optimizers Topic ── */
  function buildOptimizers(container) {
    var card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<h3>PyTorch Optimizers Comparison</h3><p class="plot-info">Choose an optimizer based on your task, model size, and convergence requirements.</p>';
    container.appendChild(card);

    var tableCard = document.createElement('div');
    tableCard.className = 'card';
    var tableWrap = document.createElement('div');
    tableWrap.className = 'ref-table-wrap';
    var html = '<table class="ref-table"><tr><th>Optimizer</th><th>Description</th><th>Key Parameters</th><th>Memory</th><th>Best For</th></tr>';
    OPTIMIZERS.forEach(function (opt) {
      html += '<tr><td><strong>' + escHtml(opt.name) + '</strong></td><td>' + escHtml(opt.desc) + '</td>' +
        '<td><code>' + escHtml(opt.params) + '</code></td><td>' + escHtml(opt.memory) + '</td><td>' + escHtml(opt.when) + '</td></tr>';
    });
    html += '</table>';
    tableWrap.innerHTML = html;
    tableCard.appendChild(tableWrap);
    container.appendChild(tableCard);

    // Individual detail cards
    OPTIMIZERS.forEach(function (opt) {
      var detailCard = document.createElement('div');
      detailCard.className = 'card';
      detailCard.innerHTML =
        '<h3>' + escHtml(opt.name) + '</h3>' +
        '<p style="margin-bottom:12px;">' + escHtml(opt.desc) + '</p>' +
        '<div class="params-grid" style="grid-template-columns:1fr 1fr;">' +
        '<div class="input-field"><label>Key Parameters</label><div style="font-family:var(--font-display);font-size:0.8rem;padding:8px 0;"><code>' + escHtml(opt.params) + '</code></div></div>' +
        '<div class="input-field"><label>Memory Overhead</label><div style="font-size:0.9rem;padding:8px 0;">' + escHtml(opt.memory) + '</div></div>' +
        '</div>' +
        '<p class="plot-info"><strong>When to use:</strong> ' + escHtml(opt.when) + '</p>';
      container.appendChild(detailCard);
    });
  }

  /* ── Build Einsum Calculator Topic ── */
  function buildEinsum(container) {
    var card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<h3>Einsum Expression Calculator</h3><p class="plot-info">Enter an einsum expression and tensor shapes to see the output shape. Supports NumPy and PyTorch notation.</p>';

    var inputWrap = document.createElement('div');
    inputWrap.className = 'einsum-input-wrap';

    var exprField = document.createElement('div');
    exprField.className = 'input-field';
    exprField.innerHTML = '<label>Einsum Expression</label>';
    var exprInput = document.createElement('input');
    exprInput.type = 'text';
    exprInput.value = 'ij,jk->ik';
    exprInput.placeholder = 'e.g., ij,jk->ik';
    exprField.appendChild(exprInput);
    inputWrap.appendChild(exprField);

    var shapesField = document.createElement('div');
    shapesField.className = 'input-field';
    shapesField.innerHTML = '<label>Input Shapes (semicolon-separated)</label>';
    var shapesInput = document.createElement('input');
    shapesInput.type = 'text';
    shapesInput.value = '3,4;4,5';
    shapesInput.placeholder = 'e.g., 3,4;4,5';
    shapesField.appendChild(shapesInput);
    inputWrap.appendChild(shapesField);

    card.appendChild(inputWrap);

    var calcBtn = document.createElement('button');
    calcBtn.className = 'btn btn-primary';
    calcBtn.textContent = 'Calculate Output Shape';
    card.appendChild(calcBtn);

    var resultDiv = document.createElement('div');
    resultDiv.className = 'einsum-result';
    resultDiv.style.display = 'none';
    card.appendChild(resultDiv);

    container.appendChild(card);

    // Examples table
    var exCard = document.createElement('div');
    exCard.className = 'card';
    exCard.innerHTML = '<h3>Common Einsum Expressions</h3>';
    var tableWrap = document.createElement('div');
    tableWrap.className = 'ref-table-wrap';
    var html = '<table class="ref-table"><tr><th>Expression</th><th>Operation</th><th>Shapes</th></tr>';
    EINSUM_EXAMPLES.forEach(function (ex) {
      html += '<tr><td><code>' + escHtml(ex.expr) + '</code></td><td>' + escHtml(ex.desc) + '</td><td>' + escHtml(ex.shapes) + '</td></tr>';
    });
    html += '</table>';
    tableWrap.innerHTML = html;
    exCard.appendChild(tableWrap);
    container.appendChild(exCard);

    // Einsum calculation logic
    function calculateEinsum() {
      var expr = exprInput.value.trim();
      var shapesStr = shapesInput.value.trim();

      if (!expr || !shapesStr) {
        resultDiv.style.display = 'none';
        return;
      }

      try {
        var parts = expr.split('->');
        if (parts.length !== 2) throw new Error('Expression must contain exactly one "->"');

        var inputParts = parts[0].split(',');
        var outputSubscripts = parts[1].trim();

        var shapeArrays = shapesStr.split(';').map(function (s) {
          return s.trim().split(',').map(function (n) { return parseInt(n.trim(), 10); });
        });

        if (inputParts.length !== shapeArrays.length) {
          throw new Error('Number of input subscript groups (' + inputParts.length + ') does not match number of shapes (' + shapeArrays.length + ')');
        }

        // Build dimension map
        var dimMap = {};
        for (var i = 0; i < inputParts.length; i++) {
          var subs = inputParts[i].trim();
          var shape = shapeArrays[i];
          if (subs.length !== shape.length) {
            throw new Error('Subscript "' + subs + '" has ' + subs.length + ' dims but shape has ' + shape.length + ' dims');
          }
          for (var j = 0; j < subs.length; j++) {
            var c = subs[j];
            if (dimMap[c] !== undefined && dimMap[c] !== shape[j]) {
              throw new Error('Dimension "' + c + '" has conflicting sizes: ' + dimMap[c] + ' vs ' + shape[j]);
            }
            dimMap[c] = shape[j];
          }
        }

        // Build output shape
        var outputShape = [];
        for (var k = 0; k < outputSubscripts.length; k++) {
          var oc = outputSubscripts[k];
          if (dimMap[oc] === undefined) {
            throw new Error('Output subscript "' + oc + '" not found in any input');
          }
          outputShape.push(dimMap[oc]);
        }

        // Identify contracted dims
        var allInputChars = inputParts.join('').replace(/\s/g, '');
        var contracted = [];
        var seen = {};
        for (var m = 0; m < allInputChars.length; m++) {
          seen[allInputChars[m]] = true;
        }
        Object.keys(seen).forEach(function (ch) {
          if (outputSubscripts.indexOf(ch) === -1) {
            contracted.push(ch + '(size ' + dimMap[ch] + ')');
          }
        });

        resultDiv.style.display = 'block';
        resultDiv.innerHTML =
          '<div class="shape" style="color:var(--accent);font-family:var(--font-display);font-size:1.1rem;font-weight:700;margin-bottom:8px;">Output Shape: [' + outputShape.join(', ') + ']</div>' +
          '<div class="formula" style="color:var(--text-dim);font-size:0.85rem;">' +
          'Dimension mapping: ' + Object.keys(dimMap).map(function (k) { return k + '=' + dimMap[k]; }).join(', ') +
          (contracted.length > 0 ? '<br>Contracted (summed) dimensions: ' + contracted.join(', ') : '<br>No contracted dimensions (element-wise operation)') +
          '<br>PyTorch: <code>torch.einsum("' + escHtml(expr) + '", A, B)</code>' +
          '<br>NumPy: <code>np.einsum("' + escHtml(expr) + '", A, B)</code>' +
          '</div>';
      } catch (e) {
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<div class="shape" style="color:var(--red);">Error</div><div class="formula" style="color:var(--text-dim);">' + escHtml(e.message) + '</div>';
      }
    }

    calcBtn.addEventListener('click', calculateEinsum);
    exprInput.addEventListener('keydown', function (e) { if (e.key === 'Enter') calculateEinsum(); });
    shapesInput.addEventListener('keydown', function (e) { if (e.key === 'Enter') calculateEinsum(); });

    // Auto-calculate
    calculateEinsum();
  }

  /* ── Init ── */
  window.HeyTensor.components['reference'] = {
    init: function (container, config) {
      var topic = config.topic || 'activation-functions';
      switch (topic) {
        case 'activation-functions': buildActivationFunctions(container); break;
        case 'loss-functions': buildLossFunctions(container); break;
        case 'optimizers': buildOptimizers(container); break;
        case 'einsum': buildEinsum(container); break;
        default:
          container.innerHTML = '<p style="color:var(--red);">Unknown reference topic: ' + escHtml(topic) + '</p>';
      }
    }
  };
})();
