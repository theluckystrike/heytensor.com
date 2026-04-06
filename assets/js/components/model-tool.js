/**
 * Model Tool Component
 * Parameter counter and GPU memory calculator for neural networks.
 */
(function () {
  'use strict';

  window.HeyTensor = window.HeyTensor || {};
  window.HeyTensor.components = window.HeyTensor.components || {};

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function formatNum(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toLocaleString();
  }

  function formatBytes(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
    return bytes + ' B';
  }

  /* ── Layer parameter formulas ── */
  var LAYER_TYPES = [
    {
      type: 'Linear',
      fields: [
        { key: 'in_f', label: 'in_features', default: 512 },
        { key: 'out_f', label: 'out_features', default: 256 },
        { key: 'bias', label: 'bias (0/1)', default: 1 }
      ],
      calcParams: function (f) {
        return f.in_f * f.out_f + (f.bias ? f.out_f : 0);
      },
      desc: function (f) {
        return 'Linear(' + f.in_f + ', ' + f.out_f + ')';
      }
    },
    {
      type: 'Conv2d',
      fields: [
        { key: 'in_ch', label: 'in_channels', default: 3 },
        { key: 'out_ch', label: 'out_channels', default: 64 },
        { key: 'k', label: 'kernel_size', default: 3 },
        { key: 'bias', label: 'bias (0/1)', default: 1 },
        { key: 'groups', label: 'groups', default: 1 }
      ],
      calcParams: function (f) {
        return f.out_ch * (f.in_ch / f.groups) * f.k * f.k + (f.bias ? f.out_ch : 0);
      },
      desc: function (f) {
        return 'Conv2d(' + f.in_ch + ', ' + f.out_ch + ', ' + f.k + ')';
      }
    },
    {
      type: 'Conv1d',
      fields: [
        { key: 'in_ch', label: 'in_channels', default: 1 },
        { key: 'out_ch', label: 'out_channels', default: 32 },
        { key: 'k', label: 'kernel_size', default: 5 },
        { key: 'bias', label: 'bias (0/1)', default: 1 },
        { key: 'groups', label: 'groups', default: 1 }
      ],
      calcParams: function (f) {
        return f.out_ch * (f.in_ch / f.groups) * f.k + (f.bias ? f.out_ch : 0);
      },
      desc: function (f) {
        return 'Conv1d(' + f.in_ch + ', ' + f.out_ch + ', ' + f.k + ')';
      }
    },
    {
      type: 'LSTM',
      fields: [
        { key: 'input_size', label: 'input_size', default: 128 },
        { key: 'hidden_size', label: 'hidden_size', default: 256 },
        { key: 'num_layers', label: 'num_layers', default: 1 },
        { key: 'bidir', label: 'bidirectional (0/1)', default: 0 },
        { key: 'bias', label: 'bias (0/1)', default: 1 }
      ],
      calcParams: function (f) {
        var numDir = f.bidir ? 2 : 1;
        var total = 0;
        for (var l = 0; l < f.num_layers; l++) {
          var inSize = l === 0 ? f.input_size : f.hidden_size * numDir;
          // 4 gates: input, forget, cell, output
          var gateParams = 4 * (inSize * f.hidden_size + f.hidden_size * f.hidden_size);
          if (f.bias) gateParams += 4 * f.hidden_size * 2; // two bias vectors per gate
          total += gateParams * numDir;
        }
        return total;
      },
      desc: function (f) {
        return 'LSTM(' + f.input_size + ', ' + f.hidden_size + ', layers=' + f.num_layers + (f.bidir ? ', bidir' : '') + ')';
      }
    },
    {
      type: 'Embedding',
      fields: [
        { key: 'num_emb', label: 'num_embeddings', default: 30000 },
        { key: 'emb_dim', label: 'embedding_dim', default: 768 }
      ],
      calcParams: function (f) {
        return f.num_emb * f.emb_dim;
      },
      desc: function (f) {
        return 'Embedding(' + f.num_emb + ', ' + f.emb_dim + ')';
      }
    },
    {
      type: 'MultiheadAttention',
      fields: [
        { key: 'embed_dim', label: 'embed_dim', default: 512 },
        { key: 'num_heads', label: 'num_heads', default: 8 },
        { key: 'bias', label: 'bias (0/1)', default: 1 }
      ],
      calcParams: function (f) {
        // Q, K, V projections + output projection
        var proj = 3 * f.embed_dim * f.embed_dim + f.embed_dim * f.embed_dim;
        if (f.bias) proj += 3 * f.embed_dim + f.embed_dim;
        return proj;
      },
      desc: function (f) {
        return 'MultiheadAttention(embed=' + f.embed_dim + ', heads=' + f.num_heads + ')';
      }
    },
    {
      type: 'BatchNorm2d',
      fields: [
        { key: 'num_features', label: 'num_features', default: 64 }
      ],
      calcParams: function (f) {
        return f.num_features * 2; // gamma + beta (running mean/var are buffers, not parameters)
      },
      desc: function (f) {
        return 'BatchNorm2d(' + f.num_features + ')';
      }
    },
    {
      type: 'LayerNorm',
      fields: [
        { key: 'normalized_shape', label: 'normalized_shape', default: 768 }
      ],
      calcParams: function (f) {
        return f.normalized_shape * 2; // gamma + beta
      },
      desc: function (f) {
        return 'LayerNorm(' + f.normalized_shape + ')';
      }
    }
  ];

  /* ── Build Parameter Counter ── */
  function buildParamCounter(container) {
    var card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<h3>Add Layers to Count Parameters</h3><p class="plot-info">Add layers to your model definition. The parameter count is calculated using the exact same formulas PyTorch uses internally.</p>';
    container.appendChild(card);

    var layersList = document.createElement('div');
    layersList.className = 'model-layers-list';
    container.appendChild(layersList);

    // Add layer controls
    var controls = document.createElement('div');
    controls.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;';
    var addSelect = document.createElement('select');
    LAYER_TYPES.forEach(function (lt) {
      var opt = document.createElement('option');
      opt.value = lt.type;
      opt.textContent = lt.type;
      addSelect.appendChild(opt);
    });
    addSelect.style.maxWidth = '200px';
    var addBtn = document.createElement('button');
    addBtn.className = 'btn btn-primary btn-sm';
    addBtn.textContent = '+ Add Layer';
    var clearBtn = document.createElement('button');
    clearBtn.className = 'btn btn-secondary btn-sm';
    clearBtn.textContent = 'Clear All';
    controls.appendChild(addSelect);
    controls.appendChild(addBtn);
    controls.appendChild(clearBtn);
    container.appendChild(controls);

    // Summary
    var summaryDiv = document.createElement('div');
    summaryDiv.className = 'model-summary';
    summaryDiv.style.display = 'none';
    container.appendChild(summaryDiv);

    var layers = [];

    function addLayer(type) {
      var layerDef = LAYER_TYPES.find(function (lt) { return lt.type === type; });
      if (!layerDef) return;
      var fields = {};
      layerDef.fields.forEach(function (f) { fields[f.key] = f.default; });
      layers.push({ typeName: type, def: layerDef, fields: fields });
      renderLayers();
    }

    function renderLayers() {
      layersList.innerHTML = '';
      var totalParams = 0;
      var breakdown = [];

      layers.forEach(function (layer, idx) {
        var row = document.createElement('div');
        row.className = 'model-layer-row';

        var numSpan = document.createElement('span');
        numSpan.className = 'step-num';
        numSpan.textContent = (idx + 1);
        numSpan.style.cssText = 'font-family:var(--font-display);font-size:0.7rem;color:var(--text-dim);min-width:20px;';
        row.appendChild(numSpan);

        var nameSpan = document.createElement('span');
        nameSpan.style.cssText = 'font-weight:500;min-width:80px;';
        nameSpan.textContent = layer.typeName;
        row.appendChild(nameSpan);

        // Field inputs
        layer.def.fields.forEach(function (f) {
          var inp = document.createElement('input');
          inp.type = 'number';
          inp.value = layer.fields[f.key];
          inp.title = f.label;
          inp.placeholder = f.label;
          inp.style.maxWidth = '90px';
          inp.addEventListener('change', function () {
            layer.fields[f.key] = parseInt(this.value, 10) || 0;
            renderLayers();
          });
          row.appendChild(inp);
        });

        var paramCount = layer.def.calcParams(layer.fields);
        totalParams += paramCount;
        breakdown.push({ name: layer.def.desc(layer.fields), params: paramCount });

        var paramSpan = document.createElement('span');
        paramSpan.className = 'layer-params';
        paramSpan.textContent = formatNum(paramCount);
        row.appendChild(paramSpan);

        var removeBtn = document.createElement('button');
        removeBtn.className = 'remove-layer-btn';
        removeBtn.innerHTML = '&#10005;';
        removeBtn.addEventListener('click', function () {
          layers.splice(idx, 1);
          renderLayers();
        });
        row.appendChild(removeBtn);

        layersList.appendChild(row);
      });

      // Summary
      if (layers.length > 0) {
        summaryDiv.style.display = 'block';
        var html = '<h3>Model Summary</h3>';
        breakdown.forEach(function (b) {
          html += '<div class="summary-row"><span>' + escHtml(b.name) + '</span><span class="val">' + formatNum(b.params) + '</span></div>';
        });
        html += '<div class="summary-total"><span>Total Parameters</span><span class="val">' + formatNum(totalParams) + '</span></div>';
        html += '<div style="margin-top:12px;font-size:0.85rem;color:var(--text-dim);">' +
          'Model size (float32): ' + formatBytes(totalParams * 4) +
          ' | Model size (float16): ' + formatBytes(totalParams * 2) +
          '</div>';
        summaryDiv.innerHTML = html;
      } else {
        summaryDiv.style.display = 'none';
      }
    }

    addBtn.addEventListener('click', function () { addLayer(addSelect.value); });
    clearBtn.addEventListener('click', function () { layers = []; renderLayers(); });

    // Pre-populate with a simple example
    addLayer('Conv2d');
    addLayer('Conv2d');
    addLayer('Linear');
  }

  /* ── Build Memory Calculator ── */
  function buildMemoryCalc(container) {
    var card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<h3>GPU Memory Estimator</h3><p class="plot-info">Estimate the GPU VRAM required to train your model. Enter the total parameter count and training configuration.</p>';

    var grid = document.createElement('div');
    grid.className = 'params-grid';

    var fields = [
      { key: 'params', label: 'Total Parameters (millions)', default: 100 },
      { key: 'batch_size', label: 'Batch Size', default: 32 },
      { key: 'seq_len', label: 'Sequence Length (0 if CNN)', default: 512 },
      { key: 'hidden_dim', label: 'Hidden Dimension', default: 768 },
      { key: 'num_layers', label: 'Number of Layers', default: 12 },
      { key: 'precision', label: 'Precision (32=fp32, 16=fp16)', default: 32 }
    ];

    var inputs = {};
    fields.forEach(function (f) {
      var fieldDiv = document.createElement('div');
      fieldDiv.className = 'input-field';
      fieldDiv.innerHTML = '<label>' + escHtml(f.label) + '</label>';
      var inp = document.createElement('input');
      inp.type = 'number';
      inp.value = f.default;
      fieldDiv.appendChild(inp);
      grid.appendChild(fieldDiv);
      inputs[f.key] = inp;
    });

    card.appendChild(grid);

    // Optimizer select
    var optDiv = document.createElement('div');
    optDiv.className = 'input-field';
    optDiv.style.cssText = 'max-width:300px;margin-bottom:16px;';
    optDiv.innerHTML = '<label>Optimizer</label>';
    var optSelect = document.createElement('select');
    [
      { val: 'adam', label: 'Adam / AdamW (2x param states)' },
      { val: 'sgd-m', label: 'SGD + Momentum (1x param states)' },
      { val: 'sgd', label: 'SGD (no extra states)' }
    ].forEach(function (o) {
      var opt = document.createElement('option');
      opt.value = o.val;
      opt.textContent = o.label;
      optSelect.appendChild(opt);
    });
    optDiv.appendChild(optSelect);
    card.appendChild(optDiv);

    // Checkboxes
    var checkDiv = document.createElement('div');
    checkDiv.style.cssText = 'display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;font-size:0.85rem;';
    var gradCheckpoint = document.createElement('label');
    gradCheckpoint.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;';
    var gcInput = document.createElement('input');
    gcInput.type = 'checkbox';
    gradCheckpoint.appendChild(gcInput);
    gradCheckpoint.appendChild(document.createTextNode('Gradient Checkpointing'));
    checkDiv.appendChild(gradCheckpoint);
    card.appendChild(checkDiv);

    var calcBtn = document.createElement('button');
    calcBtn.className = 'btn btn-primary';
    calcBtn.textContent = 'Estimate Memory';
    card.appendChild(calcBtn);

    var resultDiv = document.createElement('div');
    resultDiv.className = 'model-summary';
    resultDiv.style.display = 'none';
    card.appendChild(resultDiv);

    container.appendChild(card);

    // GPU reference card
    var gpuCard = document.createElement('div');
    gpuCard.className = 'card';
    gpuCard.innerHTML = '<h3>GPU Memory Reference</h3>' +
      '<div class="ref-table-wrap"><table class="ref-table">' +
      '<tr><th>GPU</th><th>VRAM</th><th>Approx Max Params (Training, fp32, Adam)</th></tr>' +
      '<tr><td>RTX 3060</td><td>12 GB</td><td>~150M</td></tr>' +
      '<tr><td>RTX 3090 / 4090</td><td>24 GB</td><td>~350M</td></tr>' +
      '<tr><td>A6000</td><td>48 GB</td><td>~750M</td></tr>' +
      '<tr><td>A100</td><td>80 GB</td><td>~1.2B</td></tr>' +
      '<tr><td>H100</td><td>80 GB</td><td>~1.5B</td></tr>' +
      '</table></div>' +
      '<p class="plot-info">These are rough estimates for training. Actual limits depend on batch size, sequence length, and activation memory. Mixed precision (fp16) roughly doubles the effective capacity.</p>';
    container.appendChild(gpuCard);

    function calculate() {
      var paramsMil = parseFloat(inputs.params.value) || 0;
      var totalParams = paramsMil * 1e6;
      var batchSize = parseInt(inputs.batch_size.value, 10) || 1;
      var seqLen = parseInt(inputs.seq_len.value, 10) || 0;
      var hiddenDim = parseInt(inputs.hidden_dim.value, 10) || 768;
      var numLayers = parseInt(inputs.num_layers.value, 10) || 1;
      var precision = parseInt(inputs.precision.value, 10) || 32;
      var bytesPerParam = precision / 8;
      var optType = optSelect.value;
      var useGC = gcInput.checked;

      // Model params memory
      var paramMemory = totalParams * bytesPerParam;

      // Gradients (same size as params)
      var gradMemory = totalParams * bytesPerParam;

      // Optimizer states
      var optStateMultiplier = 0;
      if (optType === 'adam') optStateMultiplier = 2; // m + v in fp32
      else if (optType === 'sgd-m') optStateMultiplier = 1; // momentum buffer
      // Optimizer states are always in fp32
      var optMemory = totalParams * 4 * optStateMultiplier;

      // Activation memory (rough estimate)
      var activationMemory;
      if (seqLen > 0) {
        // Transformer-like: each layer stores activations of size batch * seq_len * hidden_dim
        // Plus attention maps: batch * num_heads * seq_len * seq_len
        var numHeads = Math.max(1, Math.round(hiddenDim / 64));
        var perLayerAct = batchSize * seqLen * hiddenDim * bytesPerParam * 4; // 4 intermediate tensors roughly
        var attentionMaps = batchSize * numHeads * seqLen * seqLen * bytesPerParam;
        activationMemory = (perLayerAct + attentionMaps) * numLayers;
      } else {
        // CNN-like: rough estimate based on params * batch ratio
        activationMemory = totalParams * bytesPerParam * batchSize * 0.5;
      }

      if (useGC) {
        // Gradient checkpointing saves ~60% of activation memory
        activationMemory *= 0.4;
      }

      // Mixed precision master weights (if fp16, keep fp32 copy for optimizer)
      var masterWeightMemory = 0;
      if (precision === 16) {
        masterWeightMemory = totalParams * 4; // fp32 master copy
      }

      var totalMemory = paramMemory + gradMemory + optMemory + activationMemory + masterWeightMemory;

      // Breakdown
      resultDiv.style.display = 'block';
      resultDiv.innerHTML =
        '<h3>Memory Breakdown</h3>' +
        '<div class="summary-row"><span>Model Parameters (' + (precision === 16 ? 'fp16' : 'fp32') + ')</span><span class="val">' + formatBytes(paramMemory) + '</span></div>' +
        '<div class="summary-row"><span>Gradients</span><span class="val">' + formatBytes(gradMemory) + '</span></div>' +
        '<div class="summary-row"><span>Optimizer States (' + optType + ')</span><span class="val">' + formatBytes(optMemory) + '</span></div>' +
        '<div class="summary-row"><span>Activations' + (useGC ? ' (with grad. ckpt.)' : '') + '</span><span class="val">' + formatBytes(activationMemory) + '</span></div>' +
        (masterWeightMemory > 0 ? '<div class="summary-row"><span>FP32 Master Weights</span><span class="val">' + formatBytes(masterWeightMemory) + '</span></div>' : '') +
        '<div class="summary-total"><span>Estimated Total VRAM</span><span class="val">' + formatBytes(totalMemory) + '</span></div>' +
        '<p class="plot-info" style="margin-top:12px;">This is an estimate. Actual memory usage depends on PyTorch memory allocator overhead, CUDA context (~300-800MB), and peak memory during forward/backward passes. Add ~20% buffer for safety.</p>';
    }

    calcBtn.addEventListener('click', calculate);

    // Auto-calculate
    calculate();
  }

  /* ── Init ── */
  window.HeyTensor.components['model-tool'] = {
    init: function (container, config) {
      var mode = config.mode || 'param-counter';
      switch (mode) {
        case 'param-counter': buildParamCounter(container); break;
        case 'memory': buildMemoryCalc(container); break;
        default:
          container.innerHTML = '<p style="color:var(--red);">Unknown mode: ' + escHtml(mode) + '</p>';
      }
    }
  };
})();
