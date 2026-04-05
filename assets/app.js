/* HeyTensor — Tensor Shape Calculator & Error Debugger */
/* All functions under 60 lines. Loops bounded to max 1000 iterations. */

(function() {
  'use strict';

  var state = {
    framework: 'pytorch',
    activeTab: 'single',
    chainLayers: [],
    chainInput: [1, 3, 224, 224]
  };

  /* ===== LAYER CALCULATIONS ===== */

  function calcConv2d(input, params) {
    var b = input[0], c = input[1], h = input[2], w = input[3];
    var outC = params.out_channels || 64;
    var kH = params.kernel_h || 3, kW = params.kernel_w || 3;
    var sH = params.stride_h || 1, sW = params.stride_w || 1;
    var pH = params.pad_h || 0, pW = params.pad_w || 0;
    var dH = params.dilation_h || 1, dW = params.dilation_w || 1;
    var outH = Math.floor((h + 2 * pH - dH * (kH - 1) - 1) / sH) + 1;
    var outW = Math.floor((w + 2 * pW - dW * (kW - 1) - 1) / sW) + 1;
    if (outH <= 0 || outW <= 0) return { error: 'Output dimensions <= 0. Check kernel/stride/padding.' };
    var formula = 'H_out = floor((' + h + ' + 2*' + pH + ' - ' + dH + '*(' + kH + '-1) - 1) / ' + sH + ') + 1 = ' + outH;
    formula += '\nW_out = floor((' + w + ' + 2*' + pW + ' - ' + dW + '*(' + kW + '-1) - 1) / ' + sW + ') + 1 = ' + outW;
    return { shape: [b, outC, outH, outW], formula: formula };
  }

  function calcConv1d(input, params) {
    var b = input[0], c = input[1], l = input[2];
    var outC = params.out_channels || 64;
    var k = params.kernel || 3;
    var s = params.stride || 1;
    var p = params.padding || 0;
    var d = params.dilation || 1;
    var outL = Math.floor((l + 2 * p - d * (k - 1) - 1) / s) + 1;
    if (outL <= 0) return { error: 'Output length <= 0.' };
    var formula = 'L_out = floor((' + l + ' + 2*' + p + ' - ' + d + '*(' + k + '-1) - 1) / ' + s + ') + 1 = ' + outL;
    return { shape: [b, outC, outL], formula: formula };
  }

  function calcLinear(input, params) {
    var flat = input.slice();
    var inFeatures = flat[flat.length - 1];
    var outFeatures = params.out_features || 128;
    var formula = 'Output = [' + flat.slice(0, -1).join(', ') + ', ' + outFeatures + ']';
    formula += '\nLinear(in_features=' + inFeatures + ', out_features=' + outFeatures + ')';
    var outShape = flat.slice(0, -1);
    outShape.push(outFeatures);
    return { shape: outShape, formula: formula };
  }

  function calcLSTM(input, params) {
    var b = input[0], seq = input[1], feat = input[2];
    var hidden = params.hidden_size || 128;
    var bidir = params.bidirectional ? 2 : 1;
    var outFeat = hidden * bidir;
    var formula = 'Output: [' + b + ', ' + seq + ', ' + outFeat + ']';
    formula += '\nHidden: ' + hidden + (params.bidirectional ? ' x 2 (bidirectional)' : '');
    return { shape: [b, seq, outFeat], formula: formula };
  }

  function calcGRU(input, params) {
    var b = input[0], seq = input[1], feat = input[2];
    var hidden = params.hidden_size || 128;
    var bidir = params.bidirectional ? 2 : 1;
    var outFeat = hidden * bidir;
    var formula = 'Output: [' + b + ', ' + seq + ', ' + outFeat + ']';
    return { shape: [b, seq, outFeat], formula: formula };
  }

  function calcMultiheadAttention(input, params) {
    var b = input[0], seq = input[1], embed = input[2];
    var numHeads = params.num_heads || 8;
    if (embed % numHeads !== 0) {
      return { error: 'embed_dim (' + embed + ') must be divisible by num_heads (' + numHeads + ')' };
    }
    var formula = 'Output: [' + b + ', ' + seq + ', ' + embed + ']';
    formula += '\nnum_heads=' + numHeads + ', head_dim=' + (embed / numHeads);
    return { shape: [b, seq, embed], formula: formula };
  }

  function calcBatchNorm(input, params) {
    var formula = 'BatchNorm does not change shape. Output = Input: [' + input.join(', ') + ']';
    return { shape: input.slice(), formula: formula };
  }

  function calcMaxPool2d(input, params) {
    var b = input[0], c = input[1], h = input[2], w = input[3];
    var k = params.kernel || 2, s = params.stride || k, p = params.padding || 0;
    var outH = Math.floor((h + 2 * p - k) / s) + 1;
    var outW = Math.floor((w + 2 * p - k) / s) + 1;
    if (outH <= 0 || outW <= 0) return { error: 'Output dimensions <= 0.' };
    var formula = 'H_out = floor((' + h + ' + 2*' + p + ' - ' + k + ') / ' + s + ') + 1 = ' + outH;
    formula += '\nW_out = floor((' + w + ' + 2*' + p + ' - ' + k + ') / ' + s + ') + 1 = ' + outW;
    return { shape: [b, c, outH, outW], formula: formula };
  }

  function calcAvgPool2d(input, params) {
    return calcMaxPool2d(input, params);
  }

  function calcFlatten(input, params) {
    var startDim = params.start_dim || 1;
    var endDim = params.end_dim || (input.length - 1);
    if (startDim < 0) startDim = input.length + startDim;
    if (endDim < 0) endDim = input.length + endDim;
    var flat = 1;
    for (var i = startDim; i <= endDim && i < input.length; i++) {
      flat *= input[i];
    }
    var out = [];
    for (var j = 0; j < startDim && j < input.length; j++) out.push(input[j]);
    out.push(flat);
    for (var k = endDim + 1; k < input.length; k++) out.push(input[k]);
    var formula = 'Flatten dims ' + startDim + ' to ' + endDim + ': ' + flat + ' features';
    return { shape: out, formula: formula };
  }

  function calcDropout(input, params) {
    var formula = 'Dropout does not change shape. Output = Input: [' + input.join(', ') + ']';
    return { shape: input.slice(), formula: formula };
  }

  function calcTranspose(input, params) {
  if ((params.dim0 || 0) >= input.length || (params.dim1 || 1) >= input.length) return { error: 'Transpose dimension out of range' };
    var d0 = params.dim0 || 0, d1 = params.dim1 || 1;
    var out = input.slice();
    var tmp = out[d0];
    out[d0] = out[d1];
    out[d1] = tmp;
    var formula = 'Transpose dims ' + d0 + ' and ' + d1;
    return { shape: out, formula: formula };
  }

  function calcReshape(input, params) {
    var targetStr = params.target_shape || '';
    var parts = targetStr.split(',').map(function(s) { return parseInt(s.trim(), 10); });
    var totalIn = 1;
    for (var i = 0; i < input.length; i++) totalIn *= input[i];
    var negIdx = -1, totalOut = 1;
    for (var j = 0; j < parts.length && j < 10; j++) {
      if (parts[j] === -1) { negIdx = j; }
      else { totalOut *= parts[j]; }
    }
    if (negIdx >= 0) { var inferred = totalIn / totalOut; if (inferred !== Math.round(inferred)) return { error: 'Cannot reshape: dimensions not evenly divisible' }; parts[negIdx] = Math.round(inferred); }
    var formula = 'Reshape [' + input.join(', ') + '] -> [' + parts.join(', ') + ']';
    return { shape: parts, formula: formula };
  }

  function calcConcatenate(input, params) {
  if ((params.dim || 0) >= input.length) return { error: 'Concatenation dimension out of range' };
    var dim = params.dim || 0;
    var addSize = params.concat_size || input[dim];
    var out = input.slice();
    out[dim] = input[dim] + addSize;
    var formula = 'Concatenate along dim ' + dim + ': ' + input[dim] + ' + ' + addSize + ' = ' + out[dim];
    return { shape: out, formula: formula };
  }

  var layerCalcs = {
    'Conv2d': calcConv2d,
    'Conv1d': calcConv1d,
    'Linear': calcLinear,
    'LSTM': calcLSTM,
    'GRU': calcGRU,
    'MultiheadAttention': calcMultiheadAttention,
    'BatchNorm': calcBatchNorm,
    'MaxPool2d': calcMaxPool2d,
    'AvgPool2d': calcAvgPool2d,
    'Flatten': calcFlatten,
    'Dropout': calcDropout,
    'Transpose': calcTranspose,
    'Reshape': calcReshape,
    'Concatenate': calcConcatenate
  };

  /* ===== LAYER PARAM DEFINITIONS ===== */

  var layerParams = {
    'Conv2d': ['out_channels:64', 'kernel_h:3', 'kernel_w:3', 'stride_h:1', 'stride_w:1', 'pad_h:0', 'pad_w:0', 'dilation_h:1', 'dilation_w:1'],
    'Conv1d': ['out_channels:64', 'kernel:3', 'stride:1', 'padding:0', 'dilation:1'],
    'Linear': ['out_features:128'],
    'LSTM': ['hidden_size:128', 'bidirectional:0'],
    'GRU': ['hidden_size:128', 'bidirectional:0'],
    'MultiheadAttention': ['num_heads:8'],
    'BatchNorm': [],
    'MaxPool2d': ['kernel:2', 'stride:2', 'padding:0'],
    'AvgPool2d': ['kernel:2', 'stride:2', 'padding:0'],
    'Flatten': ['start_dim:1', 'end_dim:-1'],
    'Dropout': [],
    'Transpose': ['dim0:0', 'dim1:1'],
    'Reshape': ['target_shape:1,-1'],
    'Concatenate': ['dim:1', 'concat_size:64']
  };

  var tfNames = {
    'Conv2d': 'Conv2D',
    'Conv1d': 'Conv1D',
    'Linear': 'Dense',
    'LSTM': 'LSTM',
    'GRU': 'GRU',
    'MultiheadAttention': 'MultiHeadAttention',
    'BatchNorm': 'BatchNormalization',
    'MaxPool2d': 'MaxPooling2D',
    'AvgPool2d': 'AveragePooling2D',
    'Flatten': 'Flatten',
    'Dropout': 'Dropout',
    'Transpose': 'Permute',
    'Reshape': 'Reshape',
    'Concatenate': 'Concatenate'
  };

  /* ===== SINGLE LAYER MODE ===== */

  function buildSingleLayerUI() {
    var container = document.getElementById('single-content');
    if (!container) return;
    var html = '<div class="framework-toggle">';
    html += '<span>PyTorch</span>';
    html += '<div class="toggle-switch" id="fw-toggle" onclick="window.HT.toggleFramework()"></div>';
    html += '<span>TensorFlow</span></div>';
    html += '<div class="card"><h3>Select Layer Type</h3>';
    html += '<select id="layer-select" onchange="window.HT.onLayerChange()">';
    var types = Object.keys(layerCalcs);
    for (var i = 0; i < types.length; i++) {
      html += '<option value="' + types[i] + '">' + types[i] + '</option>';
    }
    html += '</select></div>';
    html += '<div class="card"><h3>Input Shape</h3>';
    html += '<div class="input-group" id="single-input-shape">';
    html += buildInputShapeFields([1, 3, 224, 224]);
    html += '</div></div>';
    html += '<div class="card"><h3 id="single-params-title">Layer Parameters</h3>';
    html += '<div class="input-group" id="single-params"></div>';
    html += '<button class="btn btn-primary" onclick="window.HT.calcSingle()">Calculate Shape</button>';
    html += '</div>';
    html += '<div id="single-result"></div>';
    html += '<div id="single-viz" class="tensor-viz"></div>';
    container.innerHTML = html;
    onLayerChange();
  }

  function buildInputShapeFields(defaults) {
    var labels = ['Batch', 'Dim 1', 'Dim 2', 'Dim 3'];
    var html = '';
    for (var i = 0; i < defaults.length && i < 6; i++) {
      html += '<div class="input-field">';
      html += '<label>' + (labels[i] || 'Dim ' + i) + '</label>';
      html += '<input type="number" class="shape-input" value="' + defaults[i] + '" min="1">';
      html += '</div>';
    }
    return html;
  }

  function onLayerChange() {
    var sel = document.getElementById('layer-select');
    if (!sel) return;
    var type = sel.value;
    var paramsDiv = document.getElementById('single-params');
    var defs = layerParams[type] || [];
    var html = '';
    for (var i = 0; i < defs.length; i++) {
      var parts = defs[i].split(':');
      var name = parts[0], def = parts[1];
      html += '<div class="input-field">';
      html += '<label>' + name + '</label>';
      if (name === 'target_shape') {
        html += '<input type="text" id="param-' + name + '" value="' + def + '">';
      } else {
        html += '<input type="number" id="param-' + name + '" value="' + def + '">';
      }
      html += '</div>';
    }
    paramsDiv.innerHTML = html;
    updateFrameworkLabels();
  }

  function getInputShape() {
    var inputs = document.querySelectorAll('#single-input-shape .shape-input');
    var shape = [];
    for (var i = 0; i < inputs.length && i < 10; i++) {
      shape.push(parseInt(inputs[i].value, 10) || 1);
    }
    return shape;
  }

  function getParams(type) {
    var defs = layerParams[type] || [];
    var params = {};
    for (var i = 0; i < defs.length; i++) {
      var name = defs[i].split(':')[0];
      var el = document.getElementById('param-' + name);
      if (!el) continue;
      if (name === 'target_shape') {
        params[name] = el.value;
      } else if (name === 'bidirectional') {
        params[name] = parseInt(el.value, 10) === 1;
      } else {
        params[name] = parseInt(el.value, 10) || 0;
      }
    }
    return params;
  }

  function calcSingle() {
    var type = document.getElementById('layer-select').value;
    var input = getInputShape();
    var params = getParams(type);
    var result = layerCalcs[type](input, params);
    var resDiv = document.getElementById('single-result');
    var vizDiv = document.getElementById('single-viz');
    if (result.error) {
      resDiv.innerHTML = '<div class="result-box error"><div class="shape">Error: ' + result.error + '</div></div>';
      vizDiv.innerHTML = '';
    } else {
      var html = '<div class="result-box success">';
      html += '<div class="shape">Output: [' + result.shape.join(', ') + ']</div>';
      html += '<div class="formula">' + result.formula.replace(/\n/g, '<br>') + '</div>';
      html += '</div>';
      resDiv.innerHTML = html;
      vizDiv.innerHTML = buildTensorViz(input, result.shape);
    }
  }

  function buildTensorViz(inputShape, outputShape) {
    var html = '';
    html += buildSingleTensorBox(inputShape, 'Input');
    html += '<span class="tensor-arrow">&#10140;</span>';
    html += buildSingleTensorBox(outputShape, 'Output');
    return html;
  }

  function buildSingleTensorBox(shape, label) {
    var maxDim = 1;
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] > maxDim) maxDim = shape[i];
    }
    var scale = Math.min(150 / maxDim, 1);
    var w = Math.max(30, Math.min(180, (shape.length > 3 ? shape[3] : shape[shape.length - 1]) * scale + 20));
    var h = Math.max(30, Math.min(100, (shape.length > 2 ? shape[2] : shape[Math.min(1, shape.length - 1)]) * scale + 20));
    var html = '<div style="text-align:center">';
    html += '<div class="tensor-box" style="width:' + w + 'px;height:' + h + 'px;">';
    html += '[' + shape.join(',') + ']</div>';
    html += '<div style="font-size:0.7rem;color:var(--text-dim);margin-top:4px;">' + label + '</div></div>';
    return html;
  }

  /* ===== CHAIN MODE ===== */

  function buildChainUI() {
    var container = document.getElementById('chain-content');
    if (!container) return;
    var html = '<div class="card"><h3>Input Shape (first layer)</h3>';
    html += '<div class="input-group" id="chain-input-fields">';
    html += '<div class="input-field"><label>Batch</label><input type="number" id="chain-b" value="1" min="1"></div>';
    html += '<div class="input-field"><label>Channels</label><input type="number" id="chain-c" value="3" min="1"></div>';
    html += '<div class="input-field"><label>Height</label><input type="number" id="chain-h" value="224" min="1"></div>';
    html += '<div class="input-field"><label>Width</label><input type="number" id="chain-w" value="224" min="1"></div>';
    html += '</div></div>';
    html += '<div class="chain-controls">';
    html += '<select id="chain-add-type">';
    var types = Object.keys(layerCalcs);
    for (var i = 0; i < types.length; i++) {
      html += '<option value="' + types[i] + '">' + types[i] + '</option>';
    }
    html += '</select>';
    html += '<button class="btn btn-primary btn-sm" onclick="window.HT.addChainLayer()">+ Add Layer</button>';
    html += '<button class="btn btn-secondary btn-sm" onclick="window.HT.clearChain()">Clear All</button>';
    html += '</div>';
    html += '<div id="chain-layers"></div>';
    html += '<div id="chain-summary"></div>';
    container.innerHTML = html;
  }

  function addChainLayer() {
    var type = document.getElementById('chain-add-type').value;
    var defs = layerParams[type] || [];
    var params = {};
    for (var i = 0; i < defs.length; i++) {
      var parts = defs[i].split(':');
      params[parts[0]] = parts[0] === 'target_shape' ? parts[1] : (parts[0] === 'bidirectional' ? false : parseInt(parts[1], 10));
    }
    state.chainLayers.push({ type: type, params: params });
    renderChain();
  }

  function removeChainLayer(idx) {
    state.chainLayers.splice(idx, 1);
    renderChain();
  }

  function clearChain() {
    state.chainLayers = [];
    renderChain();
  }

  function getChainInput() {
    return [
      parseInt(document.getElementById('chain-b').value, 10) || 1,
      parseInt(document.getElementById('chain-c').value, 10) || 3,
      parseInt(document.getElementById('chain-h').value, 10) || 224,
      parseInt(document.getElementById('chain-w').value, 10) || 224
    ];
  }

  function renderChain() {
    var container = document.getElementById('chain-layers');
    var summary = document.getElementById('chain-summary');
    if (!container) return;
    var currentShape = getChainInput();
    var html = '<div class="chain-layer"><span class="chain-shape" style="width:100%">Input: [' + currentShape.join(', ') + ']</span></div>';
    html += '<div class="chain-arrow">&#8595;</div>';
    var hasError = false;
    for (var i = 0; i < state.chainLayers.length && i < 100; i++) {
      var layer = state.chainLayers[i];
      var result = layerCalcs[layer.type](currentShape, layer.params);
      var isMismatch = !!result.error;
      if (isMismatch) hasError = true;
      html += '<div class="chain-layer' + (isMismatch ? ' mismatch' : '') + '">';
      html += '<span class="layer-num">#' + (i + 1) + '</span>';
      html += '<strong style="min-width:100px;font-size:0.85rem;">' + layer.type + '</strong>';
      html += '<span style="flex:1;font-size:0.75rem;color:var(--text-dim)">' + formatParams(layer.params) + '</span>';
      if (isMismatch) {
        html += '<span class="chain-shape err">' + result.error + '</span>';
      } else {
        html += '<span class="chain-shape">[' + result.shape.join(', ') + ']</span>';
        currentShape = result.shape;
      }
      html += '<button class="remove-layer-btn" onclick="window.HT.removeChainLayer(' + i + ')">&times;</button>';
      html += '</div>';
      if (i < state.chainLayers.length - 1) {
        html += '<div class="chain-arrow">&#8595;</div>';
      }
    }
    container.innerHTML = html;
    if (state.chainLayers.length > 0) {
      var sumHtml = '<div class="result-box ' + (hasError ? 'error' : 'success') + '">';
      if (hasError) {
        sumHtml += '<div class="shape">Chain has shape mismatches — fix red layers above</div>';
      } else {
        sumHtml += '<div class="shape">Final Output: [' + currentShape.join(', ') + ']</div>';
      }
      sumHtml += '</div>';
      summary.innerHTML = sumHtml;
    } else {
      summary.innerHTML = '';
    }
  }

  function formatParams(params) {
    var parts = [];
    var keys = Object.keys(params);
    for (var i = 0; i < keys.length && i < 20; i++) {
      parts.push(keys[i] + '=' + params[keys[i]]);
    }
    return parts.join(', ');
  }

  /* ===== ERROR PASTE MODE ===== */

  function buildErrorUI() {
    var container = document.getElementById('error-content');
    if (!container) return;
    var html = '<div class="card">';
    html += '<h3>Paste Your PyTorch Error</h3>';
    html += '<p style="color:var(--text-dim);font-size:0.85rem;margin-bottom:12px;">Paste a RuntimeError from PyTorch and we\'ll extract the dimensions and suggest fixes.</p>';
    html += '<textarea id="error-input" placeholder="e.g. RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)"></textarea>';
    html += '<button class="btn btn-primary" onclick="window.HT.parseError()" style="margin-top:12px;">Analyze Error</button>';
    html += '</div>';
    html += '<div id="error-result"></div>';
    container.innerHTML = html;
  }

  var errorPatterns = [
    {
      name: 'Matrix multiplication mismatch',
      regex: /mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)/,
      handler: function(m) {
        var a = m[1], b = m[2], c = m[3], d = m[4];
        return {
          extracted: 'mat1: [' + a + ' x ' + b + '], mat2: [' + c + ' x ' + d + ']',
          cause: 'For matrix multiplication, mat1 columns (' + b + ') must equal mat2 rows (' + c + ').',
          fix: 'Change your Linear layer: nn.Linear(' + b + ', out_features) or adjust the previous layer to output ' + c + ' features.'
        };
      }
    },
    {
      name: 'Size mismatch (m1/m2)',
      regex: /size mismatch,?\s*m1:\s*\[(\d+)\s*x\s*(\d+)\],?\s*m2:\s*\[(\d+)\s*x\s*(\d+)\]/,
      handler: function(m) {
        return {
          extracted: 'm1: [' + m[1] + ' x ' + m[2] + '], m2: [' + m[3] + ' x ' + m[4] + ']',
          cause: 'm1 columns (' + m[2] + ') != m2 rows (' + m[3] + ').',
          fix: 'Set in_features=' + m[2] + ' in your Linear layer, or flatten/reshape to match ' + m[3] + '.'
        };
      }
    },
    {
      name: 'Batch size mismatch',
      regex: /Expected input batch_size \((\d+)\) to match target batch_size \((\d+)\)/,
      handler: function(m) {
        return {
          extracted: 'Input batch: ' + m[1] + ', Target batch: ' + m[2],
          cause: 'Your input tensor and target tensor have different batch sizes.',
          fix: 'Ensure both tensors have the same batch dimension. Check your data loader and reshape operations.'
        };
      }
    },
    {
      name: 'Expected 4D input (Conv2d)',
      regex: /Expected (\d+)D.*input.*got (\d+)D/i,
      handler: function(m) {
        return {
          extracted: 'Expected ' + m[1] + 'D, got ' + m[2] + 'D',
          cause: 'Conv2d expects (batch, channels, height, width). Your input has wrong number of dimensions.',
          fix: 'Use .unsqueeze() to add dimensions or .view()/.reshape() to get ' + m[1] + ' dimensions.'
        };
      }
    },
    {
      name: 'Shape mismatch in view/reshape',
      regex: /shape '([^']+)' is invalid for input of size (\d+)/,
      handler: function(m) {
        return {
          extracted: 'Target shape: ' + m[1] + ', Input size: ' + m[2],
          cause: 'Total elements in target shape do not match input total of ' + m[2] + '.',
          fix: 'Ensure product of target dimensions equals ' + m[2] + '. Use -1 for one dimension to auto-calculate.'
        };
      }
    },
    {
      name: 'Channel mismatch',
      regex: /Given groups=(\d+), weight of size \[(\d+), (\d+)/,
      handler: function(m) {
        return {
          extracted: 'Groups: ' + m[1] + ', Weight: [' + m[2] + ', ' + m[3] + ', ...]',
          cause: 'Conv layer expects ' + m[3] + ' input channels but received different.',
          fix: 'Set in_channels=' + m[3] + ' or adjust previous layer output channels to match.'
        };
      }
    },
    {
      name: 'Mismatched tensor sizes for operation',
      regex: /The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\) at.*dimension (\d+)/,
      handler: function(m) {
        return {
          extracted: 'Tensor a: ' + m[1] + ' vs Tensor b: ' + m[2] + ' at dim ' + m[3],
          cause: 'Element-wise operation requires matching sizes at dimension ' + m[3] + '.',
          fix: 'Reshape one tensor to match, or use broadcasting (one dim should be 1).'
        };
      }
    }
  ];

  function parseError() {
    var input = document.getElementById('error-input').value.trim();
    var resultDiv = document.getElementById('error-result');
    if (!input) {
      resultDiv.innerHTML = '<div class="result-box error"><div class="shape">Please paste an error message.</div></div>';
      return;
    }
    var matched = false;
    var html = '';
    for (var i = 0; i < errorPatterns.length; i++) {
      var m = input.match(errorPatterns[i].regex);
      if (m) {
        matched = true;
        var res = errorPatterns[i].handler(m);
        html += '<div class="error-match">';
        html += '<h4>' + errorPatterns[i].name + '</h4>';
        html += '<div class="extracted">' + res.extracted + '</div>';
        html += '<p style="color:var(--text-dim);margin-bottom:8px;"><strong>Cause:</strong> ' + res.cause + '</p>';
        html += '<p class="suggestion"><strong>Fix:</strong> ' + res.fix + '</p>';
        html += '</div>';
        break;
      }
    }
    if (!matched) {
      html = '<div class="result-box"><div class="shape" style="color:var(--yellow)">Could not match a known error pattern.</div>';
      html += '<div class="formula">Try pasting the exact RuntimeError line. Supported patterns:<br>';
      html += '- mat1 and mat2 shapes cannot be multiplied<br>';
      html += '- size mismatch, m1: [...], m2: [...]<br>';
      html += '- Expected input batch_size (X) to match target batch_size (Y)<br>';
      html += '- Expected ND input got MD<br>';
      html += '- shape is invalid for input of size N<br>';
      html += '- Given groups=N, weight of size [...]<br>';
      html += '- size of tensor a (X) must match size of tensor b (Y)</div></div>';
    }
    resultDiv.innerHTML = html;
  }

  /* ===== PRESETS MODE ===== */

  var architecturePresets = {
    'LeNet-5': {
      input: [1, 1, 32, 32],
      layers: [
        { type: 'Conv2d', params: { out_channels: 6, kernel_h: 5, kernel_w: 5, stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0, dilation_h: 1, dilation_w: 1 } },
        { type: 'AvgPool2d', params: { kernel: 2, stride: 2, padding: 0 } },
        { type: 'Conv2d', params: { out_channels: 16, kernel_h: 5, kernel_w: 5, stride_h: 1, stride_w: 1, pad_h: 0, pad_w: 0, dilation_h: 1, dilation_w: 1 } },
        { type: 'AvgPool2d', params: { kernel: 2, stride: 2, padding: 0 } },
        { type: 'Flatten', params: { start_dim: 1, end_dim: -1 } },
        { type: 'Linear', params: { out_features: 120 } },
        { type: 'Linear', params: { out_features: 84 } },
        { type: 'Linear', params: { out_features: 10 } }
      ]
    },
    'Simple CNN': {
      input: [1, 3, 224, 224],
      layers: [
        { type: 'Conv2d', params: { out_channels: 32, kernel_h: 3, kernel_w: 3, stride_h: 1, stride_w: 1, pad_h: 1, pad_w: 1, dilation_h: 1, dilation_w: 1 } },
        { type: 'BatchNorm', params: {} },
        { type: 'MaxPool2d', params: { kernel: 2, stride: 2, padding: 0 } },
        { type: 'Conv2d', params: { out_channels: 64, kernel_h: 3, kernel_w: 3, stride_h: 1, stride_w: 1, pad_h: 1, pad_w: 1, dilation_h: 1, dilation_w: 1 } },
        { type: 'BatchNorm', params: {} },
        { type: 'MaxPool2d', params: { kernel: 2, stride: 2, padding: 0 } },
        { type: 'Flatten', params: { start_dim: 1, end_dim: -1 } },
        { type: 'Linear', params: { out_features: 256 } },
        { type: 'Dropout', params: {} },
        { type: 'Linear', params: { out_features: 10 } }
      ]
    },
    'ResNet Block': {
      input: [1, 64, 56, 56],
      layers: [
        { type: 'Conv2d', params: { out_channels: 64, kernel_h: 3, kernel_w: 3, stride_h: 1, stride_w: 1, pad_h: 1, pad_w: 1, dilation_h: 1, dilation_w: 1 } },
        { type: 'BatchNorm', params: {} },
        { type: 'Conv2d', params: { out_channels: 64, kernel_h: 3, kernel_w: 3, stride_h: 1, stride_w: 1, pad_h: 1, pad_w: 1, dilation_h: 1, dilation_w: 1 } },
        { type: 'BatchNorm', params: {} }
      ]
    },
    'Transformer Encoder': {
      input: [1, 50, 512],
      layers: [
        { type: 'MultiheadAttention', params: { num_heads: 8 } },
        { type: 'Linear', params: { out_features: 2048 } },
        { type: 'Dropout', params: {} },
        { type: 'Linear', params: { out_features: 512 } },
        { type: 'Dropout', params: {} }
      ]
    },
    'LSTM Classifier': {
      input: [1, 100, 300],
      layers: [
        { type: 'LSTM', params: { hidden_size: 256, bidirectional: true } },
        { type: 'Flatten', params: { start_dim: 1, end_dim: -1 } },
        { type: 'Linear', params: { out_features: 128 } },
        { type: 'Dropout', params: {} },
        { type: 'Linear', params: { out_features: 5 } }
      ]
    }
  };

  function buildPresetsUI() {
    var container = document.getElementById('presets-content');
    if (!container) return;
    var html = '<div class="card"><h3>Architecture Presets</h3>';
    html += '<p style="color:var(--text-dim);font-size:0.85rem;margin-bottom:16px;">Click a preset to load it into Chain Mode and see shapes at every layer.</p>';
    html += '<div class="preset-grid">';
    var names = Object.keys(architecturePresets);
    for (var i = 0; i < names.length; i++) {
      var p = architecturePresets[names[i]];
      html += '<div class="preset-card" onclick="window.HT.loadPreset(\'' + names[i] + '\')">';
      html += '<h4>' + names[i] + '</h4>';
      html += '<p>Input: [' + p.input.join(', ') + ']<br>' + p.layers.length + ' layers</p>';
      html += '</div>';
    }
    html += '</div></div>';
    html += '<div id="preset-chain"></div>';
    container.innerHTML = html;
  }

  function loadPreset(name) {
    var preset = architecturePresets[name];
    if (!preset) return;
    state.chainLayers = [];
    for (var i = 0; i < preset.layers.length; i++) {
      var l = preset.layers[i];
      var params = {};
      var keys = Object.keys(l.params);
      for (var j = 0; j < keys.length; j++) params[keys[j]] = l.params[keys[j]];
      state.chainLayers.push({ type: l.type, params: params });
    }
    switchTab('chain');
    var bEl = document.getElementById('chain-b');
    var cEl = document.getElementById('chain-c');
    var hEl = document.getElementById('chain-h');
    var wEl = document.getElementById('chain-w');
    if (bEl) bEl.value = preset.input[0];
    if (cEl) cEl.value = preset.input[1];
    if (hEl) hEl.value = preset.input[2];
    if (wEl) wEl.value = preset.input.length > 3 ? preset.input[3] : '';
    renderChain();
  }

  /* ===== TAB SWITCHING ===== */

  function switchTab(tab) {
    state.activeTab = tab;
    var btns = document.querySelectorAll('.tab-btn');
    var panes = document.querySelectorAll('.tab-pane');
    for (var i = 0; i < btns.length && i < 20; i++) {
      btns[i].classList.toggle('active', btns[i].dataset.tab === tab);
    }
    for (var j = 0; j < panes.length && j < 20; j++) {
      panes[j].classList.toggle('active', panes[j].id === tab + '-content');
    }
    if (tab === 'chain') renderChain();
  }

  function toggleFramework() {
    state.framework = state.framework === 'pytorch' ? 'tensorflow' : 'pytorch';
    var toggle = document.getElementById('fw-toggle');
    if (toggle) toggle.classList.toggle('active', state.framework === 'tensorflow');
    updateFrameworkLabels();
  }

  function updateFrameworkLabels() {
    var sel = document.getElementById('layer-select');
    if (!sel) return;
    var opts = sel.options;
    var types = Object.keys(layerCalcs);
    for (var i = 0; i < opts.length && i < 50; i++) {
      if (state.framework === 'tensorflow' && tfNames[types[i]]) {
        opts[i].textContent = tfNames[types[i]] + ' (' + types[i] + ')';
      } else {
        opts[i].textContent = types[i];
      }
    }
  }

  /* ===== FAQ ===== */

  function initFAQ() {
    var items = document.querySelectorAll('.faq-item');
    for (var i = 0; i < items.length && i < 50; i++) {
      items[i].querySelector('.faq-q').addEventListener('click', (function(item) {
        return function() { item.classList.toggle('open'); };
      })(items[i]));
    }
  }

  /* ===== NAV ===== */

  function initNav() {
    var toggle = document.querySelector('.mobile-toggle');
    var nav = document.querySelector('nav');
    if (toggle && nav) {
      toggle.addEventListener('click', function() { nav.classList.toggle('open'); });
    }
  }

  /* ===== INIT ===== */

  function init() {
    initNav();
    buildSingleLayerUI();
    buildChainUI();
    buildErrorUI();
    buildPresetsUI();
    initFAQ();
    var tabBtns = document.querySelectorAll('.tab-btn');
    for (var i = 0; i < tabBtns.length && i < 20; i++) {
      tabBtns[i].addEventListener('click', (function(btn) {
        return function() { switchTab(btn.dataset.tab); };
      })(tabBtns[i]));
    }
  }

  /* Expose public API */
  window.HT = {
    calcSingle: calcSingle,
    parseError: parseError,
    addChainLayer: addChainLayer,
    removeChainLayer: removeChainLayer,
    clearChain: clearChain,
    loadPreset: loadPreset,
    onLayerChange: onLayerChange,
    toggleFramework: toggleFramework,
    switchTab: switchTab
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();


// === Zovo V5 Pro Nudge System ===
(function() {
  var V5_LIMIT = 4;
  var V5_FEATURE = 'Advanced layer chains';
  var v5Count = 0;
  var v5Shown = false;

  function v5ShowNudge() {
    if (v5Shown || sessionStorage.getItem('v5_pro_nudge')) return;
    v5Shown = true;
    sessionStorage.setItem('v5_pro_nudge', '1');
    var host = location.hostname;
    var el = document.createElement('div');
    el.className = 'pro-nudge';
    el.innerHTML = '<div class="pro-nudge-inner">' +
      '<span class="pro-nudge-icon">\u2726</span>' +
      '<div class="pro-nudge-text">' +
      '<strong>' + V5_FEATURE + '</strong> is a Pro feature. ' +
      '<a href="https://zovo.one/pricing?utm_source=' + host +
      '&utm_medium=satellite&utm_campaign=pro-nudge" target="_blank">' +
      'Get Zovo Lifetime \u2014 $99 once, access everything forever.</a>' +
      '</div></div>';
    var target = document.querySelector('main') ||
      document.querySelector('.tool-section') ||
      document.querySelector('.container') ||
      document.querySelector('section') ||
      document.body;
    if (target) target.appendChild(el);
  }

  // Track meaningful user actions (button clicks, form submits)
  document.addEventListener('click', function(e) {
    var t = e.target;
    if (t.closest('button, [onclick], .btn, input[type="submit"], input[type="button"]')) {
      v5Count++;
      if (v5Count >= V5_LIMIT) v5ShowNudge();
    }
  }, true);

  // Track file drops/selections (for file-based tools)
  document.addEventListener('change', function(e) {
    if (e.target && e.target.type === 'file') {
      v5Count++;
      if (v5Count >= V5_LIMIT) v5ShowNudge();
    }
  }, true);
})();
