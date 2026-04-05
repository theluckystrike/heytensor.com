/* heytensor.com — Tensor Shape Calculator */
'use strict';

var OPS = {
  Conv2d: { params: ['out_channels','kernel_size','stride','padding'], defaults: [64,3,1,1] },
  Linear: { params: ['out_features'], defaults: [256] },
  MaxPool2d: { params: ['kernel_size','stride','padding'], defaults: [2,2,0] },
  BatchNorm: { params: [], defaults: [] },
  Flatten: { params: ['start_dim'], defaults: [1] },
  Reshape: { params: ['shape'], defaults: ['0,-1'] },
  Transpose: { params: ['dim0','dim1'], defaults: [1,2] },
  Concatenate: { params: ['dim','other_size'], defaults: [1,64] }
};

var PRESETS = {
  'Simple CNN': [
    { op:'Conv2d', vals:[32,3,1,1] },
    { op:'BatchNorm', vals:[] },
    { op:'MaxPool2d', vals:[2,2,0] },
    { op:'Conv2d', vals:[64,3,1,1] },
    { op:'BatchNorm', vals:[] },
    { op:'MaxPool2d', vals:[2,2,0] },
    { op:'Flatten', vals:[1] },
    { op:'Linear', vals:[128] },
    { op:'Linear', vals:[10] }
  ],
  'ResNet Block': [
    { op:'Conv2d', vals:[64,3,1,1] },
    { op:'BatchNorm', vals:[] },
    { op:'Conv2d', vals:[64,3,1,1] },
    { op:'BatchNorm', vals:[] }
  ],
  'Transformer Layer': [
    { op:'Linear', vals:[512] },
    { op:'Reshape', vals:['0,8,64'] },
    { op:'Transpose', vals:[1,2] },
    { op:'Reshape', vals:['0,-1'] },
    { op:'Linear', vals:[512] }
  ]
};

var layers = [];
var inputShape = [1, 3, 32, 32]; // batch, channels, H, W

function init() {
  bindPresets();
  bindAddLayer();
  bindInputShape();
  recalculate();
}

function bindPresets() {
  var btns = document.querySelectorAll('.preset-btn');
  for (var i = 0; i < btns.length && i < 20; i++) {
    btns[i].addEventListener('click', function() {
      var name = this.getAttribute('data-preset');
      loadPreset(name);
    });
  }
}

function loadPreset(name) {
  var preset = PRESETS[name];
  if (!preset) return;
  layers = [];
  for (var i = 0; i < preset.length && i < 50; i++) {
    layers.push({ op: preset[i].op, vals: preset[i].vals.slice() });
  }
  // Set default input shapes per preset
  if (name === 'Transformer Layer') {
    inputShape = [1, 32, 512]; // batch, seq_len, d_model
  } else {
    inputShape = [1, 3, 32, 32];
  }
  updateInputFields();
  recalculate();
}

function updateInputFields() {
  var container = document.getElementById('input-dims');
  if (!container) return;
  var html = '';
  for (var i = 0; i < inputShape.length && i < 10; i++) {
    var label = i === 0 ? 'Batch' : 'Dim ' + i;
    if (inputShape.length === 4) {
      label = ['Batch','C','H','W'][i];
    } else if (inputShape.length === 3) {
      label = ['Batch','Seq','Features'][i];
    }
    html += '<div class="shape-field"><label>' + label + '</label>';
    html += '<input type="number" data-dim="' + i + '" value="' + inputShape[i] + '" min="1"></div>';
  }
  html += '<button class="btn btn-sm btn-outline" id="add-dim-btn">+ Dim</button>';
  html += '<button class="btn btn-sm btn-outline" id="rm-dim-btn">- Dim</button>';
  container.innerHTML = html;
}

function bindInputShape() {
  updateInputFields();
  document.getElementById('input-dims').addEventListener('input', function(e) {
    if (e.target.tagName === 'INPUT') {
      var dim = parseInt(e.target.getAttribute('data-dim'), 10);
      inputShape[dim] = parseInt(e.target.value, 10) || 1;
      recalculate();
    }
  });
  document.getElementById('input-dims').addEventListener('click', function(e) {
    if (e.target.id === 'add-dim-btn' && inputShape.length < 8) {
      inputShape.push(1);
      updateInputFields();
      recalculate();
    }
    if (e.target.id === 'rm-dim-btn' && inputShape.length > 2) {
      inputShape.pop();
      updateInputFields();
      recalculate();
    }
  });
}

function bindAddLayer() {
  var btn = document.getElementById('add-layer-btn');
  var sel = document.getElementById('op-select');
  if (btn && sel) {
    btn.addEventListener('click', function() {
      var op = sel.value;
      var defs = OPS[op].defaults.slice();
      layers.push({ op: op, vals: defs });
      recalculate();
    });
  }
}

function removeLayer(idx) {
  layers.splice(idx, 1);
  recalculate();
}

function moveLayer(idx, dir) {
  var newIdx = idx + dir;
  if (newIdx < 0 || newIdx >= layers.length) return;
  var temp = layers[idx];
  layers[idx] = layers[newIdx];
  layers[newIdx] = temp;
  recalculate();
}

function computeShape(shape, layer) {
  var s = shape.slice();
  var op = layer.op;
  var v = layer.vals;

  if (op === 'Conv2d') {
    if (s.length < 3) return { shape: null, error: 'Conv2d requires at least 3D input (batch, C, H, W)', formula: '' };
    var outC = v[0] || 64;
    var k = v[1] || 3;
    var stride = v[2] || 1;
    var pad = v[3] || 0;
    var H = s[s.length - 2];
    var W = s[s.length - 1];
    var newH = Math.floor((H + 2 * pad - k) / stride) + 1;
    var newW = Math.floor((W + 2 * pad - k) / stride) + 1;
    if (newH <= 0 || newW <= 0) return { shape: null, error: 'Output dimensions <= 0. Check kernel/stride/padding.', formula: '' };
    var out = s.slice(0, -3).concat([outC, newH, newW]);
    var formula = 'H_out = floor((' + H + ' + 2*' + pad + ' - ' + k + ') / ' + stride + ') + 1 = ' + newH;
    return { shape: out, error: null, formula: formula };
  }

  if (op === 'Linear') {
    if (s.length < 2) return { shape: null, error: 'Linear requires at least 2D input', formula: '' };
    var outF = v[0] || 256;
    var out = s.slice(0, -1).concat([outF]);
    var formula = 'last_dim: ' + s[s.length - 1] + ' -> ' + outF;
    return { shape: out, error: null, formula: formula };
  }

  if (op === 'MaxPool2d') {
    if (s.length < 3) return { shape: null, error: 'MaxPool2d requires at least 3D input', formula: '' };
    var k = v[0] || 2;
    var stride = v[1] || k;
    var pad = v[2] || 0;
    var H = s[s.length - 2];
    var W = s[s.length - 1];
    var newH = Math.floor((H + 2 * pad - k) / stride) + 1;
    var newW = Math.floor((W + 2 * pad - k) / stride) + 1;
    if (newH <= 0 || newW <= 0) return { shape: null, error: 'Output dimensions <= 0.', formula: '' };
    var out = s.slice(0, -2).concat([newH, newW]);
    var formula = 'H_out = floor((' + H + ' + 2*' + pad + ' - ' + k + ') / ' + stride + ') + 1 = ' + newH;
    return { shape: out, error: null, formula: formula };
  }

  if (op === 'BatchNorm') {
    return { shape: s.slice(), error: null, formula: 'shape unchanged (normalizes over batch)' };
  }

  if (op === 'Flatten') {
    var startDim = v[0] !== undefined ? v[0] : 1;
    if (startDim < 0) startDim = s.length + startDim;
    if (startDim >= s.length) return { shape: null, error: 'start_dim out of range', formula: '' };
    var flat = 1;
    for (var i = startDim; i < s.length && i < 20; i++) flat *= s[i];
    var out = s.slice(0, startDim).concat([flat]);
    var formula = 'flatten dims [' + startDim + ':] -> ' + flat;
    return { shape: out, error: null, formula: formula };
  }

  if (op === 'Reshape') {
    var shapeStr = (v[0] !== undefined ? v[0] : '0,-1') + '';
    var parts = shapeStr.split(',');
    var totalIn = 1;
    for (var i = 0; i < s.length && i < 20; i++) totalIn *= s[i];
    var newShape = [];
    var negIdx = -1;
    var totalOut = 1;
    for (var i = 0; i < parts.length && i < 20; i++) {
      var d = parseInt(parts[i].trim(), 10);
      if (d === 0 && i < s.length) d = s[i];
      if (d === -1) { negIdx = i; newShape.push(-1); }
      else { newShape.push(d); totalOut *= d; }
    }
    if (negIdx >= 0) {
      newShape[negIdx] = totalIn / totalOut;
      if (newShape[negIdx] !== Math.floor(newShape[negIdx]) || newShape[negIdx] <= 0) {
        return { shape: null, error: 'Cannot reshape ' + totalIn + ' elements into shape [' + shapeStr + ']', formula: '' };
      }
    }
    var formula = 'reshape [' + s.join(',') + '] -> [' + newShape.join(',') + ']';
    return { shape: newShape, error: null, formula: formula };
  }

  if (op === 'Transpose') {
    var d0 = v[0] !== undefined ? v[0] : 0;
    var d1 = v[1] !== undefined ? v[1] : 1;
    if (d0 >= s.length || d1 >= s.length || d0 < 0 || d1 < 0) {
      return { shape: null, error: 'Transpose dims out of range for ' + s.length + 'D tensor', formula: '' };
    }
    var out = s.slice();
    var tmp = out[d0];
    out[d0] = out[d1];
    out[d1] = tmp;
    var formula = 'swap dim ' + d0 + ' <-> dim ' + d1;
    return { shape: out, error: null, formula: formula };
  }

  if (op === 'Concatenate') {
    var dim = v[0] !== undefined ? v[0] : 1;
    var otherSize = v[1] !== undefined ? v[1] : 64;
    if (dim >= s.length || dim < 0) {
      return { shape: null, error: 'Concat dim out of range', formula: '' };
    }
    var out = s.slice();
    out[dim] = out[dim] + otherSize;
    var formula = 'dim ' + dim + ': ' + s[dim] + ' + ' + otherSize + ' = ' + out[dim];
    return { shape: out, error: null, formula: formula };
  }

  return { shape: s.slice(), error: 'Unknown operation', formula: '' };
}

function recalculate() {
  var chain = document.getElementById('layer-chain');
  if (!chain) return;
  var html = '';
  var currentShape = inputShape.slice();
  var hasError = false;

  for (var i = 0; i < layers.length && i < 100; i++) {
    var layer = layers[i];
    var result = computeShape(currentShape, layer);
    var isError = result.error !== null;
    if (isError) hasError = true;

    html += '<div class="shape-arrow">&#8595;</div>';
    html += '<div class="layer-card' + (isError ? ' error' : '') + '">';
    html += '<div class="layer-header">';
    html += '<span class="layer-name">' + esc(layer.op) + '</span>';
    html += '<div class="layer-params">';

    var paramNames = OPS[layer.op].params;
    for (var j = 0; j < paramNames.length && j < 10; j++) {
      var val = layer.vals[j] !== undefined ? layer.vals[j] : OPS[layer.op].defaults[j];
      html += '<div class="layer-param">';
      html += '<label>' + esc(paramNames[j]) + '</label>';
      html += '<input type="' + (layer.op === 'Reshape' && paramNames[j] === 'shape' ? 'text' : 'number') + '" ';
      html += 'data-layer="' + i + '" data-param="' + j + '" value="' + esc(val + '') + '">';
      html += '</div>';
    }
    html += '</div>';
    html += '<div class="layer-actions">';
    html += '<button class="btn btn-sm btn-outline" data-move="' + i + '" data-dir="-1" title="Move up">&#8593;</button>';
    html += '<button class="btn btn-sm btn-outline" data-move="' + i + '" data-dir="1" title="Move down">&#8595;</button>';
    html += '<button class="btn btn-sm btn-danger" data-remove="' + i + '" title="Remove">&#10005;</button>';
    html += '</div></div>';

    html += '<div class="shape-result">';
    if (isError) {
      html += '<span class="shape-val">Error</span>';
      html += '<span class="shape-formula">' + esc(result.error) + '</span>';
    } else {
      html += '<span class="shape-val">[' + result.shape.join(', ') + ']</span>';
      html += '<span class="shape-formula">' + esc(result.formula) + '</span>';
    }
    html += '</div></div>';

    if (!isError) {
      currentShape = result.shape;
    } else {
      // Try to continue with previous shape for subsequent layers
    }
  }

  chain.innerHTML = html;

  // Final output
  var finalEl = document.getElementById('final-output');
  if (finalEl) {
    if (layers.length === 0) {
      finalEl.className = 'final-output';
      finalEl.innerHTML = '<h3>Output Shape</h3><div class="final-shape">[' + inputShape.join(', ') + ']</div>';
    } else if (hasError) {
      finalEl.className = 'final-output error';
      finalEl.innerHTML = '<h3>Output Shape</h3><div class="final-shape">Shape Mismatch Detected</div>';
    } else {
      finalEl.className = 'final-output';
      finalEl.innerHTML = '<h3>Output Shape</h3><div class="final-shape">[' + currentShape.join(', ') + ']</div>';
    }
  }

  bindLayerEvents();
}

function bindLayerEvents() {
  // Param changes
  var inputs = document.querySelectorAll('[data-layer]');
  for (var i = 0; i < inputs.length && i < 500; i++) {
    inputs[i].addEventListener('input', function() {
      var li = parseInt(this.getAttribute('data-layer'), 10);
      var pi = parseInt(this.getAttribute('data-param'), 10);
      var op = layers[li].op;
      if (op === 'Reshape' && OPS[op].params[pi] === 'shape') {
        layers[li].vals[pi] = this.value;
      } else {
        layers[li].vals[pi] = parseInt(this.value, 10) || 0;
      }
      recalculate();
    });
  }
  // Remove buttons
  var rmBtns = document.querySelectorAll('[data-remove]');
  for (var i = 0; i < rmBtns.length && i < 100; i++) {
    rmBtns[i].addEventListener('click', function() {
      removeLayer(parseInt(this.getAttribute('data-remove'), 10));
    });
  }
  // Move buttons
  var moveBtns = document.querySelectorAll('[data-move]');
  for (var i = 0; i < moveBtns.length && i < 200; i++) {
    moveBtns[i].addEventListener('click', function() {
      var idx = parseInt(this.getAttribute('data-move'), 10);
      var dir = parseInt(this.getAttribute('data-dir'), 10);
      moveLayer(idx, dir);
    });
  }
}

function esc(s) {
  if (s === null || s === undefined) return '';
  var d = document.createElement('div');
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}

document.addEventListener('DOMContentLoaded', init);
