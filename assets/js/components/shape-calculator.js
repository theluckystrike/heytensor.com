/**
 * Shape Calculator Component
 * Calculates output shapes for Conv2d, Conv1d, ConvTranspose2d, Linear,
 * LSTM, MaxPool2d, AvgPool2d, BatchNorm2d, Flatten, Embedding, MultiheadAttention.
 * Includes Chain Mode for stacking layers.
 */
(function () {
  'use strict';

  window.HeyTensor = window.HeyTensor || {};
  window.HeyTensor.components = window.HeyTensor.components || {};

  /* ── Layer Definitions ── */
  const LAYERS = {
    Conv2d: {
      label: 'Conv2d',
      inputDesc: '[B, C_in, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'c_in', label: 'C_in', default: 3 },
        { key: 'h_in', label: 'H_in', default: 224 },
        { key: 'w_in', label: 'W_in', default: 224 },
        { key: 'out_channels', label: 'out_channels', default: 64 },
        { key: 'kernel_size', label: 'kernel_size', default: 3 },
        { key: 'stride', label: 'stride', default: 1 },
        { key: 'padding', label: 'padding', default: 1 },
        { key: 'dilation', label: 'dilation', default: 1 }
      ],
      calc: function (p) {
        var hOut = Math.floor((p.h_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / p.stride) + 1;
        var wOut = Math.floor((p.w_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / p.stride) + 1;
        if (hOut <= 0 || wOut <= 0) return { error: 'Output dimensions are zero or negative. Increase padding or decrease kernel/dilation.' };
        return {
          shape: [p.batch, p.out_channels, hOut, wOut],
          formula: 'H_out = floor((' + p.h_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + p.stride + ') + 1 = ' + hOut +
                   '\nW_out = floor((' + p.w_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + p.stride + ') + 1 = ' + wOut +
                   '\nOutput: [' + p.batch + ', ' + p.out_channels + ', ' + hOut + ', ' + wOut + ']'
        };
      }
    },
    Conv1d: {
      label: 'Conv1d',
      inputDesc: '[B, C_in, L]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'c_in', label: 'C_in', default: 1 },
        { key: 'l_in', label: 'L_in', default: 1000 },
        { key: 'out_channels', label: 'out_channels', default: 32 },
        { key: 'kernel_size', label: 'kernel_size', default: 5 },
        { key: 'stride', label: 'stride', default: 1 },
        { key: 'padding', label: 'padding', default: 0 },
        { key: 'dilation', label: 'dilation', default: 1 }
      ],
      calc: function (p) {
        var lOut = Math.floor((p.l_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / p.stride) + 1;
        if (lOut <= 0) return { error: 'Output length is zero or negative.' };
        return {
          shape: [p.batch, p.out_channels, lOut],
          formula: 'L_out = floor((' + p.l_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + p.stride + ') + 1 = ' + lOut +
                   '\nOutput: [' + p.batch + ', ' + p.out_channels + ', ' + lOut + ']'
        };
      }
    },
    ConvTranspose2d: {
      label: 'ConvTranspose2d',
      inputDesc: '[B, C_in, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'c_in', label: 'C_in', default: 64 },
        { key: 'h_in', label: 'H_in', default: 7 },
        { key: 'w_in', label: 'W_in', default: 7 },
        { key: 'out_channels', label: 'out_channels', default: 32 },
        { key: 'kernel_size', label: 'kernel_size', default: 3 },
        { key: 'stride', label: 'stride', default: 2 },
        { key: 'padding', label: 'padding', default: 1 },
        { key: 'output_padding', label: 'output_padding', default: 1 },
        { key: 'dilation', label: 'dilation', default: 1 }
      ],
      calc: function (p) {
        var hOut = (p.h_in - 1) * p.stride - 2 * p.padding + p.dilation * (p.kernel_size - 1) + p.output_padding + 1;
        var wOut = (p.w_in - 1) * p.stride - 2 * p.padding + p.dilation * (p.kernel_size - 1) + p.output_padding + 1;
        if (hOut <= 0 || wOut <= 0) return { error: 'Output dimensions are zero or negative.' };
        return {
          shape: [p.batch, p.out_channels, hOut, wOut],
          formula: 'H_out = (' + p.h_in + '-1)*' + p.stride + ' - 2*' + p.padding + ' + ' + p.dilation + '*(' + p.kernel_size + '-1) + ' + p.output_padding + ' + 1 = ' + hOut +
                   '\nW_out = (' + p.w_in + '-1)*' + p.stride + ' - 2*' + p.padding + ' + ' + p.dilation + '*(' + p.kernel_size + '-1) + ' + p.output_padding + ' + 1 = ' + wOut +
                   '\nOutput: [' + p.batch + ', ' + p.out_channels + ', ' + hOut + ', ' + wOut + ']'
        };
      }
    },
    Linear: {
      label: 'Linear',
      inputDesc: '[B, *, in_features]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'in_features', label: 'in_features', default: 512 },
        { key: 'out_features', label: 'out_features', default: 10 }
      ],
      calc: function (p) {
        return {
          shape: [p.batch, p.out_features],
          formula: 'Output: [' + p.batch + ', ' + p.out_features + ']\nLinear transforms the last dimension: in_features(' + p.in_features + ') -> out_features(' + p.out_features + ')\nParameters: ' + p.in_features + ' * ' + p.out_features + ' + ' + p.out_features + ' = ' + (p.in_features * p.out_features + p.out_features)
        };
      }
    },
    LSTM: {
      label: 'LSTM',
      inputDesc: '[B, seq_len, input_size]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'seq_len', label: 'seq_len', default: 100 },
        { key: 'input_size', label: 'input_size', default: 128 },
        { key: 'hidden_size', label: 'hidden_size', default: 256 },
        { key: 'num_layers', label: 'num_layers', default: 1 },
        { key: 'bidirectional', label: 'bidirectional (0/1)', default: 0 }
      ],
      calc: function (p) {
        var numDir = p.bidirectional ? 2 : 1;
        var outSize = p.hidden_size * numDir;
        return {
          shape: [p.batch, p.seq_len, outSize],
          formula: 'num_directions = ' + numDir +
                   '\nOutput: [' + p.batch + ', ' + p.seq_len + ', ' + p.hidden_size + ' * ' + numDir + '] = [' + p.batch + ', ' + p.seq_len + ', ' + outSize + ']' +
                   '\nh_n: [' + (p.num_layers * numDir) + ', ' + p.batch + ', ' + p.hidden_size + ']' +
                   '\nc_n: [' + (p.num_layers * numDir) + ', ' + p.batch + ', ' + p.hidden_size + ']'
        };
      }
    },
    MaxPool2d: {
      label: 'MaxPool2d',
      inputDesc: '[B, C, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'channels', label: 'Channels (C)', default: 64 },
        { key: 'h_in', label: 'H_in', default: 224 },
        { key: 'w_in', label: 'W_in', default: 224 },
        { key: 'kernel_size', label: 'kernel_size', default: 2 },
        { key: 'stride', label: 'stride (0=kernel)', default: 0 },
        { key: 'padding', label: 'padding', default: 0 },
        { key: 'dilation', label: 'dilation', default: 1 }
      ],
      calc: function (p) {
        var stride = p.stride === 0 ? p.kernel_size : p.stride;
        var hOut = Math.floor((p.h_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / stride) + 1;
        var wOut = Math.floor((p.w_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / stride) + 1;
        if (hOut <= 0 || wOut <= 0) return { error: 'Output dimensions are zero or negative.' };
        return {
          shape: [p.batch, p.channels, hOut, wOut],
          formula: 'stride = ' + stride + (p.stride === 0 ? ' (defaults to kernel_size)' : '') +
                   '\nH_out = floor((' + p.h_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + stride + ') + 1 = ' + hOut +
                   '\nW_out = floor((' + p.w_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + stride + ') + 1 = ' + wOut +
                   '\nOutput: [' + p.batch + ', ' + p.channels + ', ' + hOut + ', ' + wOut + ']'
        };
      }
    },
    AvgPool2d: {
      label: 'AvgPool2d',
      inputDesc: '[B, C, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'channels', label: 'Channels (C)', default: 64 },
        { key: 'h_in', label: 'H_in', default: 224 },
        { key: 'w_in', label: 'W_in', default: 224 },
        { key: 'kernel_size', label: 'kernel_size', default: 2 },
        { key: 'stride', label: 'stride (0=kernel)', default: 0 },
        { key: 'padding', label: 'padding', default: 0 }
      ],
      calc: function (p) {
        var stride = p.stride === 0 ? p.kernel_size : p.stride;
        var hOut = Math.floor((p.h_in + 2 * p.padding - p.kernel_size) / stride) + 1;
        var wOut = Math.floor((p.w_in + 2 * p.padding - p.kernel_size) / stride) + 1;
        if (hOut <= 0 || wOut <= 0) return { error: 'Output dimensions are zero or negative.' };
        return {
          shape: [p.batch, p.channels, hOut, wOut],
          formula: 'stride = ' + stride + (p.stride === 0 ? ' (defaults to kernel_size)' : '') +
                   '\nH_out = floor((' + p.h_in + ' + 2*' + p.padding + ' - ' + p.kernel_size + ') / ' + stride + ') + 1 = ' + hOut +
                   '\nW_out = floor((' + p.w_in + ' + 2*' + p.padding + ' - ' + p.kernel_size + ') / ' + stride + ') + 1 = ' + wOut +
                   '\nOutput: [' + p.batch + ', ' + p.channels + ', ' + hOut + ', ' + wOut + ']'
        };
      }
    },
    BatchNorm2d: {
      label: 'BatchNorm2d',
      inputDesc: '[B, C, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'channels', label: 'Channels (C)', default: 64 },
        { key: 'h_in', label: 'H', default: 32 },
        { key: 'w_in', label: 'W', default: 32 },
        { key: 'num_features', label: 'num_features', default: 64 }
      ],
      calc: function (p) {
        if (p.num_features !== p.channels) {
          return { error: 'num_features (' + p.num_features + ') must match number of channels (' + p.channels + '). Set num_features = ' + p.channels + '.' };
        }
        return {
          shape: [p.batch, p.channels, p.h_in, p.w_in],
          formula: 'BatchNorm2d does not change the shape.\nnum_features (' + p.num_features + ') matches channels (' + p.channels + ') ✓\nOutput: [' + p.batch + ', ' + p.channels + ', ' + p.h_in + ', ' + p.w_in + ']'
        };
      }
    },
    Flatten: {
      label: 'Flatten',
      inputDesc: '[B, C, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'c', label: 'C', default: 64 },
        { key: 'h', label: 'H', default: 7 },
        { key: 'w', label: 'W', default: 7 },
        { key: 'start_dim', label: 'start_dim', default: 1 }
      ],
      calc: function (p) {
        var dims = [p.batch, p.c, p.h, p.w];
        var startDim = Math.max(0, Math.min(p.start_dim, dims.length - 1));
        var flatPart = 1;
        for (var i = startDim; i < dims.length; i++) flatPart *= dims[i];
        var outDims = dims.slice(0, startDim);
        outDims.push(flatPart);
        return {
          shape: outDims,
          formula: 'Input: [' + dims.join(', ') + ']' +
                   '\nFlatten from dim ' + startDim + ': product of dims [' + dims.slice(startDim).join(', ') + '] = ' + flatPart +
                   '\nOutput: [' + outDims.join(', ') + ']'
        };
      }
    },
    Embedding: {
      label: 'Embedding',
      inputDesc: '[B, seq_len] (integer indices)',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'seq_len', label: 'seq_len', default: 128 },
        { key: 'num_embeddings', label: 'num_embeddings (vocab)', default: 30000 },
        { key: 'embedding_dim', label: 'embedding_dim', default: 768 }
      ],
      calc: function (p) {
        return {
          shape: [p.batch, p.seq_len, p.embedding_dim],
          formula: 'Input: [' + p.batch + ', ' + p.seq_len + '] (integer indices in range [0, ' + (p.num_embeddings - 1) + '])' +
                   '\nEmbedding lookup: each index -> ' + p.embedding_dim + '-dim vector' +
                   '\nOutput: [' + p.batch + ', ' + p.seq_len + ', ' + p.embedding_dim + ']' +
                   '\nParameters: ' + p.num_embeddings + ' * ' + p.embedding_dim + ' = ' + (p.num_embeddings * p.embedding_dim).toLocaleString()
        };
      }
    },
    MultiheadAttention: {
      label: 'MultiheadAttention',
      inputDesc: '[B, seq_len, embed_dim]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'seq_len', label: 'seq_len', default: 128 },
        { key: 'embed_dim', label: 'embed_dim', default: 512 },
        { key: 'num_heads', label: 'num_heads', default: 8 }
      ],
      calc: function (p) {
        if (p.embed_dim % p.num_heads !== 0) {
          return { error: 'embed_dim (' + p.embed_dim + ') must be divisible by num_heads (' + p.num_heads + '). ' + p.embed_dim + ' % ' + p.num_heads + ' = ' + (p.embed_dim % p.num_heads) + '.' };
        }
        var headDim = p.embed_dim / p.num_heads;
        return {
          shape: [p.batch, p.seq_len, p.embed_dim],
          formula: 'head_dim = embed_dim / num_heads = ' + p.embed_dim + ' / ' + p.num_heads + ' = ' + headDim +
                   '\nOutput (attn_output): [' + p.batch + ', ' + p.seq_len + ', ' + p.embed_dim + ']' +
                   '\nAttention weights: [' + p.batch + ', ' + p.num_heads + ', ' + p.seq_len + ', ' + p.seq_len + ']' +
                   '\nParameters: 3*' + p.embed_dim + '*' + p.embed_dim + ' + ' + p.embed_dim + '*' + p.embed_dim + ' + biases = ~' + (4 * p.embed_dim * p.embed_dim + 4 * p.embed_dim).toLocaleString()
        };
      }
    },
    GRU: {
      label: 'GRU',
      inputDesc: '[B, seq_len, input_size]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'seq_len', label: 'seq_len', default: 100 },
        { key: 'input_size', label: 'input_size', default: 128 },
        { key: 'hidden_size', label: 'hidden_size', default: 256 },
        { key: 'num_layers', label: 'num_layers', default: 1 },
        { key: 'bidirectional', label: 'bidirectional (0/1)', default: 0 }
      ],
      calc: function (p) {
        var numDir = p.bidirectional ? 2 : 1;
        var outSize = p.hidden_size * numDir;
        return {
          shape: [p.batch, p.seq_len, outSize],
          formula: 'num_directions = ' + numDir +
                   '\nOutput: [' + p.batch + ', ' + p.seq_len + ', ' + p.hidden_size + ' * ' + numDir + '] = [' + p.batch + ', ' + p.seq_len + ', ' + outSize + ']' +
                   '\nh_n: [' + (p.num_layers * numDir) + ', ' + p.batch + ', ' + p.hidden_size + ']'
        };
      }
    },
    LayerNorm: {
      label: 'LayerNorm',
      inputDesc: '[B, *, normalized_shape]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'seq_len', label: 'seq_len', default: 128 },
        { key: 'features', label: 'features', default: 512 },
        { key: 'normalized_shape', label: 'normalized_shape', default: 512 }
      ],
      calc: function (p) {
        return {
          shape: [p.batch, p.seq_len, p.features],
          formula: 'LayerNorm does not change the shape.\nnormalized_shape = ' + p.normalized_shape +
                   '\nOutput: [' + p.batch + ', ' + p.seq_len + ', ' + p.features + ']' +
                   '\nParameters: 2 * ' + p.normalized_shape + ' = ' + (2 * p.normalized_shape) + ' (gamma + beta)'
        };
      }
    },
    Conv3d: {
      label: 'Conv3d',
      inputDesc: '[B, C_in, D, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'c_in', label: 'C_in', default: 3 },
        { key: 'd_in', label: 'D_in', default: 16 },
        { key: 'h_in', label: 'H_in', default: 112 },
        { key: 'w_in', label: 'W_in', default: 112 },
        { key: 'out_channels', label: 'out_channels', default: 64 },
        { key: 'kernel_size', label: 'kernel_size', default: 3 },
        { key: 'stride', label: 'stride', default: 1 },
        { key: 'padding', label: 'padding', default: 1 },
        { key: 'dilation', label: 'dilation', default: 1 }
      ],
      calc: function (p) {
        var dOut = Math.floor((p.d_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / p.stride) + 1;
        var hOut = Math.floor((p.h_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / p.stride) + 1;
        var wOut = Math.floor((p.w_in + 2 * p.padding - p.dilation * (p.kernel_size - 1) - 1) / p.stride) + 1;
        if (dOut <= 0 || hOut <= 0 || wOut <= 0) return { error: 'Output dimensions are zero or negative. Increase padding or decrease kernel/dilation.' };
        return {
          shape: [p.batch, p.out_channels, dOut, hOut, wOut],
          formula: 'D_out = floor((' + p.d_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + p.stride + ') + 1 = ' + dOut +
                   '\nH_out = floor((' + p.h_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + p.stride + ') + 1 = ' + hOut +
                   '\nW_out = floor((' + p.w_in + ' + 2*' + p.padding + ' - ' + p.dilation + '*(' + p.kernel_size + '-1) - 1) / ' + p.stride + ') + 1 = ' + wOut +
                   '\nOutput: [' + p.batch + ', ' + p.out_channels + ', ' + dOut + ', ' + hOut + ', ' + wOut + ']'
        };
      }
    },
    AdaptiveAvgPool2d: {
      label: 'AdaptiveAvgPool2d',
      inputDesc: '[B, C, H, W]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 1 },
        { key: 'channels', label: 'Channels (C)', default: 512 },
        { key: 'h_in', label: 'H_in', default: 7 },
        { key: 'w_in', label: 'W_in', default: 7 },
        { key: 'output_h', label: 'output_size H', default: 1 },
        { key: 'output_w', label: 'output_size W', default: 1 }
      ],
      calc: function (p) {
        if (p.output_h <= 0 || p.output_w <= 0) return { error: 'Output size must be positive.' };
        return {
          shape: [p.batch, p.channels, p.output_h, p.output_w],
          formula: 'AdaptiveAvgPool2d adapts kernel/stride to produce the target output size.' +
                   '\nInput: [' + p.batch + ', ' + p.channels + ', ' + p.h_in + ', ' + p.w_in + ']' +
                   '\nOutput: [' + p.batch + ', ' + p.channels + ', ' + p.output_h + ', ' + p.output_w + ']'
        };
      }
    },
    Reshape: {
      label: 'Reshape / View',
      inputDesc: '[B, *]',
      params: [
        { key: 'batch', label: 'Batch (B)', default: 32 },
        { key: 'c', label: 'dim 1', default: 64 },
        { key: 'h', label: 'dim 2', default: 7 },
        { key: 'w', label: 'dim 3', default: 7 },
        { key: 'new_d1', label: 'new dim 1', default: 32 },
        { key: 'new_d2', label: 'new dim 2 (0=auto)', default: 0 }
      ],
      calc: function (p) {
        var totalIn = p.c * p.h * p.w;
        var d1 = p.new_d1;
        var d2 = p.new_d2;
        if (d1 <= 0 && d2 <= 0) return { error: 'At least one target dimension must be positive.' };
        if (d2 === 0 && d1 > 0) {
          if (totalIn % d1 !== 0) return { error: 'Cannot reshape: ' + totalIn + ' elements not divisible by ' + d1 + '.' };
          d2 = totalIn / d1;
        } else if (d1 === 0 && d2 > 0) {
          if (totalIn % d2 !== 0) return { error: 'Cannot reshape: ' + totalIn + ' elements not divisible by ' + d2 + '.' };
          d1 = totalIn / d2;
        } else {
          if (d1 * d2 !== totalIn) return { error: 'Cannot reshape: ' + d1 + ' * ' + d2 + ' = ' + (d1 * d2) + ' != ' + totalIn + ' elements.' };
        }
        return {
          shape: [p.batch, d1, d2],
          formula: 'Input elements (excl. batch): ' + p.c + ' * ' + p.h + ' * ' + p.w + ' = ' + totalIn +
                   '\nOutput: [' + p.batch + ', ' + d1 + ', ' + d2 + ']' +
                   '\nTotal elements preserved: ' + totalIn
        };
      }
    }
  };

  /* ── Helper: create element ── */
  function el(tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'className') e.className = attrs[k];
      else if (k === 'textContent') e.textContent = attrs[k];
      else if (k === 'innerHTML') e.innerHTML = attrs[k];
      else if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    });
    if (children) children.forEach(function (c) {
      if (typeof c === 'string') e.appendChild(document.createTextNode(c));
      else if (c) e.appendChild(c);
    });
    return e;
  }

  /* ── Single Layer UI ── */
  function buildSingleLayerUI(container, defaultLayer) {
    var card = el('div', { className: 'card shape-calc-container' });

    // Layer selector
    var layerKeys = Object.keys(LAYERS);
    var selectWrap = el('div', { className: 'layer-select-wrap' });
    var selectLabel = el('label', { textContent: 'Layer Type' });
    var select = el('select');
    layerKeys.forEach(function (key) {
      var opt = el('option', { value: key, textContent: LAYERS[key].label });
      if (key === defaultLayer) opt.selected = true;
      select.appendChild(opt);
    });
    selectWrap.appendChild(selectLabel);
    selectWrap.appendChild(select);
    card.appendChild(selectWrap);

    // Params area
    var paramsDiv = el('div', { className: 'params-grid' });
    card.appendChild(paramsDiv);

    // Calculate button
    var calcBtn = el('button', { className: 'btn btn-primary', textContent: 'Calculate Output Shape' });
    card.appendChild(calcBtn);

    // Result area
    var resultDiv = el('div', { className: 'result-box', style: 'display:none' });
    card.appendChild(resultDiv);

    // State
    var currentInputs = {};

    function renderParams() {
      paramsDiv.innerHTML = '';
      currentInputs = {};
      var layerKey = select.value;
      var layer = LAYERS[layerKey];
      layer.params.forEach(function (param) {
        var field = el('div', { className: 'input-field' });
        var lbl = el('label', { textContent: param.label });
        var inp = el('input', { type: 'number', value: param.default, id: 'p-' + param.key });
        field.appendChild(lbl);
        field.appendChild(inp);
        paramsDiv.appendChild(field);
        currentInputs[param.key] = inp;
        inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') calculate(); });
      });
    }

    function calculate() {
      var layerKey = select.value;
      var layer = LAYERS[layerKey];
      var p = {};
      Object.keys(currentInputs).forEach(function (key) {
        p[key] = parseInt(currentInputs[key].value, 10) || 0;
      });
      var result = layer.calc(p);
      resultDiv.style.display = 'block';
      if (result.error) {
        resultDiv.className = 'result-box error';
        resultDiv.innerHTML = '<div class="shape">Error</div><div class="formula">' + escHtml(result.error) + '</div>';
      } else {
        resultDiv.className = 'result-box success';
        resultDiv.innerHTML = '<div class="shape">[' + result.shape.join(', ') + ']</div><div class="formula">' + escHtml(result.formula).replace(/\n/g, '<br>') + '</div>';
      }
    }

    select.addEventListener('change', function () { renderParams(); resultDiv.style.display = 'none'; });
    calcBtn.addEventListener('click', calculate);

    renderParams();
    container.appendChild(card);

    // Auto-calculate on load
    calculate();
  }

  /* ── Chain Mode UI ── */
  function buildChainUI(container, defaultLayer) {
    var card = el('div', { className: 'card' });
    var h3 = el('h3', { textContent: 'Chain Mode — Stack Layers Sequentially' });
    card.appendChild(h3);

    var desc = el('p', { className: 'plot-info', textContent: 'Add layers one by one. Each layer takes the previous output as its input. The shape is traced automatically.' });
    card.appendChild(desc);

    // Input shape
    var inputLabel = el('label', { textContent: 'Starting Input Shape (comma-separated, e.g. 1,3,224,224)' });
    var inputField = el('input', { type: 'text', value: '1,3,224,224', id: 'chain-input-shape' });
    card.appendChild(inputLabel);
    card.appendChild(inputField);
    card.appendChild(el('br'));

    // Chain layers list
    var chainList = el('div', { className: 'chain-output', id: 'chain-list' });
    card.appendChild(chainList);

    // Add layer controls
    var controls = el('div', { className: 'chain-controls', style: 'margin-top:12px' });
    var addSelect = el('select', { id: 'chain-add-select' });
    Object.keys(LAYERS).forEach(function (key) {
      addSelect.appendChild(el('option', { value: key, textContent: LAYERS[key].label }));
    });
    var addBtn = el('button', { className: 'btn btn-secondary btn-sm', textContent: '+ Add Layer' });
    var clearBtn = el('button', { className: 'btn btn-secondary btn-sm', textContent: 'Clear All' });
    controls.appendChild(addSelect);
    controls.appendChild(addBtn);
    controls.appendChild(clearBtn);
    card.appendChild(controls);

    // Chain result
    var chainResult = el('div', { className: 'result-box', style: 'display:none', id: 'chain-result' });
    card.appendChild(chainResult);

    container.appendChild(card);

    // State
    var chainLayers = [];

    function addChainLayer(layerKey) {
      var layer = LAYERS[layerKey];
      var params = {};
      layer.params.forEach(function (p) { params[p.key] = p.default; });
      chainLayers.push({ type: layerKey, params: params });
      recalcChain();
    }

    function recalcChain() {
      chainList.innerHTML = '';
      var shapeParts = inputField.value.split(',').map(function (s) { return parseInt(s.trim(), 10) || 0; });
      var currentShape = shapeParts;

      // Show input
      var inputStep = el('div', { className: 'chain-step' });
      inputStep.innerHTML = '<span class="step-num">IN</span><span class="step-layer">Input</span><span class="step-shape">[' + currentShape.join(', ') + ']</span>';
      chainList.appendChild(inputStep);

      chainLayers.forEach(function (cl, idx) {
        var layerDef = LAYERS[cl.type];
        // Auto-set input params from current shape
        var p = Object.assign({}, cl.params);
        autoSetInput(cl.type, p, currentShape);

        var result = layerDef.calc(p);
        var step = el('div', { className: 'chain-step' + (result.error ? ' error' : '') });
        var removeBtn = '<button class="remove-layer-btn" data-idx="' + idx + '" title="Remove">&#10005;</button>';
        step.innerHTML = '<span class="step-num">' + (idx + 1) + '</span>' +
          '<span class="step-layer">' + layerDef.label + '</span>' +
          '<span class="step-shape' + (result.error ? ' err' : '') + '">' +
          (result.error ? 'Error' : '[' + result.shape.join(', ') + ']') + '</span>' + removeBtn;
        chainList.appendChild(step);

        if (!result.error) currentShape = result.shape;
      });

      // Remove layer buttons
      chainList.querySelectorAll('.remove-layer-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
          chainLayers.splice(parseInt(this.dataset.idx), 1);
          recalcChain();
        });
      });

      // Final result
      if (chainLayers.length > 0) {
        chainResult.style.display = 'block';
        chainResult.className = 'result-box success';
        chainResult.innerHTML = '<div class="shape">Final Output: [' + currentShape.join(', ') + ']</div>';
      } else {
        chainResult.style.display = 'none';
      }
    }

    function autoSetInput(type, p, shape) {
      // Automatically map current shape to layer input params
      switch (type) {
        case 'Conv2d':
          if (shape.length >= 4) { p.batch = shape[0]; p.c_in = shape[1]; p.h_in = shape[2]; p.w_in = shape[3]; }
          break;
        case 'Conv1d':
          if (shape.length >= 3) { p.batch = shape[0]; p.c_in = shape[1]; p.l_in = shape[2]; }
          break;
        case 'ConvTranspose2d':
          if (shape.length >= 4) { p.batch = shape[0]; p.c_in = shape[1]; p.h_in = shape[2]; p.w_in = shape[3]; }
          break;
        case 'Linear':
          if (shape.length >= 2) { p.batch = shape[0]; p.in_features = shape[shape.length - 1]; }
          break;
        case 'LSTM':
          if (shape.length >= 3) { p.batch = shape[0]; p.seq_len = shape[1]; p.input_size = shape[2]; }
          break;
        case 'MaxPool2d':
        case 'AvgPool2d':
          if (shape.length >= 4) { p.batch = shape[0]; p.channels = shape[1]; p.h_in = shape[2]; p.w_in = shape[3]; }
          break;
        case 'BatchNorm2d':
          if (shape.length >= 4) { p.batch = shape[0]; p.channels = shape[1]; p.h_in = shape[2]; p.w_in = shape[3]; p.num_features = shape[1]; }
          break;
        case 'Flatten':
          if (shape.length >= 4) { p.batch = shape[0]; p.c = shape[1]; p.h = shape[2]; p.w = shape[3]; }
          else if (shape.length === 3) { p.batch = shape[0]; p.c = shape[1]; p.h = shape[2]; p.w = 1; }
          break;
        case 'Embedding':
          if (shape.length >= 2) { p.batch = shape[0]; p.seq_len = shape[1]; }
          break;
        case 'MultiheadAttention':
          if (shape.length >= 3) { p.batch = shape[0]; p.seq_len = shape[1]; p.embed_dim = shape[2]; }
          break;
        case 'GRU':
          if (shape.length >= 3) { p.batch = shape[0]; p.seq_len = shape[1]; p.input_size = shape[2]; }
          break;
        case 'LayerNorm':
          if (shape.length >= 3) { p.batch = shape[0]; p.seq_len = shape[1]; p.features = shape[2]; p.normalized_shape = shape[2]; }
          break;
        case 'Conv3d':
          if (shape.length >= 5) { p.batch = shape[0]; p.c_in = shape[1]; p.d_in = shape[2]; p.h_in = shape[3]; p.w_in = shape[4]; }
          break;
        case 'AdaptiveAvgPool2d':
          if (shape.length >= 4) { p.batch = shape[0]; p.channels = shape[1]; p.h_in = shape[2]; p.w_in = shape[3]; }
          break;
        case 'Reshape':
          if (shape.length >= 4) { p.batch = shape[0]; p.c = shape[1]; p.h = shape[2]; p.w = shape[3]; }
          else if (shape.length === 3) { p.batch = shape[0]; p.c = shape[1]; p.h = shape[2]; p.w = 1; }
          break;
      }
    }

    addBtn.addEventListener('click', function () { addChainLayer(addSelect.value); });
    clearBtn.addEventListener('click', function () { chainLayers = []; recalcChain(); });
    inputField.addEventListener('change', recalcChain);

    // Pre-add default layer
    addChainLayer(defaultLayer);
  }

  /* ── Escape HTML ── */
  function escHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /* ── Init ── */
  window.HeyTensor.components['shape-calculator'] = {
    init: function (container, config) {
      var defaultLayer = config.layer || 'Conv2d';

      // Tab bar for single vs chain
      var tabBar = el('div', { className: 'tab-bar' });
      var singleBtn = el('button', { className: 'tab-btn active', textContent: 'Single Layer', 'data-tab': 'single' });
      var chainBtn = el('button', { className: 'tab-btn', textContent: 'Chain Mode', 'data-tab': 'chain' });
      tabBar.appendChild(singleBtn);
      tabBar.appendChild(chainBtn);
      container.appendChild(tabBar);

      var singlePane = el('div', { className: 'tab-pane active', id: 'single-pane' });
      var chainPane = el('div', { className: 'tab-pane', id: 'chain-pane' });
      container.appendChild(singlePane);
      container.appendChild(chainPane);

      buildSingleLayerUI(singlePane, defaultLayer);
      buildChainUI(chainPane, defaultLayer);

      // Tab switching
      [singleBtn, chainBtn].forEach(function (btn) {
        btn.addEventListener('click', function () {
          singleBtn.classList.toggle('active', btn === singleBtn);
          chainBtn.classList.toggle('active', btn === chainBtn);
          singlePane.classList.toggle('active', btn === singleBtn);
          chainPane.classList.toggle('active', btn === chainBtn);
        });
      });
    }
  };
})();
