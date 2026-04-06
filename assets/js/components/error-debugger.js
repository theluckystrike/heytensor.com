/**
 * Error Debugger Component
 * Parses PyTorch error messages and provides explanations + fixes.
 */
(function () {
  'use strict';

  window.HeyTensor = window.HeyTensor || {};
  window.HeyTensor.components = window.HeyTensor.components || {};

  /* ── Error Patterns ── */
  var PATTERNS = [
    {
      id: 'mat1-mat2',
      regex: /mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)/i,
      title: 'Matrix Multiplication Shape Mismatch',
      explain: function (m) {
        return 'PyTorch tried to multiply two matrices: mat1 has shape (' + m[1] + ' x ' + m[2] + ') and mat2 has shape (' + m[3] + ' x ' + m[4] + '). For matrix multiplication to work, the number of columns in mat1 (' + m[2] + ') must equal the number of rows in mat2 (' + m[3] + ').';
      },
      cause: function (m) {
        return 'This typically happens in a nn.Linear layer where in_features=' + m[3] + ' but the actual input has ' + m[2] + ' features. The input tensor has batch_size=' + m[1] + ' with ' + m[2] + ' features, but the Linear layer weight matrix expects ' + m[3] + ' input features.';
      },
      fix: function (m) {
        return 'Change your Linear layer from nn.Linear(' + m[3] + ', ' + m[4] + ') to nn.Linear(' + m[2] + ', ' + m[4] + '). If this is after a Flatten layer, the flattened size is ' + m[2] + '. Alternatively, use model Chain Mode to trace shapes through your network and find where the mismatch occurs.';
      }
    },
    {
      id: 'shape-invalid-for-input',
      regex: /shape '?\[?([\d, ]+)\]?'? is invalid for input of size (\d+)/i,
      title: 'Invalid Reshape / View',
      explain: function (m) {
        var targetShape = m[1].trim();
        var inputSize = parseInt(m[2]);
        var targetParts = targetShape.split(/[,\s]+/).map(Number).filter(function (n) { return !isNaN(n); });
        var targetProduct = targetParts.reduce(function (a, b) { return a * b; }, 1);
        return 'You tried to reshape a tensor of total size ' + inputSize + ' into shape [' + targetShape + '], which has a total size of ' + targetProduct + '. These must be equal.';
      },
      cause: function (m) {
        return 'The total number of elements in the new shape must match the original tensor. This often happens when you change a layer upstream (like modifying Conv2d parameters) but forget to update the reshape/view operation.';
      },
      fix: function (m) {
        var inputSize = parseInt(m[2]);
        return 'Make sure your target shape multiplies to exactly ' + inputSize + '. Use tensor.view(-1) to flatten completely, or tensor.view(batch_size, -1) to flatten all but the batch dimension. The -1 lets PyTorch infer that dimension automatically.';
      }
    },
    {
      id: 'expected-channels',
      regex: /expected input\[?\d*\]? to have (\d+) channels?, but got (\d+) channels?/i,
      title: 'Channel Count Mismatch',
      explain: function (m) {
        return 'A layer expected ' + m[1] + ' input channels, but received a tensor with ' + m[2] + ' channels.';
      },
      cause: function (m) {
        return 'This happens when a Conv2d or BatchNorm layer\'s in_channels/num_features doesn\'t match the actual number of channels from the previous layer. For example, if the previous Conv2d outputs ' + m[2] + ' channels but the next layer expects ' + m[1] + '.';
      },
      fix: function (m) {
        return 'Change the layer\'s input channel parameter from ' + m[1] + ' to ' + m[2] + '. For Conv2d, set in_channels=' + m[2] + '. For BatchNorm2d, set num_features=' + m[2] + '. Or fix the upstream layer to output ' + m[1] + ' channels.';
      }
    },
    {
      id: 'cuda-oom',
      regex: /CUDA out of memory\. Tried to allocate ([\d.]+ [KMGT]?i?B)/i,
      title: 'CUDA Out of Memory',
      explain: function (m) {
        return 'Your GPU ran out of VRAM. PyTorch tried to allocate ' + m[1] + ' but the GPU does not have enough free memory.';
      },
      cause: function (m) {
        return 'Training requires memory for: model parameters, gradients, optimizer states (2x params for Adam), and intermediate activations (scales with batch size). The activation memory usually dominates during training.';
      },
      fix: function (m) {
        return 'Try these solutions in order:\n1. Reduce batch size (halving batch size roughly halves activation memory)\n2. Use mixed precision: with torch.cuda.amp.autocast():\n3. Enable gradient checkpointing: model.gradient_checkpointing_enable()\n4. Use gradient accumulation: accumulate gradients over N small batches\n5. Clear cache: torch.cuda.empty_cache() before training\n6. Use a smaller model or fewer layers';
      }
    },
    {
      id: 'cuda-oom-generic',
      regex: /CUDA out of memory/i,
      title: 'CUDA Out of Memory',
      explain: function () {
        return 'Your GPU ran out of VRAM during a tensor operation.';
      },
      cause: function () {
        return 'Training requires memory for: model parameters, gradients, optimizer states (2x params for Adam), and intermediate activations. The total can be 4-20x the raw model size.';
      },
      fix: function () {
        return 'Try these solutions in order:\n1. Reduce batch size (halving batch size roughly halves activation memory)\n2. Use mixed precision: with torch.cuda.amp.autocast():\n3. Enable gradient checkpointing: model.gradient_checkpointing_enable()\n4. Use gradient accumulation to simulate larger batches\n5. Call torch.cuda.empty_cache() before training\n6. Monitor memory with torch.cuda.memory_summary()';
      }
    },
    {
      id: 'view-not-compatible',
      regex: /view size is not compatible with input tensor/i,
      title: 'View / Contiguous Memory Error',
      explain: function () {
        return 'tensor.view() requires the tensor to be stored contiguously in memory. Your tensor is not contiguous, likely because of a previous transpose, permute, or indexing operation.';
      },
      cause: function () {
        return 'Operations like .transpose(), .permute(), .expand(), .narrow(), and advanced indexing create non-contiguous views of the underlying data. view() cannot work with these because it requires a single contiguous block of memory.';
      },
      fix: function () {
        return 'Replace tensor.view(shape) with one of:\n1. tensor.contiguous().view(shape) — creates a contiguous copy first\n2. tensor.reshape(shape) — handles non-contiguous tensors automatically\n\nreshape() is generally safer. Use view() only when you want to enforce contiguity as a correctness check.';
      }
    },
    {
      id: 'size-mismatch',
      regex: /size mismatch.* (\[[\d, ]+\]).* (\[[\d, ]+\])/i,
      title: 'Tensor Size Mismatch',
      explain: function (m) {
        return 'Two tensors have incompatible sizes: ' + m[1] + ' and ' + m[2] + '. The operation requires these dimensions to match or be broadcastable.';
      },
      cause: function () {
        return 'This happens during element-wise operations (add, multiply), concatenation, or loss computation when tensors have different shapes. Check that your model output and target tensors have matching dimensions.';
      },
      fix: function (m) {
        return 'Check both tensor shapes: ' + m[1] + ' vs ' + m[2] + '.\n1. For loss functions: ensure model output and target have compatible shapes\n2. For element-wise ops: tensors must be broadcastable (matching dims or one dim is 1)\n3. For concatenation: all dims except the concat dim must match\n4. Print shapes with tensor.shape to trace the mismatch to its source.';
      }
    },
    {
      id: 'size-mismatch-simple',
      regex: /size mismatch/i,
      title: 'Tensor Size Mismatch',
      explain: function () {
        return 'Two tensors in an operation have incompatible shapes.';
      },
      cause: function () {
        return 'This usually occurs during element-wise operations, loss computation, or when combining tensors. The dimensions must either match exactly or be broadcastable.';
      },
      fix: function () {
        return 'Add print(tensor.shape) before the failing line for each tensor involved. Common fixes:\n1. Verify model output shape matches target shape for loss computation\n2. Check that concatenation dimensions match (all dims except cat dim must be equal)\n3. Use .unsqueeze() or .squeeze() to align dimensions for broadcasting\n4. Use the Shape Calculator to trace shapes through your model.';
      }
    },
    {
      id: 'expected-batch-size',
      regex: /expected input batch_size \((\d+)\) to match target batch_size \((\d+)\)/i,
      title: 'Batch Size Mismatch',
      explain: function (m) {
        return 'The model output has batch size ' + m[1] + ' but the target/labels have batch size ' + m[2] + '.';
      },
      cause: function () {
        return 'Your input data and labels have different batch sizes, or a layer accidentally modified the batch dimension.';
      },
      fix: function (m) {
        return 'Ensure your DataLoader returns matching input and target batch sizes. Check that no layer modifies dimension 0 (the batch dimension). If using .view() or .reshape(), always use -1 or the actual batch size for the first dimension, not a hardcoded value.';
      }
    },
    {
      id: 'runtime-generic',
      regex: /RuntimeError/i,
      title: 'PyTorch RuntimeError',
      explain: function () {
        return 'A RuntimeError occurred during tensor computation. This is the most common category of PyTorch errors and usually relates to shape mismatches, device mismatches, or invalid operations.';
      },
      cause: function () {
        return 'Common causes: tensor shape incompatibility, mixing CPU and CUDA tensors, invalid layer parameters, or operations on tensors of wrong dtype.';
      },
      fix: function () {
        return 'Debug steps:\n1. Print tensor.shape, tensor.device, and tensor.dtype before the failing line\n2. Use the Chain Mode shape calculator to trace shapes through your model\n3. Check that all tensors are on the same device (.to(device))\n4. Verify input shapes match layer expectations';
      }
    }
  ];

  /* ── Preset hints ── */
  var PRESET_HINTS = {
    'shape-mismatch': 'Paste any PyTorch shape-related error message below. Common errors include dimension mismatches, incompatible tensor sizes, and broadcasting failures.',
    'mat1-mat2': 'Paste the full "mat1 and mat2 shapes cannot be multiplied" error message below. Example: RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)',
    'cuda-oom': 'Paste your CUDA out of memory error below, or use the quick tips displayed. Example: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB...',
    'view-size': 'Paste the "view size is not compatible" error below. This tool will explain why it happened and how to fix it.'
  };

  var PRESET_EXAMPLES = {
    'mat1-mat2': 'RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x512 and 256x10)',
    'cuda-oom': 'RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 11.17 GiB total capacity; 8.87 GiB already allocated; 1.48 GiB free; 9.58 GiB reserved)',
    'view-size': 'RuntimeError: view size is not compatible with input tensor\'s size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.',
    'shape-mismatch': 'RuntimeError: The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 0'
  };

  function escHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  /* ── Init ── */
  window.HeyTensor.components['error-debugger'] = {
    init: function (container, config) {
      var preset = config.preset || 'shape-mismatch';
      var hint = PRESET_HINTS[preset] || PRESET_HINTS['shape-mismatch'];
      var example = PRESET_EXAMPLES[preset] || '';

      var card = document.createElement('div');
      card.className = 'card';

      var h3 = document.createElement('h3');
      h3.textContent = 'Paste Your Error Message';
      card.appendChild(h3);

      var hintP = document.createElement('p');
      hintP.className = 'plot-info';
      hintP.textContent = hint;
      card.appendChild(hintP);

      var inputWrap = document.createElement('div');
      inputWrap.className = 'error-input-wrap';
      var textarea = document.createElement('textarea');
      textarea.placeholder = 'Paste your PyTorch error message here...\n\nExample: ' + example;
      textarea.value = '';
      inputWrap.appendChild(textarea);
      card.appendChild(inputWrap);

      var btnRow = document.createElement('div');
      btnRow.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap';

      var analyzeBtn = document.createElement('button');
      analyzeBtn.className = 'btn btn-primary';
      analyzeBtn.textContent = 'Analyze Error';
      btnRow.appendChild(analyzeBtn);

      var exampleBtn = document.createElement('button');
      exampleBtn.className = 'btn btn-secondary btn-sm';
      exampleBtn.textContent = 'Load Example';
      btnRow.appendChild(exampleBtn);

      card.appendChild(btnRow);

      var resultDiv = document.createElement('div');
      resultDiv.className = 'error-result';
      resultDiv.style.display = 'none';
      card.appendChild(resultDiv);

      container.appendChild(card);

      // Quick tips for CUDA OOM
      if (preset === 'cuda-oom') {
        var tipsCard = document.createElement('div');
        tipsCard.className = 'card';
        tipsCard.innerHTML = '<h3>Quick Solutions for CUDA Out of Memory</h3>' +
          '<div class="ref-table-wrap"><table class="ref-table">' +
          '<tr><th>Solution</th><th>Memory Savings</th><th>Code Change</th></tr>' +
          '<tr><td>Reduce batch size</td><td>~Linear reduction</td><td><code>batch_size = batch_size // 2</code></td></tr>' +
          '<tr><td>Mixed precision (AMP)</td><td>~40% less</td><td><code>with torch.cuda.amp.autocast():</code></td></tr>' +
          '<tr><td>Gradient checkpointing</td><td>~60% less activations</td><td><code>model.gradient_checkpointing_enable()</code></td></tr>' +
          '<tr><td>Gradient accumulation</td><td>Same as smaller batch</td><td><code>loss.backward(); if step % N == 0: optimizer.step()</code></td></tr>' +
          '<tr><td>torch.no_grad() for eval</td><td>~50% less</td><td><code>with torch.no_grad():</code></td></tr>' +
          '<tr><td>Clear cache</td><td>Frees fragmented memory</td><td><code>torch.cuda.empty_cache()</code></td></tr>' +
          '</table></div>';
        container.appendChild(tipsCard);
      }

      function analyze() {
        var text = textarea.value.trim();
        if (!text) {
          resultDiv.style.display = 'none';
          return;
        }

        var matched = false;
        for (var i = 0; i < PATTERNS.length; i++) {
          var pat = PATTERNS[i];
          var m = text.match(pat.regex);
          if (m) {
            resultDiv.style.display = 'block';
            resultDiv.innerHTML =
              '<div class="error-type">' + escHtml(pat.title) + '</div>' +
              '<div class="error-explain">' + escHtml(pat.explain(m)).replace(/\n/g, '<br>') + '</div>' +
              '<div class="error-cause"><strong>Likely cause:</strong> ' + escHtml(pat.cause(m)).replace(/\n/g, '<br>') + '</div>' +
              '<div class="error-fix"><strong>Fix:</strong> ' + escHtml(pat.fix(m)).replace(/\n/g, '<br>') + '</div>';
            matched = true;
            break;
          }
        }

        if (!matched) {
          resultDiv.style.display = 'block';
          resultDiv.innerHTML =
            '<div class="error-type">Unrecognized Error Pattern</div>' +
            '<div class="error-explain">This error pattern is not in our database yet. Here are general debugging steps:</div>' +
            '<div class="error-fix"><strong>Debug steps:</strong><br>' +
            '1. Print tensor.shape before the failing line<br>' +
            '2. Check tensor.device (CPU vs CUDA mismatch?)<br>' +
            '3. Check tensor.dtype (float32 vs int64 mismatch?)<br>' +
            '4. Use the Shape Calculator to trace shapes through your model<br>' +
            '5. Search the error message on the PyTorch forums</div>';
        }
      }

      analyzeBtn.addEventListener('click', analyze);
      textarea.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
      });
      exampleBtn.addEventListener('click', function () {
        textarea.value = example;
        analyze();
      });
    }
  };
})();
