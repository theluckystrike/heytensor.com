/* Dense forward-operation estimates. One MAC = two FLOPs; bias counted separately. */
(function (root) {
  'use strict';
  function integer(value, name, zero) {
    var n = Number(value);
    if (value === '' || value === null || typeof value === 'boolean' || !Number.isSafeInteger(n) || n < (zero ? 0 : 1) || n > 1000000)
      throw new Error(name + ' must be a whole number from ' + (zero ? '0' : '1') + ' to 1000000.');
    return n;
  }
  function estimate(mode, values) {
    var n = {}, keys = mode === 'linear' ? ['vectors', 'in_features', 'out_features'] :
      ['batch', 'in_channels', 'out_channels', 'height', 'width', 'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'padding_h', 'padding_w', 'dilation_h', 'dilation_w', 'groups'];
    if (mode !== 'linear' && mode !== 'conv2d') throw new Error('Choose Linear or Conv2d.');
    keys.forEach(function (key) { n[key] = integer(values[key], key, key.indexOf('padding') === 0); });
    var output, macs;
    if (mode === 'linear') {
      output = [n.vectors, n.out_features];
      macs = BigInt(n.vectors) * BigInt(n.in_features) * BigInt(n.out_features);
    } else {
      if (n.in_channels % n.groups || n.out_channels % n.groups) throw new Error('Both input and output channels must be divisible by groups.');
      var h = Math.floor((n.height + 2 * n.padding_h - n.dilation_h * (n.kernel_h - 1) - 1) / n.stride_h + 1);
      var w = Math.floor((n.width + 2 * n.padding_w - n.dilation_w * (n.kernel_w - 1) - 1) / n.stride_w + 1);
      if (h <= 0 || w <= 0) throw new Error('Effective kernel exceeds the padded input. Output dimensions must be positive.');
      output = [n.batch, n.out_channels, h, w];
      macs = output.reduce(function (a, b) { return a * BigInt(b); }, 1n) * BigInt(n.in_channels / n.groups) * BigInt(n.kernel_h) * BigInt(n.kernel_w);
    }
    var core = 2n * macs;
    var bias = values.bias === true ? output.reduce(function (a, b) { return a * BigInt(b); }, 1n) : 0n;
    return { output: output, macs: macs.toString(), core_flops: core.toString(), bias_flops: bias.toString(), total_flops: (core + bias).toString() };
  }
  if (typeof module !== 'undefined' && module.exports) { module.exports = { estimate: estimate }; return; }
  root.HeyTensor = root.HeyTensor || {}; root.HeyTensor.components = root.HeyTensor.components || {};
  root.HeyTensor.components['flops-calculator'] = { init: function (container) {
    container.innerHTML = '<div class="card"><h2>Estimate one forward operation</h2><p class="plot-info">One multiply-accumulate (MAC) counts as two FLOPs. Dense real-valued arithmetic estimate; no training/backward, activation, normalization, memory transfer or hardware timing estimate.</p><form id="flops-form"><label for="flops-mode">Operation</label><select id="flops-mode"><option value="linear">Linear</option><option value="conv2d">Conv2d</option></select><div class="params-grid" id="flops-fields"></div><label><input id="flops-bias" type="checkbox"> Include separate bias additions</label><p><button class="btn btn-primary" type="submit">Calculate FLOPs</button></p></form><p id="flops-error" role="alert" hidden></p><div id="flops-result" class="model-summary" role="status" aria-live="polite"></div></div>';
    var fields = container.querySelector('#flops-fields'), mode = container.querySelector('#flops-mode');
    var result = container.querySelector('#flops-result'), error = container.querySelector('#flops-error');
    var labels = {vectors:'Input vectors (batch × sequence positions)',in_features:'Input features',out_features:'Output features',batch:'Batch size',in_channels:'Input channels',out_channels:'Output channels',height:'Input height',width:'Input width',kernel_h:'Kernel height',kernel_w:'Kernel width',stride_h:'Stride height',stride_w:'Stride width',padding_h:'Padding height (each side)',padding_w:'Padding width (each side)',dilation_h:'Dilation height',dilation_w:'Dilation width',groups:'Groups'};
    function renderFields() {
      var defaults = mode.value === 'linear' ? {vectors:1,in_features:512,out_features:256} : {batch:1,in_channels:3,out_channels:64,height:224,width:224,kernel_h:3,kernel_w:3,stride_h:1,stride_w:1,padding_h:1,padding_w:1,dilation_h:1,dilation_w:1,groups:1};
      fields.innerHTML = '';
      Object.keys(defaults).forEach(function (key) {
        var label=document.createElement('label');label.textContent=labels[key];label.htmlFor='flops-'+key;
        var input=document.createElement('input');input.id='flops-'+key;input.name=key;input.type='number';input.min=key.indexOf('padding')===0?'0':'1';input.max='1000000';input.step='1';input.required=true;input.value=defaults[key];label.appendChild(input);fields.appendChild(label);
      });
      calculate();
    }
    function calculate() {
      var values={bias:container.querySelector('#flops-bias').checked};
      fields.querySelectorAll('input').forEach(function (input) { values[input.name]=input.value; });
      try {
        var data=estimate(mode.value,values);error.hidden=true;error.textContent='';result.hidden=false;
        function row(label, value) { return '<div class="summary-row"><span>'+label+'</span><span class="val">'+BigInt(value).toLocaleString('en-US')+'</span></div>'; }
        result.innerHTML='<h3>Forward operation estimate</h3><p>Output shape: <code>['+data.output.join(', ')+']</code></p>'+row('MACs',data.macs)+row('Core FLOPs (2 × MACs)',data.core_flops)+row('Separate bias FLOPs',data.bias_flops)+row('Total estimated FLOPs',data.total_flops)+'<p class="plot-info">'+(mode.value==='linear'?'MACs = input vectors × input features × output features.':'MACs = batch × output channels × output height × output width × (input channels / groups) × kernel height × kernel width. Symmetric numeric zero padding per axis; output size uses stride and dilation.')+'</p><p class="plot-info">Counts dense operations including padded positions. Optimized kernels may execute differently. Bias adds one operation per output element when enabled. This is not a whole-model profiler.</p>';
      } catch (e) { result.hidden=true;result.innerHTML='';error.hidden=false;error.textContent=e.message; }
    }
    mode.addEventListener('change',renderFields);
    container.querySelector('#flops-form').addEventListener('submit',function(e){e.preventDefault();calculate();});
    container.querySelector('#flops-form').addEventListener('input',function(){result.hidden=true;result.innerHTML='';error.hidden=true;});
    renderFields();
  }};
})(typeof window !== 'undefined' ? window : globalThis);
