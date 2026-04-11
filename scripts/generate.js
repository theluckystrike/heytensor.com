#!/usr/bin/env node
/**
 * HeyTensor Programmatic Answer Page Generator (V31)
 * Generates Conv2d shape pages, parameter count pages, and error fix pages.
 * No npm dependencies required.
 */

const fs = require('fs');
const path = require('path');

const ANSWERS_DIR = path.join(__dirname, '..', 'answers');
const SITEMAP_PATH = path.join(__dirname, '..', 'sitemap.xml');
const TODAY = '2026-04-11';

// ============================================================
// EXISTING V30 FILES — never overwrite these
// ============================================================
const EXISTING_FILES = new Set(
  fs.readdirSync(ANSWERS_DIR).map(f => f)
);

function fileExists(filename) {
  return EXISTING_FILES.has(filename);
}

// ============================================================
// Conv2d output formula
// ============================================================
function conv2dOutput(input, kernel, stride, padding) {
  return Math.floor((input + 2 * padding - kernel) / stride) + 1;
}

// ============================================================
// Architecture annotations
// ============================================================
function getArchNote(input, kernel, stride, padding) {
  if (input === 224 && kernel === 7 && stride === 2 && padding === 3)
    return 'This is the first convolution layer in <strong>ResNet</strong> (conv1). It reduces the 224&times;224 ImageNet input to 112&times;112 before max-pooling.';
  if (input === 224 && kernel === 11 && stride === 4 && padding === 2)
    return 'This is the first convolution layer in <strong>AlexNet</strong>. The large 11&times;11 kernel with stride 4 aggressively reduces spatial dimensions from 224&times;224 to 55&times;55.';
  if (kernel === 3 && stride === 1 && padding === 1)
    return 'This is the standard &ldquo;same&rdquo; convolution preserving spatial dimensions. Used extensively in <strong>VGG</strong>, <strong>ResNet</strong>, and <strong>DenseNet</strong> architectures.';
  if (kernel === 3 && stride === 2 && padding === 1)
    return 'This is a strided convolution that halves spatial dimensions. Modern architectures like <strong>ResNet</strong> and <strong>ConvNeXt</strong> use this instead of max-pooling for downsampling.';
  if (kernel === 5 && stride === 1 && padding === 2)
    return 'A 5&times;5 kernel with padding=2 preserves spatial dimensions (same convolution). Used in early layers of <strong>Inception/GoogLeNet</strong> modules.';
  if (kernel === 5 && stride === 2 && padding === 2)
    return 'A strided 5&times;5 convolution used in some <strong>GAN generators and discriminators</strong>, as well as certain <strong>Inception</strong> variants for spatial reduction.';
  if (kernel === 7 && stride === 2 && padding === 3)
    return 'A 7&times;7 strided convolution that halves spatial dimensions. This is the classic <strong>ResNet conv1</strong> configuration for processing large input images.';
  if (input === 7 && kernel === 7 && stride === 1 && padding === 0)
    return 'This reduces a 7&times;7 feature map to 1&times;1 &mdash; equivalent to <strong>global average pooling</strong>. Found at the end of <strong>ResNet</strong> before the fully connected layer.';
  if (input === 56 && kernel === 3 && stride === 2 && padding === 1)
    return 'Downsampling from 56&times;56 to 28&times;28. This is the transition between <strong>ResNet layer1 and layer2</strong> (or similar stage boundaries in EfficientNet).';
  if (input === 28 && kernel === 3 && stride === 2 && padding === 1)
    return 'Downsampling from 28&times;28 to 14&times;14. This is the transition between <strong>ResNet layer2 and layer3</strong>.';
  if (input === 14 && kernel === 3 && stride === 2 && padding === 1)
    return 'Downsampling from 14&times;14 to 7&times;7. This is the transition between <strong>ResNet layer3 and layer4</strong>.';
  if (input === 112 && kernel === 3 && stride === 2 && padding === 1)
    return 'Downsampling from 112&times;112 to 56&times;56. In <strong>ResNet</strong>, this is typically done via max-pooling after conv1, but strided convolutions achieve the same result without information loss.';
  if (kernel === 3 && stride === 1 && padding === 0)
    return 'A 3&times;3 convolution without padding reduces spatial size by 2 in each dimension. This &ldquo;valid&rdquo; convolution is common in early architectures like <strong>LeNet-5</strong>.';
  if (kernel === 5 && stride === 1 && padding === 0)
    return 'A 5&times;5 valid convolution that reduces spatial size by 4 in each dimension. Used in the original <strong>LeNet-5</strong> architecture for handwritten digit recognition.';
  return 'This configuration is used in various custom CNN architectures for feature extraction.';
}

function getRelatedConv2dLinks(input, kernel, stride, padding, allConfigs) {
  const currentFile = makeConv2dFilename(input, kernel, stride, padding);
  const candidates = [];

  // Add related pages from all configs + existing V30 pages
  const fixedLinks = [
    { file: 'conv2d-output-224x224-kernel-3.html', label: 'Conv2d 224&times;224 Kernel 3' },
    { file: 'conv2d-output-224x224-kernel-7-stride-2.html', label: 'Conv2d 224&times;224 Kernel 7 Stride 2' },
    { file: 'conv2d-output-32x32-kernel-5.html', label: 'Conv2d 32&times;32 Kernel 5' },
    { file: 'what-is-stride-in-conv2d.html', label: 'What is Stride in Conv2d?' },
    { file: 'what-is-padding-same-pytorch.html', label: 'What is padding="same" in PyTorch?' },
    { file: 'maxpool2d-output-size.html', label: 'MaxPool2d Output Size' },
  ];

  // Same input different kernels
  for (const c of allConfigs) {
    if (c.input === input && (c.kernel !== kernel || c.stride !== stride || c.padding !== padding)) {
      const fn = makeConv2dFilename(c.input, c.kernel, c.stride, c.padding);
      const out = conv2dOutput(c.input, c.kernel, c.stride, c.padding);
      candidates.push({
        file: fn,
        label: `Conv2d ${c.input}&times;${c.input} Kernel ${c.kernel} Stride ${c.stride} &rarr; ${out}&times;${out}`
      });
    }
  }

  // Same kernel different inputs
  for (const c of allConfigs) {
    if (c.kernel === kernel && c.stride === stride && c.input !== input) {
      const fn = makeConv2dFilename(c.input, c.kernel, c.stride, c.padding);
      const out = conv2dOutput(c.input, c.kernel, c.stride, c.padding);
      candidates.push({
        file: fn,
        label: `Conv2d ${c.input}&times;${c.input} Kernel ${c.kernel} Stride ${c.stride} &rarr; ${out}&times;${out}`
      });
    }
  }

  // Add fixed links
  for (const fl of fixedLinks) {
    if (fl.file !== currentFile + '.html') {
      candidates.push(fl);
    }
  }

  // Deduplicate and pick 5
  const seen = new Set();
  const result = [];
  for (const c of candidates) {
    if (!seen.has(c.file) && c.file !== currentFile + '.html') {
      seen.add(c.file);
      result.push(c);
      if (result.length >= 5) break;
    }
  }
  // Pad with fixed links if needed
  if (result.length < 3) {
    for (const fl of fixedLinks) {
      if (!seen.has(fl.file) && fl.file !== currentFile + '.html') {
        seen.add(fl.file);
        result.push(fl);
        if (result.length >= 5) break;
      }
    }
  }
  return result;
}

function makeConv2dFilename(input, kernel, stride, padding) {
  let name = `conv2d-output-${input}x${input}-kernel-${kernel}`;
  if (stride !== 1) name += `-stride-${stride}`;
  // Add padding suffix when padding is non-zero, UNLESS it's a "same" conv
  // that is the only config for that input+kernel+stride combo
  if (padding !== 0) {
    // Check if there's a collision: another config with same input/kernel/stride but pad=0
    const hasZeroPadVariant = CONV2D_CONFIGS.some(c =>
      c.input === input && c.kernel === kernel && c.stride === stride && c.padding === 0
    );
    if (hasZeroPadVariant) {
      name += `-pad-${padding}`;
    }
    // For configs where padding is non-zero but there's no pad=0 variant,
    // omit the suffix (e.g., 224x224-kernel-3 always means pad=1 same conv)
  }
  return name;
}

// ============================================================
// Pattern A: Conv2d Shape Calculation Pages
// ============================================================
const CONV2D_CONFIGS = [
  // Input 64x64
  { input: 64, kernel: 3, stride: 1, padding: 1 },
  { input: 64, kernel: 3, stride: 2, padding: 1 },
  { input: 64, kernel: 5, stride: 1, padding: 2 },
  { input: 64, kernel: 5, stride: 2, padding: 2 },
  { input: 64, kernel: 7, stride: 2, padding: 3 },
  // Input 128x128
  { input: 128, kernel: 3, stride: 1, padding: 1 },
  { input: 128, kernel: 3, stride: 2, padding: 1 },
  { input: 128, kernel: 5, stride: 2, padding: 2 },
  { input: 128, kernel: 7, stride: 2, padding: 3 },
  // Input 224x224
  { input: 224, kernel: 3, stride: 2, padding: 1 },
  { input: 224, kernel: 5, stride: 1, padding: 2 },
  { input: 224, kernel: 5, stride: 2, padding: 2 },
  { input: 224, kernel: 11, stride: 4, padding: 2 },
  // Input 256x256
  { input: 256, kernel: 3, stride: 1, padding: 1 },
  { input: 256, kernel: 3, stride: 2, padding: 1 },
  { input: 256, kernel: 5, stride: 2, padding: 2 },
  { input: 256, kernel: 7, stride: 2, padding: 3 },
  // Input 512x512
  { input: 512, kernel: 3, stride: 1, padding: 1 },
  { input: 512, kernel: 3, stride: 2, padding: 1 },
  { input: 512, kernel: 7, stride: 2, padding: 3 },
  // Input 32x32
  { input: 32, kernel: 3, stride: 1, padding: 0 },
  { input: 32, kernel: 3, stride: 1, padding: 1 },
  { input: 32, kernel: 5, stride: 1, padding: 0 },
  // Special
  { input: 7, kernel: 7, stride: 1, padding: 0 },
  { input: 14, kernel: 3, stride: 2, padding: 1 },
  { input: 56, kernel: 3, stride: 2, padding: 1 },
  { input: 28, kernel: 3, stride: 2, padding: 1 },
  { input: 112, kernel: 3, stride: 2, padding: 1 },
];

function generateConv2dPage(config) {
  const { input, kernel, stride, padding } = config;
  const output = conv2dOutput(input, kernel, stride, padding);
  const filename = makeConv2dFilename(input, kernel, stride, padding) + '.html';

  if (fileExists(filename)) {
    console.log(`  SKIP (exists): ${filename}`);
    return null;
  }

  const archNote = getArchNote(input, kernel, stride, padding);
  const relatedLinks = getRelatedConv2dLinks(input, kernel, stride, padding, CONV2D_CONFIGS);

  // Parameter count for common in/out channel combos
  const inCh = input >= 224 ? 3 : 64;
  const outCh = input >= 224 ? 64 : 128;
  const paramCount = (inCh * outCh * kernel * kernel + outCh).toLocaleString();

  // Determine if same convolution
  const isSame = (output === input);
  const sameNote = isSame
    ? ' This is a &ldquo;same&rdquo; convolution &mdash; the output has the same spatial dimensions as the input.'
    : '';

  // Step by step calculation
  const step1 = input + 2 * padding;
  const step2 = step1 - kernel;
  const step3val = step2 / stride;
  const step3 = Math.floor(step3val) + 1;

  // Check if padding should be in the title to disambiguate
  const needsPadInTitle = CONV2D_CONFIGS.some(c =>
    c.input === input && c.kernel === kernel && c.stride === stride && c.padding !== padding
  );

  const padSuffix = needsPadInTitle ? `, Padding ${padding}` : '';
  const padBreadcrumb = needsPadInTitle ? ` Pad ${padding}` : '';

  const titleStr = output === 1
    ? `What Does Conv2d Output with ${input}&times;${input} Input, Kernel ${kernel}${padSuffix}?`
    : `What Does Conv2d Output with ${input}&times;${input} Input, Kernel ${kernel}${stride !== 1 ? ', Stride ' + stride : ''}${padSuffix}?`;

  const breadcrumbStr = output === 1
    ? `Conv2d ${input}&times;${input} Kernel ${kernel}${padBreadcrumb}`
    : `Conv2d ${input}&times;${input} Kernel ${kernel}${stride !== 1 ? ' Stride ' + stride : ''}${padBreadcrumb}`;

  const descStr = `Conv2d with ${input}x${input} input, kernel_size=${kernel}, stride=${stride}, padding=${padding} outputs ${output}x${output}. Step-by-step formula breakdown with PyTorch code.`;

  const faqAnswer = `Conv2d with ${input}x${input} input, kernel_size=${kernel}, stride=${stride}, padding=${padding} outputs ${output}x${output}. The formula is: output = floor((input + 2*padding - kernel) / stride) + 1 = floor((${input} + 2*${padding} - ${kernel}) / ${stride}) + 1 = ${output}.`;

  const relatedHtml = relatedLinks.map(l =>
    `        <li><a href="/answers/${l.file}">${l.label}</a></li>`
  ).join('\n');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${titleStr.replace(/&times;/g, 'x')} | HeyTensor</title>
  <meta name="description" content="${descStr}">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://heytensor.com/answers/${filename}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="${titleStr.replace(/&times;/g, 'x')}">
  <meta property="og:description" content="${descStr}">
  <meta property="og:url" content="https://heytensor.com/answers/${filename}">
  <meta property="og:site_name" content="HeyTensor">
  <meta property="og:image" content="https://heytensor.com/assets/og-image.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${titleStr.replace(/&times;/g, 'x')}">
  <meta name="twitter:description" content="${descStr}">
  <meta name="twitter:image" content="https://heytensor.com/assets/og-image.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/assets/style.css">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "mainEntity": [{
      "@type": "Question",
      "name": "${titleStr.replace(/&times;/g, 'x').replace(/"/g, '\\"')}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "${faqAnswer.replace(/"/g, '\\"')}"
      }
    }]
  }
  </script>
</head>
<body>
  <header>
    <div class="container header-inner">
      <a href="/" class="logo">Hey<span>Tensor</span></a>
      <button class="mobile-toggle" aria-label="Menu">&#9776;</button>
      <nav>
        <a href="/">Calculator</a>
        <a href="/tools/">Tools</a>
        <a href="/about.html">About</a>
        <a href="/blog/">Blog</a>
        <div class="nav-right">
          <a href="https://zovo.one/pricing?utm_source=heytensor.com&amp;utm_medium=satellite&amp;utm_campaign=nav-link" class="nav-pro" target="_blank">Go Pro &#10022;</a>
          <a href="https://zovo.one/tools" class="nav-zovo">Zovo Tools</a>
        </div>
      </nav>
    </div>
  </header>

  <main class="container" style="max-width:720px;">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="/">Home</a> <span>/</span> <a href="/answers/">Answers</a> <span>/</span> <span>${breadcrumbStr}</span>
    </nav>

    <article>
      <h1>${titleStr}</h1>

      <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:var(--radius);padding:1.25rem 1.5rem;margin:1.5rem 0;">
        <p style="margin:0;line-height:1.7;"><strong>Conv2d with ${input}&times;${input} input, kernel_size=${kernel}, stride=${stride}, padding=${padding} outputs ${output}&times;${output}.</strong>${sameNote} The formula gives: floor((${input} + 2&times;${padding} - ${kernel}) / ${stride}) + 1 = ${output}.</p>
      </div>

      <h2>Formula Breakdown</h2>
      <p>The Conv2d output size formula is:</p>
      <pre><code>output_size = floor((input_size - kernel_size + 2 * padding) / stride) + 1</code></pre>
      <p>Plugging in the values for ${input}&times;${input} input:</p>
      <pre><code>output = floor((${input} - ${kernel} + 2*${padding}) / ${stride}) + 1
output = floor((${input} - ${kernel} + ${2 * padding}) / ${stride}) + 1
output = floor(${step2} / ${stride}) + 1
output = floor(${step3val}) + 1
output = ${output}</code></pre>
      <p>So the spatial dimensions go from <strong>${input}&times;${input}</strong> to <strong>${output}&times;${output}</strong>.</p>

      <h2>PyTorch Code Example</h2>
      <pre><code>import torch
import torch.nn as nn

# Define the Conv2d layer
conv = nn.Conv2d(in_channels=${inCh}, out_channels=${outCh}, kernel_size=${kernel}, stride=${stride}, padding=${padding})

# Create input tensor: (batch, channels, height, width)
x = torch.randn(1, ${inCh}, ${input}, ${input})
output = conv(x)
print(output.shape)  # torch.Size([1, ${outCh}, ${output}, ${output}])

# Verify with formula
expected = (${input} + 2 * ${padding} - ${kernel}) // ${stride} + 1
print(f"Expected output size: {expected}x{expected}")  # ${output}x${output}</code></pre>

      <h2>Architecture Context</h2>
      <p>${archNote}</p>

      <h2>Parameter Count</h2>
      <p>A Conv2d(${inCh}, ${outCh}, ${kernel}) layer has:</p>
      <pre><code>parameters = in_channels * out_channels * kernel_size^2 + out_channels (bias)
parameters = ${inCh} * ${outCh} * ${kernel} * ${kernel} + ${outCh}
parameters = ${paramCount}</code></pre>
      <p>This layer has <strong>${paramCount} trainable parameters</strong> (${inCh * outCh * kernel * kernel} weights + ${outCh} bias terms).</p>

      <h2>Practical Tips</h2>
      <ul>
        <li><strong>Memory usage:</strong> The output feature map for a single image is ${outCh} &times; ${output} &times; ${output} = ${(outCh * output * output).toLocaleString()} float values (${(outCh * output * output * 4 / 1024 / 1024).toFixed(2)} MB in float32).</li>
        <li><strong>Batch dimension:</strong> Multiply memory by batch size. A batch of 32 uses ${(outCh * output * output * 4 * 32 / 1024 / 1024).toFixed(1)} MB for this layer's output alone.</li>
        <li><strong>Same padding rule:</strong> For any kernel, setting padding = (kernel_size - 1) / 2 with stride=1 preserves spatial dimensions.</li>
      </ul>

      <h2>Related Questions</h2>
      <ul>
${relatedHtml}
      </ul>

      <div style="text-align:center;margin:2rem 0;">
        <a href="/tools/conv2d-calculator.html" style="display:inline-block;padding:0.75rem 2rem;background:var(--accent);color:#fff;border-radius:var(--radius);text-decoration:none;font-weight:600;">Try the Conv2d Calculator</a>
      </div>
    </article>
  </main>

  <footer class="site-footer">
    <div class="footer-inner">
      <div class="footer-brand">Zovo Tools</div>
      <div class="footer-tagline">Free developer tools by a solo dev. No tracking.</div>
      <a href="https://zovo.one/pricing?utm_source=heytensor.com&utm_medium=satellite&utm_campaign=footer-link" class="footer-cta">Zovo Lifetime &mdash; $99 once, free forever &rarr;</a>
      <div class="footer-copy">&copy; 2026 <a href="https://zovo.one">Zovo</a> &middot; 47/500 founding spots</div>
    </div>
  </footer>

  <nav class="zovo-network" aria-label="Zovo Tools Network">
    <div class="zovo-network-inner">
      <h3 class="zovo-network-title">Explore More Tools</h3>
      <div class="zovo-network-links">
        <a href="https://abwex.com">ABWex &mdash; A/B Testing</a>
        <a href="https://claudflow.com">ClaudFlow &mdash; Workflows</a>
        <a href="https://claudhq.com">ClaudHQ &mdash; Prompts</a>
        <a href="https://claudkit.com">ClaudKit &mdash; API</a>
        <a href="https://enhio.com">Enhio &mdash; Text Tools</a>
        <a href="https://epochpilot.com">EpochPilot &mdash; Timestamps</a>
        <a href="https://gen8x.com">Gen8X &mdash; Color Tools</a>
        <a href="https://gpt0x.com">GPT0X &mdash; AI Models</a>
        <a href="https://invokebot.com">InvokeBot &mdash; Webhooks</a>
        <a href="https://kappafy.com">Kappafy &mdash; JSON</a>
        <a href="https://kappakit.com">KappaKit &mdash; Dev Toolkit</a>
        <a href="https://kickllm.com">KickLLM &mdash; LLM Costs</a>
        <a href="https://krzen.com">Krzen &mdash; Image Tools</a>
        <a href="https://lochbot.com">LochBot &mdash; Security</a>
        <a href="https://lockml.com">LockML &mdash; ML Compare</a>
        <a href="https://ml3x.com">ML3X &mdash; Matrix Math</a>
      </div>
    </div>
  </nav>
  <script src="/assets/js/share.js"></script>
</body>
</html>`;

  return { filename, html };
}

// ============================================================
// Pattern B: Layer Parameter Count Pages
// ============================================================
const PARAM_CONFIGS = [
  // Linear layers
  { type: 'Linear', inF: 1024, outF: 512, params: 1024 * 512 + 512, arch: 'general-purpose classifier heads', label: 'Linear(1024, 512)' },
  { type: 'Linear', inF: 768, outF: 3072, params: 768 * 3072 + 3072, arch: 'BERT feed-forward network (FFN) intermediate layer', label: 'Linear(768, 3072)' },
  { type: 'Linear', inF: 4096, outF: 4096, params: 4096 * 4096 + 4096, arch: 'VGG-16 and VGG-19 fully connected layers (fc6 and fc7)', label: 'Linear(4096, 4096)' },
  { type: 'Linear', inF: 2048, outF: 1000, params: 2048 * 1000 + 1000, arch: 'ResNet-50/101/152 final classification layer (1000 ImageNet classes)', label: 'Linear(2048, 1000)' },
  { type: 'Linear', inF: 256, outF: 10, params: 256 * 10 + 10, arch: 'small classifiers for MNIST/CIFAR-10 (10 classes)', label: 'Linear(256, 10)' },
  // Conv2d layers
  { type: 'Conv2d', inCh: 3, outCh: 64, kernel: 7, params: 3 * 64 * 7 * 7 + 64, arch: 'ResNet conv1 — the first convolution layer processing raw RGB images', label: 'Conv2d(3, 64, 7)' },
  { type: 'Conv2d', inCh: 64, outCh: 128, kernel: 3, params: 64 * 128 * 3 * 3 + 128, arch: 'the transition from 64 to 128 channels, common in VGG, ResNet, and most CNNs', label: 'Conv2d(64, 128, 3)' },
  { type: 'Conv2d', inCh: 128, outCh: 256, kernel: 3, params: 128 * 256 * 3 * 3 + 256, arch: 'the 128-to-256 channel expansion in VGG and ResNet deeper stages', label: 'Conv2d(128, 256, 3)' },
  { type: 'Conv2d', inCh: 256, outCh: 512, kernel: 3, params: 256 * 512 * 3 * 3 + 512, arch: 'the 256-to-512 channel expansion in VGG-16/19 and ResNet layer4', label: 'Conv2d(256, 512, 3)' },
  { type: 'Conv2d', inCh: 512, outCh: 512, kernel: 3, params: 512 * 512 * 3 * 3 + 512, arch: 'VGG-16 repeated 512-channel convolution blocks (layers 10-13)', label: 'Conv2d(512, 512, 3)' },
];

function makeParamFilename(config) {
  if (config.type === 'Linear') {
    return `parameters-linear-${config.inF}-to-${config.outF}.html`;
  } else {
    return `parameters-conv2d-${config.inCh}-${config.outCh}-kernel-${config.kernel}.html`;
  }
}

function generateParamPage(config) {
  const filename = makeParamFilename(config);
  if (fileExists(filename)) {
    console.log(`  SKIP (exists): ${filename}`);
    return null;
  }

  const isLinear = config.type === 'Linear';
  const params = config.params;
  const paramsStr = params.toLocaleString();

  let titleStr, formulaHtml, codeHtml, descStr, faqAnswer, breadcrumb;

  if (isLinear) {
    titleStr = `How Many Parameters Does ${config.label} Have?`;
    breadcrumb = `Parameters ${config.label}`;
    descStr = `${config.label} has ${paramsStr} parameters. Formula: in_features * out_features + out_features = ${config.inF} * ${config.outF} + ${config.outF} = ${paramsStr}.`;
    faqAnswer = `${config.label} has ${paramsStr} parameters. The formula is: in_features * out_features + out_features (bias) = ${config.inF} * ${config.outF} + ${config.outF} = ${paramsStr}. The weight matrix has ${(config.inF * config.outF).toLocaleString()} elements and the bias vector has ${config.outF} elements.`;

    formulaHtml = `      <h2>Formula Breakdown</h2>
      <p>For a Linear layer, the parameter count is:</p>
      <pre><code>parameters = in_features * out_features + out_features (bias)
parameters = ${config.inF} * ${config.outF} + ${config.outF}
parameters = ${(config.inF * config.outF).toLocaleString()} + ${config.outF}
parameters = ${paramsStr}</code></pre>
      <p>The weight matrix <code>W</code> has shape (${config.outF}, ${config.inF}) = ${(config.inF * config.outF).toLocaleString()} values. The bias vector <code>b</code> has ${config.outF} values. Together: <strong>${paramsStr} trainable parameters</strong>.</p>

      <h2>Memory Usage</h2>
      <p>In float32, this layer uses ${(params * 4 / 1024 / 1024).toFixed(2)} MB of memory for weights alone. During training with Adam optimizer, multiply by 3 (weights + momentum + variance) = ${(params * 4 * 3 / 1024 / 1024).toFixed(2)} MB.</p>`;

    codeHtml = `      <h2>PyTorch Code to Verify</h2>
      <pre><code>import torch.nn as nn

layer = nn.Linear(${config.inF}, ${config.outF})

# Count parameters
total = sum(p.numel() for p in layer.parameters())
print(f"Total parameters: {total}")  # ${paramsStr}

# Break it down
print(f"Weight shape: {layer.weight.shape}")  # (${config.outF}, ${config.inF})
print(f"Weight params: {layer.weight.numel()}")  # ${(config.inF * config.outF).toLocaleString()}
print(f"Bias shape: {layer.bias.shape}")  # (${config.outF},)
print(f"Bias params: {layer.bias.numel()}")  # ${config.outF}

# Without bias
layer_no_bias = nn.Linear(${config.inF}, ${config.outF}, bias=False)
print(f"Without bias: {sum(p.numel() for p in layer_no_bias.parameters())}")  # ${(config.inF * config.outF).toLocaleString()}</code></pre>`;
  } else {
    titleStr = `How Many Parameters Does ${config.label} Have?`;
    breadcrumb = `Parameters ${config.label}`;
    descStr = `${config.label} has ${paramsStr} parameters. Formula: in_ch * out_ch * kernel^2 + out_ch = ${config.inCh} * ${config.outCh} * ${config.kernel}^2 + ${config.outCh} = ${paramsStr}.`;
    faqAnswer = `${config.label} has ${paramsStr} parameters. The formula is: in_channels * out_channels * kernel_size^2 + out_channels (bias) = ${config.inCh} * ${config.outCh} * ${config.kernel} * ${config.kernel} + ${config.outCh} = ${paramsStr}. Each of the ${config.outCh} filters has a ${config.inCh}x${config.kernel}x${config.kernel} weight tensor plus one bias value.`;

    const weightParams = config.inCh * config.outCh * config.kernel * config.kernel;

    formulaHtml = `      <h2>Formula Breakdown</h2>
      <p>For a Conv2d layer, the parameter count is:</p>
      <pre><code>parameters = in_channels * out_channels * kernel_size^2 + out_channels (bias)
parameters = ${config.inCh} * ${config.outCh} * ${config.kernel} * ${config.kernel} + ${config.outCh}
parameters = ${config.inCh} * ${config.outCh} * ${config.kernel * config.kernel} + ${config.outCh}
parameters = ${weightParams.toLocaleString()} + ${config.outCh}
parameters = ${paramsStr}</code></pre>
      <p>Each of the ${config.outCh} output filters is a 3D kernel of shape (${config.inCh}, ${config.kernel}, ${config.kernel}). That gives ${config.outCh} &times; ${config.inCh} &times; ${config.kernel} &times; ${config.kernel} = ${weightParams.toLocaleString()} weights, plus ${config.outCh} bias terms. Total: <strong>${paramsStr} trainable parameters</strong>.</p>

      <h2>Memory Usage</h2>
      <p>In float32, this layer uses ${(params * 4 / 1024 / 1024).toFixed(2)} MB of memory for weights alone. During training with Adam optimizer, multiply by 3 = ${(params * 4 * 3 / 1024 / 1024).toFixed(2)} MB.</p>`;

    codeHtml = `      <h2>PyTorch Code to Verify</h2>
      <pre><code>import torch.nn as nn

layer = nn.Conv2d(${config.inCh}, ${config.outCh}, kernel_size=${config.kernel})

# Count parameters
total = sum(p.numel() for p in layer.parameters())
print(f"Total parameters: {total}")  # ${paramsStr}

# Break it down
print(f"Weight shape: {layer.weight.shape}")  # (${config.outCh}, ${config.inCh}, ${config.kernel}, ${config.kernel})
print(f"Weight params: {layer.weight.numel()}")  # ${weightParams.toLocaleString()}
print(f"Bias shape: {layer.bias.shape}")  # (${config.outCh},)
print(f"Bias params: {layer.bias.numel()}")  # ${config.outCh}

# Without bias (common in batch-normalized networks)
layer_no_bias = nn.Conv2d(${config.inCh}, ${config.outCh}, kernel_size=${config.kernel}, bias=False)
print(f"Without bias: {sum(p.numel() for p in layer_no_bias.parameters())}")  # ${weightParams.toLocaleString()}</code></pre>`;
  }

  // Related links for parameter pages
  const paramRelated = [];
  for (const c of PARAM_CONFIGS) {
    if (c !== config) {
      paramRelated.push({
        file: makeParamFilename(c),
        label: `How many parameters does ${c.label} have?`
      });
      if (paramRelated.length >= 3) break;
    }
  }
  paramRelated.push({ file: 'how-many-parameters-resnet50.html', label: 'How many parameters does ResNet-50 have?' });
  paramRelated.push({ file: 'how-many-parameters-bert-base.html', label: 'How many parameters does BERT-base have?' });

  const relatedHtml = paramRelated.map(l =>
    `        <li><a href="/answers/${l.file}">${l.label}</a></li>`
  ).join('\n');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${titleStr} | HeyTensor</title>
  <meta name="description" content="${descStr}">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://heytensor.com/answers/${filename}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="${titleStr}">
  <meta property="og:description" content="${descStr}">
  <meta property="og:url" content="https://heytensor.com/answers/${filename}">
  <meta property="og:site_name" content="HeyTensor">
  <meta property="og:image" content="https://heytensor.com/assets/og-image.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${titleStr}">
  <meta name="twitter:description" content="${descStr}">
  <meta name="twitter:image" content="https://heytensor.com/assets/og-image.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/assets/style.css">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "mainEntity": [{
      "@type": "Question",
      "name": "${titleStr.replace(/"/g, '\\"')}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "${faqAnswer.replace(/"/g, '\\"')}"
      }
    }]
  }
  </script>
</head>
<body>
  <header>
    <div class="container header-inner">
      <a href="/" class="logo">Hey<span>Tensor</span></a>
      <button class="mobile-toggle" aria-label="Menu">&#9776;</button>
      <nav>
        <a href="/">Calculator</a>
        <a href="/tools/">Tools</a>
        <a href="/about.html">About</a>
        <a href="/blog/">Blog</a>
        <div class="nav-right">
          <a href="https://zovo.one/pricing?utm_source=heytensor.com&amp;utm_medium=satellite&amp;utm_campaign=nav-link" class="nav-pro" target="_blank">Go Pro &#10022;</a>
          <a href="https://zovo.one/tools" class="nav-zovo">Zovo Tools</a>
        </div>
      </nav>
    </div>
  </header>

  <main class="container" style="max-width:720px;">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="/">Home</a> <span>/</span> <a href="/answers/">Answers</a> <span>/</span> <span>${breadcrumb}</span>
    </nav>

    <article>
      <h1>${titleStr}</h1>

      <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:var(--radius);padding:1.25rem 1.5rem;margin:1.5rem 0;">
        <p style="margin:0;line-height:1.7;"><strong>${config.label} has ${paramsStr} trainable parameters.</strong> This includes ${isLinear ? (config.inF * config.outF).toLocaleString() + ' weights and ' + config.outF + ' bias terms' : (config.inCh * config.outCh * config.kernel * config.kernel).toLocaleString() + ' weights and ' + config.outCh + ' bias terms'}.</p>
      </div>

${formulaHtml}

      <h2>Architecture Context</h2>
      <p>This layer configuration is found in ${config.arch}. Understanding parameter counts helps you estimate model size, memory requirements, and the risk of overfitting. Layers with more parameters need more training data and compute to train effectively.</p>
      <p>${isLinear
        ? `Linear layers are often the most parameter-heavy part of a network. For example, VGG-16 has ~124M parameters in its three fully connected layers versus only ~14M in all its convolutional layers. Modern architectures minimize linear layers by using global average pooling.`
        : `Convolutional layers are parameter-efficient compared to fully connected layers because weights are shared across spatial positions. A Conv2d(${config.inCh}, ${config.outCh}, ${config.kernel}) processes any input spatial size with the same ${paramsStr} parameters.`
      }</p>

${codeHtml}

      <h2>Comparison: With vs. Without Bias</h2>
      <table style="width:100%;border-collapse:collapse;margin:1rem 0;">
        <tr style="border-bottom:2px solid var(--border);">
          <th style="padding:0.5rem;text-align:left;">Configuration</th>
          <th style="padding:0.5rem;text-align:right;">Parameters</th>
        </tr>
        <tr style="border-bottom:1px solid var(--border);">
          <td style="padding:0.5rem;">${config.label} (with bias)</td>
          <td style="padding:0.5rem;text-align:right;">${paramsStr}</td>
        </tr>
        <tr style="border-bottom:1px solid var(--border);">
          <td style="padding:0.5rem;">${config.label.replace(')', ', bias=False)')} </td>
          <td style="padding:0.5rem;text-align:right;">${(params - (isLinear ? config.outF : config.outCh)).toLocaleString()}</td>
        </tr>
      </table>
      <p>When using BatchNorm after a convolutional layer, the bias is redundant because BatchNorm has its own bias term. Setting <code>bias=False</code> saves ${isLinear ? config.outF : config.outCh} parameters per layer.</p>

      <h2>Related Questions</h2>
      <ul>
${relatedHtml}
      </ul>

      <div style="text-align:center;margin:2rem 0;">
        <a href="/tools/parameter-counter.html" style="display:inline-block;padding:0.75rem 2rem;background:var(--accent);color:#fff;border-radius:var(--radius);text-decoration:none;font-weight:600;">Try the Parameter Counter</a>
      </div>
    </article>
  </main>

  <footer class="site-footer">
    <div class="footer-inner">
      <div class="footer-brand">Zovo Tools</div>
      <div class="footer-tagline">Free developer tools by a solo dev. No tracking.</div>
      <a href="https://zovo.one/pricing?utm_source=heytensor.com&utm_medium=satellite&utm_campaign=footer-link" class="footer-cta">Zovo Lifetime &mdash; $99 once, free forever &rarr;</a>
      <div class="footer-copy">&copy; 2026 <a href="https://zovo.one">Zovo</a> &middot; 47/500 founding spots</div>
    </div>
  </footer>

  <nav class="zovo-network" aria-label="Zovo Tools Network">
    <div class="zovo-network-inner">
      <h3 class="zovo-network-title">Explore More Tools</h3>
      <div class="zovo-network-links">
        <a href="https://abwex.com">ABWex &mdash; A/B Testing</a>
        <a href="https://claudflow.com">ClaudFlow &mdash; Workflows</a>
        <a href="https://claudhq.com">ClaudHQ &mdash; Prompts</a>
        <a href="https://claudkit.com">ClaudKit &mdash; API</a>
        <a href="https://enhio.com">Enhio &mdash; Text Tools</a>
        <a href="https://epochpilot.com">EpochPilot &mdash; Timestamps</a>
        <a href="https://gen8x.com">Gen8X &mdash; Color Tools</a>
        <a href="https://gpt0x.com">GPT0X &mdash; AI Models</a>
        <a href="https://invokebot.com">InvokeBot &mdash; Webhooks</a>
        <a href="https://kappafy.com">Kappafy &mdash; JSON</a>
        <a href="https://kappakit.com">KappaKit &mdash; Dev Toolkit</a>
        <a href="https://kickllm.com">KickLLM &mdash; LLM Costs</a>
        <a href="https://krzen.com">Krzen &mdash; Image Tools</a>
        <a href="https://lochbot.com">LochBot &mdash; Security</a>
        <a href="https://lockml.com">LockML &mdash; ML Compare</a>
        <a href="https://ml3x.com">ML3X &mdash; Matrix Math</a>
      </div>
    </div>
  </nav>
  <script src="/assets/js/share.js"></script>
</body>
</html>`;

  return { filename, html };
}

// ============================================================
// Pattern C: Error Message Pages
// ============================================================
const ERROR_CONFIGS = [
  {
    slug: 'expected-float-got-long',
    errorMsg: 'expected dtype Float but got dtype Long',
    title: 'How to Fix "expected dtype Float but got dtype Long" in PyTorch',
    breadcrumb: 'Expected Float Got Long',
    description: 'Fix PyTorch "expected dtype Float but got dtype Long" by converting tensors with .float() or .to(torch.float32). Common with integer labels and loss functions.',
    faqAnswer: 'This error occurs when a function expects float32 tensors but receives int64 (Long) tensors. Fix it by calling .float() or .to(torch.float32) on the tensor. This commonly happens with labels in regression tasks, or when loading data that defaults to integer types.',
    cause: 'PyTorch operations like loss functions, matrix multiplications, and neural network layers expect float tensors. When you pass integer (Long) tensors, PyTorch raises this error because it does not auto-cast between these types.',
    scenarios: [
      {
        title: 'Scenario 1: Labels in Binary Classification (BCELoss)',
        problem: 'BCELoss and BCEWithLogitsLoss expect float targets, but integer labels are common.',
        badCode: `labels = torch.tensor([0, 1, 1, 0])  # dtype: torch.int64 (Long)
output = model(x)  # float32
loss = nn.BCEWithLogitsLoss()(output, labels)
# RuntimeError: expected dtype Float but got dtype Long`,
        fixCode: `labels = torch.tensor([0, 1, 1, 0]).float()  # Convert to float32
output = model(x)
loss = nn.BCEWithLogitsLoss()(output, labels)  # Works!

# Alternative: specify dtype at creation
labels = torch.tensor([0, 1, 1, 0], dtype=torch.float32)`,
        explanation: 'Binary cross-entropy needs float targets because it computes log probabilities. Integer labels must be cast to float first.'
      },
      {
        title: 'Scenario 2: Regression with MSELoss',
        problem: 'MSELoss needs float inputs but data loading may produce integers.',
        badCode: `targets = torch.tensor([3, 7, 2, 9])  # int64
predictions = model(x)  # float32
loss = nn.MSELoss()(predictions, targets)
# RuntimeError: expected dtype Float but got dtype Long`,
        fixCode: `targets = torch.tensor([3, 7, 2, 9]).float()  # float32
predictions = model(x)
loss = nn.MSELoss()(predictions, targets)  # Works!

# Or in your Dataset.__getitem__:
def __getitem__(self, idx):
    return self.features[idx].float(), self.targets[idx].float()`,
        explanation: 'MSE computes (prediction - target)^2, which requires both tensors to be floating point. Converting in your Dataset is the cleanest fix.'
      },
      {
        title: 'Scenario 3: Matrix Multiplication with Integer Tensors',
        problem: 'torch.matmul and @ operator require float tensors for most operations.',
        badCode: `a = torch.tensor([[1, 2], [3, 4]])  # int64
b = torch.tensor([[5, 6], [7, 8]])  # int64
# On some operations this works, but mixing with float fails:
w = torch.randn(2, 2)  # float32
result = a @ w
# RuntimeError: expected dtype Float but got dtype Long`,
        fixCode: `a = torch.tensor([[1, 2], [3, 4]]).float()
w = torch.randn(2, 2)
result = a @ w  # Works! Both are float32

# Or use .to() for explicit control:
a = torch.tensor([[1, 2], [3, 4]]).to(torch.float32)`,
        explanation: 'When mixing tensor types in arithmetic, PyTorch requires explicit casting. Using .float() or .to(torch.float32) converts Long tensors to Float.'
      }
    ],
    relatedLinks: [
      { file: 'runtime-error-expected-scalar-type-float.html', label: 'Fix "expected scalar type Float"' },
      { file: 'cuda-out-of-memory-fix.html', label: 'Fix CUDA out of memory' },
      { file: 'mat1-mat2-shapes-cannot-be-multiplied.html', label: 'Fix mat1 mat2 shapes cannot be multiplied' },
      { file: 'cannot-broadcast-tensors.html', label: 'Fix cannot broadcast tensors' },
    ],
    ctaLink: '/tools/shape-mismatch.html',
    ctaLabel: 'Try the Shape Mismatch Solver'
  },
  {
    slug: 'sizes-of-tensors-must-match',
    errorMsg: 'Sizes of tensors must match except in dimension',
    title: 'How to Fix "Sizes of tensors must match except in dimension" in PyTorch',
    breadcrumb: 'Sizes Must Match',
    description: 'Fix PyTorch "Sizes of tensors must match except in dimension" by reshaping, padding, or slicing tensors to compatible shapes before concatenation.',
    faqAnswer: 'This error occurs when using torch.cat() or torch.stack() with tensors that have mismatched dimensions. All tensors must have the same shape except in the concatenation dimension. Fix by padding shorter tensors, slicing longer ones, or reshaping to match.',
    cause: 'When concatenating tensors with torch.cat() or torch.stack(), all non-concatenation dimensions must match exactly. For example, torch.cat([tensor_a, tensor_b], dim=0) requires both tensors to have the same shape in dimensions 1, 2, etc.',
    scenarios: [
      {
        title: 'Scenario 1: Concatenating Feature Maps of Different Sizes',
        problem: 'Skip connections or multi-scale features may produce tensors of different spatial sizes.',
        badCode: `features_high = torch.randn(1, 64, 32, 32)  # 32x32
features_low = torch.randn(1, 64, 16, 16)   # 16x16
combined = torch.cat([features_high, features_low], dim=1)
# RuntimeError: Sizes of tensors must match except in dimension 1.
# Expected size 32 but got size 16 for tensor number 1 in the list`,
        fixCode: `import torch.nn.functional as F

features_high = torch.randn(1, 64, 32, 32)
features_low = torch.randn(1, 64, 16, 16)

# Option 1: Upsample the smaller tensor
features_low_up = F.interpolate(features_low, size=(32, 32), mode='bilinear', align_corners=False)
combined = torch.cat([features_high, features_low_up], dim=1)  # Works: [1, 128, 32, 32]

# Option 2: Downsample the larger tensor
features_high_down = F.adaptive_avg_pool2d(features_high, (16, 16))
combined = torch.cat([features_high_down, features_low], dim=1)  # Works: [1, 128, 16, 16]`,
        explanation: 'In U-Net and FPN architectures, feature maps at different scales must be resized before concatenation. Use F.interpolate for upsampling or adaptive pooling for downsampling.'
      },
      {
        title: 'Scenario 2: Batching Sequences of Different Lengths',
        problem: 'NLP tasks often have variable-length sequences that cannot be directly concatenated.',
        badCode: `seq1 = torch.randn(5, 768)   # 5 tokens
seq2 = torch.randn(8, 768)   # 8 tokens
batch = torch.stack([seq1, seq2])
# RuntimeError: Sizes of tensors must match except in dimension 0`,
        fixCode: `# Option 1: Pad to maximum length
from torch.nn.utils.rnn import pad_sequence

seq1 = torch.randn(5, 768)
seq2 = torch.randn(8, 768)
batch = pad_sequence([seq1, seq2], batch_first=True)  # [2, 8, 768], padded with zeros

# Option 2: Truncate to minimum length
min_len = min(seq1.size(0), seq2.size(0))
batch = torch.stack([seq1[:min_len], seq2[:min_len]])  # [2, 5, 768]`,
        explanation: 'pad_sequence pads shorter tensors with zeros to match the longest. Use attention masks to ignore padded positions during training.'
      },
      {
        title: 'Scenario 3: Residual Connection Shape Mismatch',
        problem: 'Skip/residual connections require the input and output to have identical shapes.',
        badCode: `class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, padding=1)  # Changes channels!

    def forward(self, x):  # x: [B, 64, H, W]
        return x + self.conv(x)  # Error! [B, 64, H, W] + [B, 128, H, W]
# RuntimeError: Sizes of tensors must match except in dimension`,
        fixCode: `class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, padding=1)
        self.shortcut = nn.Conv2d(64, 128, 1)  # 1x1 conv to match channels

    def forward(self, x):
        return self.shortcut(x) + self.conv(x)  # Both [B, 128, H, W]. Works!`,
        explanation: 'ResNet uses 1x1 convolutions (projection shortcuts) to match channel dimensions when the residual path changes the number of channels.'
      }
    ],
    relatedLinks: [
      { file: 'cannot-broadcast-tensors.html', label: 'Fix "cannot broadcast tensors"' },
      { file: 'view-size-not-compatible.html', label: 'Fix "view size not compatible"' },
      { file: 'expected-4d-input-got-3d.html', label: 'Fix "expected 4D input got 3D"' },
      { file: 'mat1-mat2-shapes-cannot-be-multiplied.html', label: 'Fix "mat1 mat2 shapes cannot be multiplied"' },
    ],
    ctaLink: '/tools/shape-mismatch.html',
    ctaLabel: 'Try the Shape Mismatch Solver'
  },
  {
    slug: 'input-target-batch-size-mismatch',
    errorMsg: "input and target batch size don\\'t match",
    title: 'How to Fix "input and target batch size don\'t match" in PyTorch',
    breadcrumb: 'Batch Size Mismatch',
    description: 'Fix PyTorch "input and target batch size don\'t match" by ensuring model output and labels have the same batch dimension. Common with CrossEntropyLoss and reshape errors.',
    faqAnswer: "This error occurs when your model's output tensor and target tensor have different batch sizes. This usually means a reshape or squeeze operation accidentally changed the batch dimension, or you're indexing into a batch incorrectly. Fix by checking tensor shapes before the loss function.",
    cause: 'Loss functions compare model predictions with targets element-by-element (or batch-element-by-batch-element). When the batch dimensions do not match, PyTorch cannot compute the loss. This typically happens due to incorrect squeezing, reshaping, or when the DataLoader returns mismatched batches.',
    scenarios: [
      {
        title: 'Scenario 1: Accidental Squeeze Removing Batch Dimension',
        problem: 'Using .squeeze() on a single-sample batch removes the batch dimension.',
        badCode: `output = model(x)  # [1, 10] — batch_size=1
output = output.squeeze()  # [10] — batch dimension removed!
target = torch.tensor([3])  # [1]
loss = nn.CrossEntropyLoss()(output, target)
# ValueError: Expected input batch_size (10) to match target batch_size (1)`,
        fixCode: `output = model(x)  # [1, 10]
# Use squeeze only on specific dimensions, never the batch dim
output = output.squeeze(dim=-1)  # Only squeeze last dim if needed
# Or better: don't squeeze at all
target = torch.tensor([3])  # [1]
loss = nn.CrossEntropyLoss()(output, target)  # Works! [1, 10] vs [1]`,
        explanation: 'Never use .squeeze() without specifying which dimension to squeeze. When batch_size=1, squeeze() removes the batch dimension, causing shape mismatches downstream.'
      },
      {
        title: 'Scenario 2: Wrong Reshape in Forward Pass',
        problem: 'Hardcoding reshape dimensions instead of using batch_size from input.',
        badCode: `class Model(nn.Module):
    def forward(self, x):
        x = self.features(x)
        x = x.view(32, -1)  # Hardcoded batch_size=32!
        return self.classifier(x)

# Fails when batch_size != 32 (e.g., last batch in epoch)`,
        fixCode: `class Model(nn.Module):
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Use actual batch size
        # Or equivalently:
        x = x.flatten(1)  # Flatten all dims except batch
        return self.classifier(x)`,
        explanation: 'Always use x.size(0) or x.shape[0] for the batch dimension in reshape operations. This handles variable batch sizes including the last (often smaller) batch in an epoch.'
      },
      {
        title: 'Scenario 3: DataLoader Returning Mismatched Batch Elements',
        problem: 'Custom Dataset returning labels with wrong shape.',
        badCode: `class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx:idx+2]  # Bug: returns 2 labels per image!
        return image, label

# DataLoader stacks images [B, C, H, W] but labels become [B, 2]
# CrossEntropyLoss sees batch mismatch`,
        fixCode: `class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]  # Single label per image
        return image, label

# Now DataLoader produces [B, C, H, W] images and [B] labels
# Always verify shapes:
for x, y in loader:
    print(f"Input: {x.shape}, Target: {y.shape}")
    break`,
        explanation: 'Verify your Dataset returns exactly one label per sample. Add shape assertions in your training loop to catch mismatches early.'
      }
    ],
    relatedLinks: [
      { file: 'expected-4d-input-got-3d.html', label: 'Fix "expected 4D input got 3D"' },
      { file: 'mat1-mat2-shapes-cannot-be-multiplied.html', label: 'Fix "mat1 mat2 shapes cannot be multiplied"' },
      { file: 'flatten-after-conv2d.html', label: 'How to flatten after Conv2d' },
      { file: 'view-size-not-compatible.html', label: 'Fix "view size not compatible"' },
    ],
    ctaLink: '/tools/shape-mismatch.html',
    ctaLabel: 'Try the Shape Mismatch Solver'
  },
  {
    slug: 'variable-modified-for-gradient',
    errorMsg: 'one of the variables needed for gradient computation has been modified',
    title: 'How to Fix "one of the variables needed for gradient computation has been modified" in PyTorch',
    breadcrumb: 'Variable Modified for Gradient',
    description: 'Fix PyTorch "one of the variables needed for gradient computation has been modified" by avoiding in-place operations on tensors that require grad. Use out-of-place alternatives.',
    faqAnswer: 'This error occurs when you modify a tensor in-place that is needed for gradient computation during backward(). PyTorch stores references to intermediate values for backpropagation, and in-place modifications corrupt these saved values. Fix by replacing in-place operations (like +=, .add_(), .mul_(), .zero_()) with out-of-place alternatives (like + , .add(), .mul(), .clone()).',
    cause: 'PyTorch autograd records operations to build a computation graph for backpropagation. When you modify a tensor in-place (e.g., x += 1, x.add_(1), x[:] = value), the saved reference becomes invalid because the underlying data changed. This causes incorrect gradients or this error.',
    scenarios: [
      {
        title: 'Scenario 1: In-place Addition in Forward Pass',
        problem: 'Using += instead of + modifies tensors in place.',
        badCode: `class Model(nn.Module):
    def forward(self, x):
        out = self.layer1(x)
        out += self.residual(x)  # In-place! Modifies out
        out = self.layer2(out)
        return out

# RuntimeError: one of the variables needed for gradient computation
# has been modified by an inplace operation`,
        fixCode: `class Model(nn.Module):
    def forward(self, x):
        out = self.layer1(x)
        out = out + self.residual(x)  # Out-of-place: creates new tensor
        out = self.layer2(out)
        return out

# Or use torch.add explicitly:
# out = torch.add(out, self.residual(x))`,
        explanation: 'Replace += with = ... + to create a new tensor instead of modifying in place. This preserves the original values needed for gradient computation.'
      },
      {
        title: 'Scenario 2: In-place Activation Functions',
        problem: 'Using inplace=True on activations that feed into operations needing gradients.',
        badCode: `model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),  # In-place modification
    nn.Linear(128, 10)
)
# May cause "modified by an inplace operation" in some graph configurations`,
        fixCode: `model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(inplace=False),  # Safe: creates new tensor
    nn.Linear(128, 10)
)

# inplace=True saves memory but risks gradient errors.
# Only use inplace=True when you're certain the tensor isn't
# needed by other branches of the computation graph.`,
        explanation: 'While inplace=True saves memory, it can break gradient computation in networks with skip connections or shared parameters. Default to inplace=False unless profiling shows a clear memory benefit.'
      },
      {
        title: 'Scenario 3: Modifying Weight Tensors During Forward',
        problem: 'Directly modifying parameters or buffers during forward pass.',
        badCode: `class Model(nn.Module):
    def forward(self, x):
        self.weight.data.zero_()  # In-place modification of parameter!
        self.weight.data.add_(compute_weight(x))
        return F.linear(x, self.weight)
# RuntimeError: one of the variables needed for gradient computation
# has been modified by an inplace operation`,
        fixCode: `class Model(nn.Module):
    def forward(self, x):
        # Create a new weight tensor instead of modifying in-place
        w = compute_weight(x)  # Compute fresh weights
        return F.linear(x, w)

# If you need conditional weights, clone first:
# w = self.weight.clone()
# w = w + delta  # out-of-place modification on clone`,
        explanation: 'Never modify .data of parameters during forward pass. Use functional operations that create new tensors, or .clone() first to avoid corrupting the computation graph.'
      }
    ],
    relatedLinks: [
      { file: 'gradient-is-none-fix.html', label: 'Fix "gradient is None"' },
      { file: 'cuda-out-of-memory-fix.html', label: 'Fix CUDA out of memory' },
      { file: 'runtime-error-expected-scalar-type-float.html', label: 'Fix "expected scalar type Float"' },
      { file: 'no-module-named-torch.html', label: 'Fix "No module named torch"' },
    ],
    ctaLink: '/tools/shape-mismatch.html',
    ctaLabel: 'Debug Shape Errors'
  },
  {
    slug: 'weight-on-cpu-input-on-cuda',
    errorMsg: 'weight is on CPU but input is on CUDA',
    title: 'How to Fix "weight is on CPU but input is on CUDA" in PyTorch',
    breadcrumb: 'Weight CPU Input CUDA',
    description: 'Fix PyTorch "weight is on CPU but input is on CUDA" by moving your model to GPU with model.to("cuda") before passing GPU tensors. Ensure model and data are on the same device.',
    faqAnswer: 'This error occurs when your model weights are on CPU but your input tensor is on GPU (or vice versa). Fix by calling model.to("cuda") or model.cuda() before inference/training. Always ensure both model and data are on the same device.',
    cause: 'PyTorch requires all tensors in an operation to be on the same device. If you move your input data to GPU with .cuda() but forget to move the model, the model weights remain on CPU while the input is on CUDA. This device mismatch causes the error.',
    scenarios: [
      {
        title: 'Scenario 1: Forgot to Move Model to GPU',
        problem: 'Moving data to CUDA without moving the model first.',
        badCode: `model = MyModel()  # Model on CPU by default
x = torch.randn(1, 3, 224, 224).cuda()  # Input on GPU
output = model(x)
# RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
# (torch.FloatTensor) should be the same`,
        fixCode: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)  # Move model to GPU
x = torch.randn(1, 3, 224, 224).to(device)  # Move input to same device
output = model(x)  # Works!

# Best practice: always use a device variable
# This makes your code work on both CPU and GPU machines`,
        explanation: 'Always define a device variable and use .to(device) for both model and data. This pattern works seamlessly on machines with or without GPUs.'
      },
      {
        title: 'Scenario 2: Loading a Checkpoint on Wrong Device',
        problem: 'Loading a GPU-trained model on CPU or vice versa without map_location.',
        badCode: `# Model was saved on GPU
# torch.save(model.state_dict(), "model.pth")

# Loading on CPU machine without map_location
model = MyModel()
model.load_state_dict(torch.load("model.pth"))  # Loads to GPU tensors!
x = torch.randn(1, 3, 224, 224)  # CPU input
output = model(x)
# RuntimeError: weight is on CUDA but input is on CPU`,
        fixCode: `# Always specify map_location when loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)

x = torch.randn(1, 3, 224, 224).to(device)
output = model(x)  # Works!

# Or force CPU loading:
# model.load_state_dict(torch.load("model.pth", map_location="cpu"))`,
        explanation: 'torch.load with map_location ensures weights are loaded to the correct device regardless of where they were saved. Always use map_location=device for portable code.'
      },
      {
        title: 'Scenario 3: Mixed Device in DataLoader Loop',
        problem: 'Forgetting to move batch data to GPU inside the training loop.',
        badCode: `model = model.cuda()
for batch_x, batch_y in dataloader:
    # batch_x and batch_y are on CPU (DataLoader default)
    output = model(batch_x)  # Error! CPU input, CUDA model
    loss = criterion(output, batch_y)`,
        fixCode: `device = torch.device("cuda")
model = model.to(device)

for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)  # Move to GPU
    batch_y = batch_y.to(device)  # Move labels too!
    output = model(batch_x)  # Works!
    loss = criterion(output, batch_y)

# Pro tip: use non_blocking=True for async transfers
# batch_x = batch_x.to(device, non_blocking=True)`,
        explanation: 'DataLoader returns CPU tensors by default. Move each batch to the model device at the start of each iteration. Use non_blocking=True with pinned memory for faster transfers.'
      }
    ],
    relatedLinks: [
      { file: 'cuda-out-of-memory-fix.html', label: 'Fix CUDA out of memory' },
      { file: 'runtime-error-expected-scalar-type-float.html', label: 'Fix "expected scalar type Float"' },
      { file: 'no-module-named-torch.html', label: 'Fix "No module named torch"' },
      { file: 'gradient-is-none-fix.html', label: 'Fix "gradient is None"' },
    ],
    ctaLink: '/tools/cuda-out-of-memory.html',
    ctaLabel: 'Try the CUDA OOM Solver'
  },
];

function generateErrorPage(config) {
  const filename = config.slug + '.html';
  if (fileExists(filename)) {
    console.log(`  SKIP (exists): ${filename}`);
    return null;
  }

  const scenariosHtml = config.scenarios.map((s, i) => `
      <h2>${s.title}</h2>
      <p>${s.problem}</p>
      <h3>The Error</h3>
      <pre><code>${escapeHtml(s.badCode)}</code></pre>
      <h3>The Fix</h3>
      <pre><code>${escapeHtml(s.fixCode)}</code></pre>
      <p>${s.explanation}</p>`
  ).join('\n');

  const relatedHtml = config.relatedLinks.map(l =>
    `        <li><a href="/answers/${l.file}">${l.label}</a></li>`
  ).join('\n');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${config.title} | HeyTensor</title>
  <meta name="description" content="${config.description}">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://heytensor.com/answers/${filename}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="${config.title}">
  <meta property="og:description" content="${config.description}">
  <meta property="og:url" content="https://heytensor.com/answers/${filename}">
  <meta property="og:site_name" content="HeyTensor">
  <meta property="og:image" content="https://heytensor.com/assets/og-image.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${config.title}">
  <meta name="twitter:description" content="${config.description}">
  <meta name="twitter:image" content="https://heytensor.com/assets/og-image.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/assets/style.css">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "mainEntity": [{
      "@type": "Question",
      "name": "${config.title.replace(/"/g, '\\"')}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "${config.faqAnswer.replace(/"/g, '\\"')}"
      }
    }]
  }
  </script>
</head>
<body>
  <header>
    <div class="container header-inner">
      <a href="/" class="logo">Hey<span>Tensor</span></a>
      <button class="mobile-toggle" aria-label="Menu">&#9776;</button>
      <nav>
        <a href="/">Calculator</a>
        <a href="/tools/">Tools</a>
        <a href="/about.html">About</a>
        <a href="/blog/">Blog</a>
        <div class="nav-right">
          <a href="https://zovo.one/pricing?utm_source=heytensor.com&amp;utm_medium=satellite&amp;utm_campaign=nav-link" class="nav-pro" target="_blank">Go Pro &#10022;</a>
          <a href="https://zovo.one/tools" class="nav-zovo">Zovo Tools</a>
        </div>
      </nav>
    </div>
  </header>

  <main class="container" style="max-width:720px;">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="/">Home</a> <span>/</span> <a href="/answers/">Answers</a> <span>/</span> <span>${config.breadcrumb}</span>
    </nav>

    <article>
      <h1>${config.title}</h1>

      <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:var(--radius);padding:1.25rem 1.5rem;margin:1.5rem 0;">
        <p style="margin:0;line-height:1.7;"><strong>The error <code>${escapeHtml(config.errorMsg)}</code> means ${config.cause.charAt(0).toLowerCase() + config.cause.slice(1)}</strong></p>
      </div>

      <h2>What Causes This Error</h2>
      <p>${config.cause}</p>
${scenariosHtml}

      <h2>Quick Debugging Checklist</h2>
      <ul>
        <li>Print tensor <code>.dtype</code> and <code>.device</code> before operations</li>
        <li>Check for in-place operations: <code>+=</code>, <code>*=</code>, <code>.add_()</code>, <code>.mul_()</code></li>
        <li>Verify shapes with <code>print(tensor.shape)</code> at each step</li>
        <li>Use <code>torch.autograd.set_detect_anomaly(True)</code> to pinpoint the exact operation</li>
      </ul>
      <pre><code># Enable anomaly detection to find the exact line
torch.autograd.set_detect_anomaly(True)

# Check tensor properties
print(f"dtype: {tensor.dtype}, device: {tensor.device}, shape: {tensor.shape}")
print(f"requires_grad: {tensor.requires_grad}")</code></pre>

      <h2>Related Questions</h2>
      <ul>
${relatedHtml}
      </ul>

      <div style="text-align:center;margin:2rem 0;">
        <a href="${config.ctaLink}" style="display:inline-block;padding:0.75rem 2rem;background:var(--accent);color:#fff;border-radius:var(--radius);text-decoration:none;font-weight:600;">${config.ctaLabel}</a>
      </div>
    </article>
  </main>

  <footer class="site-footer">
    <div class="footer-inner">
      <div class="footer-brand">Zovo Tools</div>
      <div class="footer-tagline">Free developer tools by a solo dev. No tracking.</div>
      <a href="https://zovo.one/pricing?utm_source=heytensor.com&utm_medium=satellite&utm_campaign=footer-link" class="footer-cta">Zovo Lifetime &mdash; $99 once, free forever &rarr;</a>
      <div class="footer-copy">&copy; 2026 <a href="https://zovo.one">Zovo</a> &middot; 47/500 founding spots</div>
    </div>
  </footer>

  <nav class="zovo-network" aria-label="Zovo Tools Network">
    <div class="zovo-network-inner">
      <h3 class="zovo-network-title">Explore More Tools</h3>
      <div class="zovo-network-links">
        <a href="https://abwex.com">ABWex &mdash; A/B Testing</a>
        <a href="https://claudflow.com">ClaudFlow &mdash; Workflows</a>
        <a href="https://claudhq.com">ClaudHQ &mdash; Prompts</a>
        <a href="https://claudkit.com">ClaudKit &mdash; API</a>
        <a href="https://enhio.com">Enhio &mdash; Text Tools</a>
        <a href="https://epochpilot.com">EpochPilot &mdash; Timestamps</a>
        <a href="https://gen8x.com">Gen8X &mdash; Color Tools</a>
        <a href="https://gpt0x.com">GPT0X &mdash; AI Models</a>
        <a href="https://invokebot.com">InvokeBot &mdash; Webhooks</a>
        <a href="https://kappafy.com">Kappafy &mdash; JSON</a>
        <a href="https://kappakit.com">KappaKit &mdash; Dev Toolkit</a>
        <a href="https://kickllm.com">KickLLM &mdash; LLM Costs</a>
        <a href="https://krzen.com">Krzen &mdash; Image Tools</a>
        <a href="https://lochbot.com">LochBot &mdash; Security</a>
        <a href="https://lockml.com">LockML &mdash; ML Compare</a>
        <a href="https://ml3x.com">ML3X &mdash; Matrix Math</a>
      </div>
    </div>
  </nav>
  <script src="/assets/js/share.js"></script>
</body>
</html>`;

  return { filename, html };
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ============================================================
// Answers Index Page
// ============================================================
function generateAnswersIndex(allPages) {
  // Collect all answer files (existing + new)
  const existingFiles = Array.from(EXISTING_FILES).filter(f => f.endsWith('.html') && f !== 'index.html');
  const newFiles = allPages.map(p => p.filename);
  const allFiles = [...new Set([...existingFiles, ...newFiles])].sort();

  const links = allFiles.map(f => {
    const name = f.replace('.html', '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `        <li><a href="/answers/${f}">${name}</a></li>`;
  }).join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PyTorch Answers &mdash; Shape Calculations, Parameters &amp; Error Fixes | HeyTensor</title>
  <meta name="description" content="Quick answers to common PyTorch questions: Conv2d output shapes, layer parameter counts, and error message fixes with code examples.">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://heytensor.com/answers/">
  <meta property="og:type" content="website">
  <meta property="og:title" content="PyTorch Answers &mdash; HeyTensor">
  <meta property="og:description" content="Quick answers to common PyTorch questions with code examples.">
  <meta property="og:url" content="https://heytensor.com/answers/">
  <meta property="og:site_name" content="HeyTensor">
  <meta property="og:image" content="https://heytensor.com/assets/og-image.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/assets/style.css">
</head>
<body>
  <header>
    <div class="container header-inner">
      <a href="/" class="logo">Hey<span>Tensor</span></a>
      <button class="mobile-toggle" aria-label="Menu">&#9776;</button>
      <nav>
        <a href="/">Calculator</a>
        <a href="/tools/">Tools</a>
        <a href="/about.html">About</a>
        <a href="/blog/">Blog</a>
        <div class="nav-right">
          <a href="https://zovo.one/pricing?utm_source=heytensor.com&amp;utm_medium=satellite&amp;utm_campaign=nav-link" class="nav-pro" target="_blank">Go Pro &#10022;</a>
          <a href="https://zovo.one/tools" class="nav-zovo">Zovo Tools</a>
        </div>
      </nav>
    </div>
  </header>

  <main class="container" style="max-width:720px;">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="/">Home</a> <span>/</span> <span>Answers</span>
    </nav>

    <h1>PyTorch Answers</h1>
    <p>Quick answers to common PyTorch questions &mdash; Conv2d output shapes, layer parameter counts, and error message fixes. Each answer includes the formula, step-by-step calculation, and PyTorch code you can copy.</p>

    <h2>All Questions (${allFiles.length})</h2>
    <ul>
${links}
    </ul>

    <div style="text-align:center;margin:2rem 0;">
      <a href="/tools/conv2d-calculator.html" style="display:inline-block;padding:0.75rem 2rem;background:var(--accent);color:#fff;border-radius:var(--radius);text-decoration:none;font-weight:600;">Try the Conv2d Calculator</a>
    </div>
  </main>

  <footer class="site-footer">
    <div class="footer-inner">
      <div class="footer-brand">Zovo Tools</div>
      <div class="footer-tagline">Free developer tools by a solo dev. No tracking.</div>
      <a href="https://zovo.one/pricing?utm_source=heytensor.com&utm_medium=satellite&utm_campaign=footer-link" class="footer-cta">Zovo Lifetime &mdash; $99 once, free forever &rarr;</a>
      <div class="footer-copy">&copy; 2026 <a href="https://zovo.one">Zovo</a> &middot; 47/500 founding spots</div>
    </div>
  </footer>

  <nav class="zovo-network" aria-label="Zovo Tools Network">
    <div class="zovo-network-inner">
      <h3 class="zovo-network-title">Explore More Tools</h3>
      <div class="zovo-network-links">
        <a href="https://abwex.com">ABWex &mdash; A/B Testing</a>
        <a href="https://claudflow.com">ClaudFlow &mdash; Workflows</a>
        <a href="https://claudhq.com">ClaudHQ &mdash; Prompts</a>
        <a href="https://claudkit.com">ClaudKit &mdash; API</a>
        <a href="https://enhio.com">Enhio &mdash; Text Tools</a>
        <a href="https://epochpilot.com">EpochPilot &mdash; Timestamps</a>
        <a href="https://gen8x.com">Gen8X &mdash; Color Tools</a>
        <a href="https://gpt0x.com">GPT0X &mdash; AI Models</a>
        <a href="https://invokebot.com">InvokeBot &mdash; Webhooks</a>
        <a href="https://kappafy.com">Kappafy &mdash; JSON</a>
        <a href="https://kappakit.com">KappaKit &mdash; Dev Toolkit</a>
        <a href="https://kickllm.com">KickLLM &mdash; LLM Costs</a>
        <a href="https://krzen.com">Krzen &mdash; Image Tools</a>
        <a href="https://lochbot.com">LochBot &mdash; Security</a>
        <a href="https://lockml.com">LockML &mdash; ML Compare</a>
        <a href="https://ml3x.com">ML3X &mdash; Matrix Math</a>
      </div>
    </div>
  </nav>
  <script src="/assets/js/share.js"></script>
</body>
</html>`;
}

// ============================================================
// Sitemap Update
// ============================================================
function updateSitemap(newPages) {
  let sitemap = fs.readFileSync(SITEMAP_PATH, 'utf8');

  const newEntries = newPages.map(p => `  <url>
<loc>https://heytensor.com/answers/${p.filename}</loc>
<lastmod>${TODAY}</lastmod>
<changefreq>monthly</changefreq>
<priority>0.5</priority>
</url>`).join('\n');

  // Add answers index
  const indexEntry = `  <url>
<loc>https://heytensor.com/answers/</loc>
<lastmod>${TODAY}</lastmod>
<changefreq>weekly</changefreq>
<priority>0.6</priority>
</url>`;

  // Check if answers index already exists
  const hasIndex = sitemap.includes('https://heytensor.com/answers/</loc>');

  const insertion = (hasIndex ? '' : indexEntry + '\n') + newEntries;

  // Insert before closing </urlset>
  sitemap = sitemap.replace('</urlset>', insertion + '\n</urlset>');

  fs.writeFileSync(SITEMAP_PATH, sitemap);
  console.log(`  Updated sitemap with ${newPages.length} new URLs${hasIndex ? '' : ' + answers index'}`);
}

// ============================================================
// MAIN
// ============================================================
function main() {
  console.log('HeyTensor V31 Answer Generator');
  console.log('==============================\n');

  const allGenerated = [];

  // Pattern A: Conv2d Shape Pages
  console.log('Pattern A: Conv2d Shape Calculations');
  let conv2dCount = 0;
  for (const config of CONV2D_CONFIGS) {
    const result = generateConv2dPage(config);
    if (result) {
      fs.writeFileSync(path.join(ANSWERS_DIR, result.filename), result.html);
      allGenerated.push(result);
      conv2dCount++;
      console.log(`  CREATED: ${result.filename}`);
    }
  }
  console.log(`  => ${conv2dCount} conv2d pages created\n`);

  // Pattern B: Parameter Count Pages
  console.log('Pattern B: Layer Parameter Counts');
  let paramCount = 0;
  for (const config of PARAM_CONFIGS) {
    const result = generateParamPage(config);
    if (result) {
      fs.writeFileSync(path.join(ANSWERS_DIR, result.filename), result.html);
      allGenerated.push(result);
      paramCount++;
      console.log(`  CREATED: ${result.filename}`);
    }
  }
  console.log(`  => ${paramCount} parameter pages created\n`);

  // Pattern C: Error Message Pages
  console.log('Pattern C: Error Message Fixes');
  let errorCount = 0;
  for (const config of ERROR_CONFIGS) {
    const result = generateErrorPage(config);
    if (result) {
      fs.writeFileSync(path.join(ANSWERS_DIR, result.filename), result.html);
      allGenerated.push(result);
      errorCount++;
      console.log(`  CREATED: ${result.filename}`);
    }
  }
  console.log(`  => ${errorCount} error pages created\n`);

  // Answers Index
  console.log('Generating answers/index.html');
  const indexHtml = generateAnswersIndex(allGenerated);
  fs.writeFileSync(path.join(ANSWERS_DIR, 'index.html'), indexHtml);
  console.log('  CREATED: answers/index.html\n');

  // Update sitemap
  console.log('Updating sitemap.xml');
  updateSitemap(allGenerated);

  console.log(`\nDone! Generated ${allGenerated.length} new answer pages.`);
  console.log(`  Conv2d shapes: ${conv2dCount}`);
  console.log(`  Parameter counts: ${paramCount}`);
  console.log(`  Error fixes: ${errorCount}`);
  console.log(`  + answers/index.html`);
}

main();
