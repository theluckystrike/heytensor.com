#!/usr/bin/env node
/**
 * HeyTensor V31 Answer Page Validator
 * Validates all generated answer pages for quality and correctness.
 */

const fs = require('fs');
const path = require('path');

const ANSWERS_DIR = path.join(__dirname, '..', 'answers');
const MIN_WORD_COUNT = 400;

// V30 files are exempt from new validation rules (they existed before V31)
const V30_FILES = new Set([
  'batch-norm-input-shape.html', 'cannot-broadcast-tensors.html',
  'conv2d-output-224x224-kernel-3.html', 'conv2d-output-224x224-kernel-7-stride-2.html',
  'conv2d-output-32x32-kernel-5.html', 'cuda-out-of-memory-fix.html',
  'difference-between-relu-and-gelu.html', 'expected-4d-input-got-3d.html',
  'flatten-after-conv2d.html', 'gradient-is-none-fix.html',
  'how-many-parameters-bert-base.html', 'how-many-parameters-resnet50.html',
  'linear-layer-512-to-10.html', 'lstm-output-shape-bidirectional.html',
  'mat1-mat2-shapes-cannot-be-multiplied.html', 'maxpool2d-output-size.html',
  'no-module-named-torch.html', 'pytorch-vs-tensorflow-2026.html',
  'resnet50-layer-shapes.html', 'runtime-error-expected-scalar-type-float.html',
  'transformer-encoder-input-shape.html', 'vgg16-layer-shapes.html',
  'view-size-not-compatible.html', 'what-is-padding-same-pytorch.html',
  'what-is-stride-in-conv2d.html',
]);

let errors = 0;
let warnings = 0;
let passed = 0;

function countWords(html) {
  // Strip HTML tags and count words
  const text = html.replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z]+;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return text.split(' ').filter(w => w.length > 0).length;
}

function validate(filename) {
  const filepath = path.join(ANSWERS_DIR, filename);
  const html = fs.readFileSync(filepath, 'utf8');
  const issues = [];
  const isV30 = V30_FILES.has(filename);
  const isIndex = filename === 'index.html';

  // 1. Minimum word count (skip V30 and index)
  const words = countWords(html);
  if (!isV30 && !isIndex && words < MIN_WORD_COUNT) {
    issues.push(`FAIL: Only ${words} words (minimum ${MIN_WORD_COUNT})`);
  }

  // 2. Required elements
  if (!html.includes('<title>') || !html.includes('| HeyTensor</title>')) {
    issues.push('FAIL: Missing or malformed title tag');
  }
  if (!html.includes('Hey<span>Tensor</span>')) {
    issues.push('FAIL: Missing HeyTensor logo');
  }
  if (!html.includes('IBM Plex Sans') && !html.includes('IBM+Plex+Sans')) {
    issues.push('FAIL: Missing Google Fonts (IBM Plex Sans)');
  }
  if (!html.includes('Space Mono') && !html.includes('Space+Mono')) {
    issues.push('FAIL: Missing Google Fonts (Space Mono)');
  }
  if (!html.includes('breadcrumb')) {
    issues.push('FAIL: Missing breadcrumb navigation');
  }
  if (!html.includes('<h1>')) {
    issues.push('FAIL: Missing H1');
  }
  if (!isIndex && !html.includes('FAQPage')) {
    issues.push('FAIL: Missing FAQPage JSON-LD');
  }
  if (!html.includes('zovo-network')) {
    issues.push('FAIL: Missing Zovo network footer');
  }
  if (!html.includes('site-footer')) {
    issues.push('FAIL: Missing site footer');
  }
  if (!html.includes('canonical')) {
    issues.push('FAIL: Missing canonical URL');
  }
  if (!html.includes('og:title')) {
    issues.push('FAIL: Missing Open Graph meta');
  }
  if (!isIndex && !html.includes('twitter:card')) {
    issues.push('FAIL: Missing Twitter Card meta');
  }

  // 3. No template artifacts
  const artifacts = ['{{', '}}', 'undefined', 'NaN', 'TODO', 'FIXME', 'PLACEHOLDER'];
  for (const art of artifacts) {
    if (html.includes(art)) {
      issues.push(`FAIL: Template artifact found: "${art}"`);
    }
  }

  // 4. Unique title check (done at collection level)

  // 5. Valid HTML structure
  if (!html.includes('<!DOCTYPE html>')) {
    issues.push('FAIL: Missing DOCTYPE');
  }
  if (!html.includes('</html>')) {
    issues.push('FAIL: Missing closing </html>');
  }

  // 6. Code examples present
  if (!html.includes('<pre><code>')) {
    issues.push('WARN: No code examples found');
  }

  // 7. Related questions section
  if (!html.includes('Related Questions') && !html.includes('All Questions')) {
    issues.push('WARN: Missing related questions section');
  }

  // 8. CTA button
  if (!html.includes('background:var(--accent)') && !html.includes('Try the')) {
    issues.push('WARN: Missing CTA button');
  }

  // 9. Conv2d math verification
  if (filename.startsWith('conv2d-output-')) {
    const titleMatch = html.match(/(\d+)&times;(\d+) Input/i) || html.match(/(\d+)x(\d+) Input/i);
    if (titleMatch) {
      const kernelMatch = html.match(/kernel_size=(\d+)/);
      const strideMatch = html.match(/stride=(\d+)/);
      const paddingMatch = html.match(/padding=(\d+)/);
      if (kernelMatch && strideMatch && paddingMatch) {
        const input = parseInt(titleMatch[1]);
        const kernel = parseInt(kernelMatch[1]);
        const stride = parseInt(strideMatch[1]);
        const padding = parseInt(paddingMatch[1]);
        const expected = Math.floor((input + 2 * padding - kernel) / stride) + 1;
        // Look for the output in the bold answer box (more reliable than step-by-step)
        const outputMatch = html.match(/outputs (\d+)&times;(\d+)/);
        if (outputMatch) {
          const stated = parseInt(outputMatch[1]);
          if (stated !== expected) {
            issues.push(`FAIL: Math error! Conv2d(${input}, k=${kernel}, s=${stride}, p=${padding}) should output ${expected}, page says ${stated}`);
          }
        }
      }
    }
  }

  // 10. Parameter count verification
  if (filename.startsWith('parameters-')) {
    if (filename.includes('linear')) {
      const inMatch = html.match(/in_features \* out_features.*?= (\d[\d,]*) \+ (\d[\d,]*)/);
      if (inMatch) {
        // Verify the addition
        const weights = parseInt(inMatch[1].replace(/,/g, ''));
        const bias = parseInt(inMatch[2].replace(/,/g, ''));
        const totalMatch = html.match(/has ([\d,]+) trainable parameters/);
        if (totalMatch) {
          const stated = parseInt(totalMatch[1].replace(/,/g, ''));
          if (stated !== weights + bias) {
            issues.push(`FAIL: Parameter math error! ${weights} + ${bias} = ${weights + bias}, page says ${stated}`);
          }
        }
      }
    }
  }

  if (issues.length === 0) {
    passed++;
    return { filename, status: 'PASS', words, issues: [] };
  } else {
    const fails = issues.filter(i => i.startsWith('FAIL'));
    const warns = issues.filter(i => i.startsWith('WARN'));
    errors += fails.length;
    warnings += warns.length;
    if (fails.length > 0) {
      return { filename, status: 'FAIL', words, issues };
    } else {
      passed++;
      return { filename, status: 'WARN', words, issues };
    }
  }
}

// ============================================================
// MAIN
// ============================================================
console.log('HeyTensor V31 Answer Page Validator');
console.log('===================================\n');

const files = fs.readdirSync(ANSWERS_DIR).filter(f => f.endsWith('.html'));
const results = [];
const titles = new Map();

for (const file of files) {
  const result = validate(file);
  results.push(result);

  // Collect titles for uniqueness check
  const html = fs.readFileSync(path.join(ANSWERS_DIR, file), 'utf8');
  const titleMatch = html.match(/<title>([^<]+)<\/title>/);
  if (titleMatch) {
    const title = titleMatch[1];
    if (titles.has(title)) {
      console.log(`  FAIL: Duplicate title "${title}" in ${file} and ${titles.get(title)}`);
      errors++;
    }
    titles.set(title, file);
  }
}

// Print results
let failCount = 0;
for (const r of results) {
  if (r.status === 'FAIL') {
    failCount++;
    console.log(`  FAIL: ${r.filename} (${r.words} words)`);
    for (const issue of r.issues) {
      console.log(`    ${issue}`);
    }
  }
}

if (failCount === 0) {
  console.log('All pages passed validation!\n');
}

// Summary
console.log(`\nResults:`);
console.log(`  Total files: ${files.length}`);
console.log(`  Passed: ${passed}`);
console.log(`  Errors: ${errors}`);
console.log(`  Warnings: ${warnings}`);

// Word count stats
const wordCounts = results.map(r => r.words);
console.log(`\nWord counts:`);
console.log(`  Min: ${Math.min(...wordCounts)}`);
console.log(`  Max: ${Math.max(...wordCounts)}`);
console.log(`  Avg: ${Math.round(wordCounts.reduce((a, b) => a + b) / wordCounts.length)}`);

// List any under minimum
const under = results.filter(r => r.words < MIN_WORD_COUNT);
if (under.length > 0) {
  console.log(`\nPages under ${MIN_WORD_COUNT} words:`);
  for (const r of under) {
    console.log(`  ${r.filename}: ${r.words} words`);
  }
}

process.exit(errors > 0 ? 1 : 0);
