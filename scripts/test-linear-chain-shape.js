const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const source = fs.readFileSync('assets/js/components/shape-calculator.js', 'utf8').replace(/\r\n/g, '\n');
const linearMatch = source.match(/Linear: \{[\s\S]*?calc: function \(p\) \{([\s\S]*?)\n      \}\n    \},\n    LSTM:/);
assert(linearMatch, 'Could not find Linear layer definition');
const linearCalc = vm.runInNewContext('(function(p) {' + linearMatch[1] + '\n})');
const autoMatch = source.match(/    function autoSetInput\(type, p, shape\) \{([\s\S]*?)\n    \}/);
assert(autoMatch, 'Could not find autoSetInput function');
const autoSetInput = vm.runInNewContext('(function(type, p, shape) {' + autoMatch[1] + '\n})');

const params = { batch: 1, in_features: 512, out_features: 10 };
autoSetInput('Linear', params, [32, 128, 768]);
const result = linearCalc(params);
assert.deepStrictEqual(Array.from(result.shape), [32, 128, 10], 'Expected Linear output [32, 128, 10]');
assert(result.formula.includes('Output: [32, 128, 10]'), 'Expected formula to print full output shape');

const flattenedParams = { batch: 1, in_features: 512, out_features: 10 };
autoSetInput('Linear', flattenedParams, [1, 3136]);
assert.deepStrictEqual(Array.from(linearCalc(flattenedParams).shape), [1, 10], 'Expected flattened output [1, 10]');
