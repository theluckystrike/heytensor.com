# HeyTensor -- PyTorch Tensor Shape Calculator & Error Debugger

**[-> Use HeyTensor (live tool)](https://heytensor.com/)**

HeyTensor is a free tensor shape calculator for deep learning practitioners. Calculate output shapes for 14 PyTorch and TensorFlow layer types, chain layers together to trace shapes through entire architectures, debug shape mismatch errors by pasting PyTorch RuntimeErrors, and explore architecture presets like LeNet-5, ResNet, and Transformers.

## Features

- Single-layer shape calculator for 14 layer types (Conv2d, Linear, LSTM, etc.)
- Chain mode for tracing shapes through multi-layer architectures
- Paste Error mode that parses PyTorch RuntimeErrors and suggests fixes
- Architecture presets: LeNet-5, ResNet Block, Transformer encoder
- Supports Conv2d, Conv1d, Linear, LSTM, GRU, MultiheadAttention, and more
- Detailed formula display for every output dimension calculation
- 100% client-side -- your data never leaves your browser
- Open source -- inspect the code yourself

## Tools

### Layer Shape Calculators
- [Conv2d Calculator](https://heytensor.com/tools/conv2d-calculator.html) -- Calculate Conv2d output shapes
- [Conv1d Calculator](https://heytensor.com/tools/conv1d-calculator.html) -- Calculate Conv1d output shapes
- [ConvTranspose2d Calculator](https://heytensor.com/tools/convtranspose2d-calculator.html) -- Transposed convolution shapes
- [Linear Layer Calculator](https://heytensor.com/tools/linear-layer-calculator.html) -- nn.Linear output shapes
- [LSTM Shape Calculator](https://heytensor.com/tools/lstm-shape-calculator.html) -- LSTM and GRU output shapes
- [MultiHead Attention Calculator](https://heytensor.com/tools/multihead-attention-calculator.html) -- Attention layer shapes
- [MaxPool2d Calculator](https://heytensor.com/tools/maxpool2d-calculator.html) -- Max pooling output shapes
- [BatchNorm Calculator](https://heytensor.com/tools/batchnorm-calculator.html) -- Batch normalization shapes
- [Flatten Calculator](https://heytensor.com/tools/flatten-calculator.html) -- Flatten operation output shapes
- [Embedding Calculator](https://heytensor.com/tools/embedding-calculator.html) -- nn.Embedding output shapes

### Error Debuggers
- [mat1 and mat2 Shapes Fix](https://heytensor.com/tools/mat1-mat2-shapes.html) -- Fix matrix multiplication errors
- [Shape Mismatch Debugger](https://heytensor.com/tools/shape-mismatch.html) -- Debug tensor shape mismatches
- [view size not compatible Fix](https://heytensor.com/tools/view-size-not-compatible.html) -- Fix view/reshape errors
- [CUDA Out of Memory](https://heytensor.com/tools/cuda-out-of-memory.html) -- Debug GPU memory errors

### Reference Tools
- [Einsum Calculator](https://heytensor.com/tools/einsum-calculator.html) -- Einstein summation notation helper
- [Activation Functions](https://heytensor.com/tools/activation-functions.html) -- Activation function reference
- [Loss Functions](https://heytensor.com/tools/loss-functions.html) -- Loss function reference and guide
- [Optimizers](https://heytensor.com/tools/optimizers.html) -- Optimizer comparison and reference
- [Parameter Counter](https://heytensor.com/tools/parameter-counter.html) -- Count model parameters
- [Memory Calculator](https://heytensor.com/tools/memory-calculator.html) -- Estimate model GPU memory usage

## Tech Stack

- Vanilla JavaScript (no frameworks, no build step)
- Static HTML hosted on GitHub Pages
- Cloudflare DNS + SSL
- Zero dependencies, zero tracking, zero cookies

## Part of Zovo Tools

HeyTensor is part of [Zovo Tools](https://zovo.one/tools) -- a collection of free developer tools.

**Other tools in the network:**
- [EpochPilot](https://epochpilot.com) -- 30+ timestamp and timezone tools
- [LochBot](https://lochbot.com) -- Prompt injection vulnerability checker
- [KappaKit](https://kappakit.com) -- Developer toolkit (Base64, JWT, hash, UUID)
- [ML3X](https://ml3x.com) -- Matrix calculator with step-by-step solutions

## License

MIT
