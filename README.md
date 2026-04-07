# HeyTensor

> Free PyTorch tensor shape calculator, layer output debugger, and deep learning architecture explorer.

**[Use HeyTensor live](https://heytensor.com/)**

HeyTensor is a browser-based tensor shape calculator for deep learning practitioners. Compute output shapes for 14+ PyTorch and TensorFlow layer types, chain layers together to trace shapes through entire architectures, and debug PyTorch RuntimeErrors by pasting real stack traces. Built for ML engineers, researchers, and students who want instant answers without spinning up a Python kernel.

## Features

- Single-layer shape calculator for 14+ PyTorch layer types (Conv2d, Linear, LSTM, GRU, MultiheadAttention, BatchNorm, MaxPool2d, Flatten, Embedding, ConvTranspose2d, and more)
- Chain mode for tracing tensor shapes through multi-layer neural network architectures
- Paste Error mode that parses PyTorch RuntimeErrors and suggests targeted fixes
- Architecture presets for LeNet-5, ResNet blocks, and Transformer encoders
- Detailed formula display for every output dimension calculation
- Model parameter counter and GPU memory estimator
- Activation, loss function, and optimizer reference library
- Einsum notation calculator for advanced tensor operations
- 100% client-side — no data leaves your browser
- MIT licensed
- No signup, no tracking

## Tools

| Tool | Description |
|------|-------------|
| [Conv2d Calculator](https://heytensor.com/tools/conv2d-calculator.html) | Calculate Conv2d output shapes with stride, padding, and dilation |
| [Conv1d Calculator](https://heytensor.com/tools/conv1d-calculator.html) | Calculate Conv1d output shapes for sequence models |
| [ConvTranspose2d Calculator](https://heytensor.com/tools/convtranspose2d-calculator.html) | Transposed convolution output shapes for decoders |
| [Linear Layer Calculator](https://heytensor.com/tools/linear-layer-calculator.html) | nn.Linear input and output dimension checker |
| [LSTM Shape Calculator](https://heytensor.com/tools/lstm-shape-calculator.html) | LSTM and GRU output shapes with hidden state dimensions |
| [MultiHead Attention Calculator](https://heytensor.com/tools/multihead-attention-calculator.html) | Transformer attention layer output shapes |
| [MaxPool2d Calculator](https://heytensor.com/tools/maxpool2d-calculator.html) | Max pooling output shape calculator |
| [BatchNorm Calculator](https://heytensor.com/tools/batchnorm-calculator.html) | Batch normalization shape checker |
| [Flatten Calculator](https://heytensor.com/tools/flatten-calculator.html) | Flatten operation output shape |
| [Embedding Calculator](https://heytensor.com/tools/embedding-calculator.html) | nn.Embedding output shape for NLP models |
| [mat1 and mat2 Shapes Fix](https://heytensor.com/tools/mat1-mat2-shapes.html) | Debug matrix multiplication shape errors |
| [Shape Mismatch Debugger](https://heytensor.com/tools/shape-mismatch.html) | Diagnose tensor shape mismatch runtime errors |
| [view size not compatible Fix](https://heytensor.com/tools/view-size-not-compatible.html) | Fix PyTorch view and reshape errors |
| [CUDA Out of Memory](https://heytensor.com/tools/cuda-out-of-memory.html) | Diagnose and resolve GPU OOM errors |
| [Einsum Calculator](https://heytensor.com/tools/einsum-calculator.html) | Einstein summation notation helper |
| [Activation Functions](https://heytensor.com/tools/activation-functions.html) | ReLU, GELU, Sigmoid, Tanh, and more reference |
| [Loss Functions](https://heytensor.com/tools/loss-functions.html) | Cross-entropy, MSE, and other loss function reference |
| [Optimizers](https://heytensor.com/tools/optimizers.html) | Adam, SGD, AdamW optimizer comparison |
| [Parameter Counter](https://heytensor.com/tools/parameter-counter.html) | Count trainable parameters in your model |
| [Memory Calculator](https://heytensor.com/tools/memory-calculator.html) | Estimate GPU memory usage for training |

## Research

- [The 20 Most Common PyTorch Errors](https://heytensor.com/research/most-common-pytorch-errors.html) — Common PyTorch errors ranked by frequency with fixes
- [PyTorch Error Database](https://heytensor.com/research/pytorch-error-database.html) — 50+ real PyTorch errors analyzed from Stack Overflow
- [PyTorch Error Statistics](https://heytensor.com/research/pytorch-error-stats.html) — What goes wrong most often in PyTorch training

## Tech Stack

- Pure HTML, CSS, and vanilla JavaScript
- No build step
- No external dependencies (except Google Fonts on some pages)
- Hosted on GitHub Pages with Cloudflare CDN

## Part of Zovo Tools

HeyTensor is part of [Zovo Tools](https://zovo.one/tools) — free developer tools by a solo developer. No tracking, no signup, no nonsense.

**Other tools in the network:**

- [EpochPilot](https://epochpilot.com) — Timestamp, timezone, and cron tools
- [KappaKit](https://kappakit.com) — Developer toolkit (Base64, JWT, hash, regex)
- [LochBot](https://lochbot.com) — Prompt injection vulnerability checker
- [ABWex](https://abwex.com) — A/B test significance calculator
- [KickLLM](https://kickllm.com) — LLM cost calculator
- [Gen8X](https://gen8x.com) — Color palette generator with WCAG checks
- [GPT0X](https://gpt0x.com) — AI model database
- [ML3X](https://ml3x.com) — Matrix calculator
- [ML0X](https://ml0x.com) — Machine learning cheat sheet generator
- [Enhio](https://enhio.com) — Text enhancement utilities
- [Krzen](https://krzen.com) — Image compression
- [Kappafy](https://kappafy.com) — JSON formatter and explorer
- [LockML](https://lockml.com) — Open source ML model comparison
- [InvokeBot](https://invokebot.com) — Webhook testing
- [ClaudHQ](https://claudhq.com) — Claude prompt library
- [ClaudKit](https://claudkit.com) — Claude API utilities
- [ClaudFlow](https://claudflow.com) — AI workflow builder

## License

MIT licensed.

## Contact

Built and maintained by [Michael Lip](https://zovo.one). For questions or feedback: support@zovo.one
