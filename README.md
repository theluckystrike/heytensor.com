# Tensor Shape Calculator — Debug Neural Network Dimensions Layer by Layer

**[Calculate Shapes →](https://heytensor.com)** | [About](https://heytensor.com/about.html) | [Blog](https://heytensor.com/blog/)

Tensor Shape Calculator helps deep learning engineers debug and plan neural network architectures by computing output tensor shapes layer by layer. Supports 10+ layer types including Conv2D, MaxPool, BatchNorm, Linear, LSTM, and Transformer blocks. Chain layers together to trace shapes through your entire network. Catches dimension mismatches before you run a single training step.

## Features

- **10+ supported layer types** — Conv2D, MaxPool, BatchNorm, Linear, LSTM, Attention, and more
- **Chain mode** — stack layers sequentially and trace tensor shapes through the full network
- **Dimension error debugger** — catch shape mismatches and get fix suggestions instantly
- **Architecture presets** — start from ResNet, VGG, Transformer, or custom templates
- **Parameter count estimation** — see trainable parameters for each layer and total network

## How It Works

Enter your input tensor shape (e.g., batch x channels x height x width) and select a layer type. Configure the layer parameters — kernel size, stride, padding, output features — and the calculator instantly shows the output shape. In chain mode, each layer's output becomes the next layer's input, letting you trace shapes through an entire architecture. If a dimension mismatch occurs, the debugger highlights the issue and suggests fixes.

## Built With

- Vanilla JavaScript (no frameworks, no dependencies)
- Client-side only — your data never leaves your browser
- Part of the [Zovo Tools](https://zovo.one) open network

## Related Tools

- [Matrix Calculator](https://ml3x.com) — compute matrix operations used in neural network layers
- [ML Model Comparison](https://lockml.com) — compare architectures and benchmarks across models
- [ML Cheat Sheet Generator](https://ml0x.com) — quick-reference formulas for deep learning math

## Contributing

Found a bug or have a feature request? [Open an issue](https://github.com/theluckystrike/heytensor.com/issues).

## License

MIT
