# LTFDyG
**LTFDyG: A Learnable Temporal Function-based Dynamic Graph Neural Network with Dual-Channel Encoding**

---

## Introduction

Dynamic graph neural networks (DGNNs) are widely used to model evolving relational systems such as **financial transaction networks** and **communication networks**. However, many existing methods rely on fixed-form time encoding functions, which limits their ability to capture complex temporal patterns.

To address this limitation, we propose **LTFDyG**, a dynamic graph neural network based on a **learnable temporal function** and a **dual-channel encoding architecture**.

The key ideas of LTFDyG include:

- **Learnable Temporal Function** that combines **Fourier** and **Spline bases** to capture both periodic and non-periodic temporal patterns.
- **Dual-channel temporal encoding**, including:
  - **Time Encoding Channel** for modeling global temporal evolution.
  - **Neighbor Interaction Channel** for capturing temporally-modulated local interaction dynamics.
- **Dual-stream Transformer architecture** that integrates node, edge, and temporal representations.

Experiments on six real-world datasets demonstrate that LTFDyG achieves strong performance on dynamic graph learning tasks.

