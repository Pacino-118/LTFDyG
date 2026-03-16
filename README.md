**LTFDyG: A Learnable Temporal Function-based Dynamic Graph Neural Network with Dual-Channel EncodingLTFDyG：一种双通道编码的可学习时态函数动态图神经网络**
---


## Introduction   

Dynamic graph neural networks (DGNNs) have become essential for modeling evolving relational systems such as financial transaction networks and communication networks. Despite their success, most existing approaches rely on fixed-form time encoding functions, which limits their ability to capture complex temporal patterns. 

To overcome this limitation, we propose LTFDyG, a dynamic graph neural network that leverages a learnable temporal function and a dual-channel encoding architecture. The learnable temporal function combines Fourier and Spline bases to effectively model both periodic and non-periodic temporal patterns. The dual-channel encoding consists of a Time Encoding Channel that captures global temporal evolution and a Neighbor Interaction Channel that models temporally-modulated local interaction dynamics. Finally, a dual-stream Transformer architecture integrates node, edge, and temporal representations. Extensive experiments on six real-world datasets demonstrate that LTFDyG achieves strong performance across diverse dynamic graph learning tasks.

<img width="3175" height="1421" alt="LTFDyG " src="https://github.com/user-attachments/assets/b161a111-3ef7-4757-bb48-33a373366188" />

# LTFDyG Hyperparameters

## General Hyperparameters

| Category | Hyperparameter | Value |
|---------|----------------|-------|
| **Input Features** | Node feature dimension | 172 |
| | Edge feature dimension | 172 |
| **Temporal Function** | Fourier basis number | 5 |
| | Spline basis number | 5 |
| **Model Architecture** | Temporal embedding dimension | 100 |
| | Neighbor interaction dimension | 50 |
| | Projection dimension | 50 |
| | Transformer layers | 2 |
| | Attention heads | 2 |
| **Training** | Optimizer | Adam |
| | Runs | 5 |
| | Epochs | 100 |
| | Early stopping patience | 20 |
| | Loss function | Binary cross-entropy |

---

## Dataset-specific Hyperparameters

| Dataset | Batch Size | Learning Rate | Dropout | Neighbor Samples | Fusion Ratio p |
|---------|------------|---------------|---------|-----------------|----------------|
| Wikipedia | 200 | 1e-4 | 0.1 | 32 | 0.2611 |
| Reddit | 200 | 1e-4 | 0.2 | 32 | 0.1159 |
| UCI | 200 | 1e-4 | 0.1 | 32 | 0.2019 |
| Enron | 200 | 1e-4 | 0.0 | 32 | 0.0686 |
| MOOC | 200 | 1e-4 | 0.1 | 32 | 0.0982 |
| Can. Parl. | 200 | 1e-4 | 0.1 | 32 | 0.0817 |
