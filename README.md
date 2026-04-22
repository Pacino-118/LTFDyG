**LTFDyG: A Learnable Temporal Function-based Dynamic Graph Neural Network with Dual-Channel Encoding****LTFDyG：基于可学习时态函数的双通道编码动态图神经网络LTFDyG：一种双通道编码的可学习时态函数动态图神经网络**
---


## Introduction      # #的介绍   介绍   # #的介绍

Dynamic graph neural networks (DGNNs) have become essential for modeling evolving relational systems such as financial transaction networks and communication networks. Despite their success, most existing approaches rely on fixed-form time encoding functions, which limits their ability to capture complex temporal patterns. 动态图神经网络（dgnn）已经成为建模不断发展的关系系统，如金融交易网络和通信网络至关重要。尽管它们取得了成功，但大多数现有的方法依赖于固定形式的时间编码函数，这限制了它们捕获复杂时间模式的能力。动态图神经网络（dgnn）已经成为建模不断发展的关系系统，如金融交易网络和通信网络至关重要。尽管它们取得了成功，但大多数现有的方法依赖于固定形式的时间编码函数，这限制了它们捕获复杂时间模式的能力。

To overcome this limitation, we propose LTFDyG, a dynamic graph neural network that leverages a learnable temporal function and a dual-channel encoding architecture. The learnable temporal function combines Fourier and Spline bases to effectively model both periodic and non-periodic temporal patterns. The dual-channel encoding consists of a Time Encoding Channel that captures global temporal evolution and a Neighbor Interaction Channel that models temporally-modulated local interaction dynamics. Finally, a dual-stream Transformer architecture integrates node, edge, and temporal representations. Extensive experiments on six real-world datasets demonstrate that LTFDyG achieves strong performance across diverse dynamic graph learning tasks.为了克服这一限制，我们提出了LTFDyG，这是一种利用可学习时间函数和双通道编码架构的动态图神经网络。可学习的时间函数结合了傅里叶和样条基来有效地模拟周期和非周期时间模式。双通道编码由捕获全局时间演化的时间编码通道和模拟时间调制的局部交互动态的邻居交互通道组成。最后，双流Transformer架构集成了节点、边缘和时态表示。在六个真实数据集上的大量实验表明，LTFDyG在不同的动态图学习任务中取得了较好的性能。为了克服这一限制，我们提出了LTFDyG，这是一种利用可学习时间函数和双通道编码架构的动态图神经网络。可学习的时间函数结合了傅里叶和样条基来有效地模拟周期和非周期时间模式。双通道编码由捕获全局时间演化的时间编码通道和模拟时间调制的局部交互动态的邻居交互通道组成。最后，双流Transformer架构集成了节点、边缘和时态表示。在六个真实数据集上的大量实验表明，LTFDyG在不同的动态图学习任务中取得了较好的性能。

<<img width   宽度="3175" height="1421" alt="LTFDyG " src="https://github.com/user-attachments/assets/b161a111-3ef7-4757-bb48-33a373366188" />img width="3175" height="1421" alt="LTFDyG " src="https://github.com/user-attachments/assets/b161a111-3ef7-4757-bb48-33a373366188" />

# LTFDyG Hyperparameters

####通用超参数 General Hyperparameters

||类别|超参数|值| Category | Hyperparameter | Value |
|---------|----------------|-------|
| | **输入特征** |节点特征维度| 172 |**Input Features   输入特性** | Node feature dimension | 172 |
| | Edge feature dimension | 172 |
| **Temporal Function   时间函数** | Fourier basis number | 5 |
| | Spline basis number | 5 |
| **Model Architecture   模型架构** | Temporal embedding dimension | 100 |
| | Neighbor interaction dimension | 50 |
| | Projection dimension | 50 |
| | Transformer layers | 2 |
| | Attention heads | 2 |
| **Training   培训** | Optimizer | Adam |
| | Runs | 5 |
| | Epochs | 100 |
| | Early stopping patience | 20 |
| | Loss function | Binary cross-entropy |

---

##特定于数据集的超参数 Dataset-specific Hyperparameters

||数据集| Batch Size |学习率| Dropout | Neighbor Samples | Fusion Ratio p | Dataset | Batch Size | Learning Rate | Dropout | Neighbor Samples | Fusion Ratio p |
|---------|------------|---------------|---------|-----------------|----------------|
| Wikipedia | 200 | 1e-4 | 0.1 | 32 | 0.2611 ||维基百科| 200 | 1e-4 | 0.1 | 32 | 0.2611 |
| Reddit | 200 | 1e-4 | 0.2 | 32 | 0.1159 |
| UCI | 200 | 1e-4 | 0.1 | 32 | 0.2019 |
| Enron | 200 | 1e-4 | 0.0 | 32 | 0.0686 |
| MOOC | 200 | 1e-4 | 0.1 | 32 | 0.0982 |
| Can. Parl. | 200 | 1e-4 | 0.1 | 32 | 0.0817 |


