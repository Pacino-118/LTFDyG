import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LTF(nn.Module):
    """
    LTF (Learnable Temporal Function)
    --------------------------------
    A unified temporal encoding module that combines two complementary
    time-basis families:
      • A Fourier-based component for modeling periodic temporal patterns.
      • A Spline-based component for capturing smooth, nonlinear, and
        non-periodic variations.

    Given scalar timestamps of shape [batch_size, seq_len], LTF generates
    time encodings of dimension `dim`. A fraction `p` determines how many
    dimensions are assigned to the Fourier component, with the remainder
    assigned to the Spline component.

    Args:
        dim (int): Total output dimension of the temporal encoding.
        p (float): Proportion of dimensions allocated to the Fourier component.
        layer_norm (bool): Whether to apply LayerNorm to the concatenated output.
        scale (bool): Whether to apply a learnable scaling vector.
        parameter_requires_grad (bool): If False, freezes all parameters.

    Attributes:
        dim_fourier (int): Dimensions allocated to Fourier encoding.
        dim_spline (int): Dimensions allocated to Spline encoding.
        w1_fourier (nn.Linear): Linear projection preparing input for Fourier basis.
        w1_spline (nn.Linear): Linear projection preparing input for Spline basis.
        w2_fourier (FourierSeries): Fourier transformation layer.
        w2_spline (Spline): Spline transformation layer.
        layernorm (nn.LayerNorm): Optional normalization for combined output.
        scale_weight (nn.Parameter): Optional learnable scaling.
    """

    def __init__(self, dim: int, p: float = 0.5, layer_norm: bool = True,
                 scale: bool = True, min_freq: float = 1e-6, freq_reg: float = 0.0, parameter_requires_grad: bool = True):
        super().__init__()
        self.dim_fourier = math.floor(dim * 0.5)
        self.dim_spline = dim - self.dim_fourier
        self.dim = dim
        self.p = torch.clamp(torch.tensor(p, dtype=torch.float32), 0.0, 1.0)
        self.layer_norm = layer_norm
        self.scale = scale
        self.min_freq = min_freq
        self.freq_reg = freq_reg

        # Fourier Branch
        if self.dim_fourier > 0:
            base_freq = 1.0 / (10 ** np.linspace(0, 9, self.dim_fourier, dtype=np.float32))
            self.register_buffer( "base_freq",torch.from_numpy(base_freq).reshape(self.dim_fourier, 1))
            self.freq_theta = nn.Parameter(torch.zeros(self.dim_fourier, 1))
            self.bias_fourier = nn.Parameter(torch.zeros(self.dim_fourier))
            self.w2_fourier = FourierSeries(dim_fourier=self.dim_fourier,grid_size_fourier=5)
        # Spline Branch
        if self.dim_spline > 0:
            self.w1_spline = nn.Linear(1, self.dim_spline)
            spline_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_spline, dtype=np.float32))
            self.w1_spline.weight = nn.Parameter(torch.from_numpy(spline_vals).reshape(self.dim_spline, -1))
            self.w1_spline.bias = nn.Parameter(torch.zeros(self.dim_spline))
            self.w2_spline = Spline(dim_spline=self.dim_spline,grid_size_spline=5)

         # Optional Scale & LayerNorm
        if self.dim_fourier > 0 and self.dim_spline > 0:
            if self.scale_flag:
                self.scale_weight = nn.Parameter(torch.ones(dim))
            if self.layer_norm_flag:
                self.layernorm = nn.LayerNorm(dim)
        else:
            self.scale_flag = False
            self.layer_norm_flag = False

        if not parameter_requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Computes the temporal encoding for input timestamps.

        Args:
            timestamps (torch.Tensor): Shape (batch_size, seq_len), raw time values.

        Returns:
            torch.Tensor: Combined Fourier–Spline encoding of shape
                (batch_size, seq_len, dim).
        """
        timestamps = timestamps.unsqueeze(-1)
        p = self.p.to(timestamps.device)

        # Fourier branch
        if self.dim_fourier > 0:
            freq_scale = F.softplus(self.freq_theta) + self.min_freq
            freq_weight = self.base_freq * freq_scale
            proj_fourier = F.linear(timestamps,freq_weight,self.bias_fourier)
            output_fourier = self.w2_fourier(proj_fourier)
        else:
            output_fourier = torch.zeros_like(timestamps[..., :0])

        # Spline branch
        if self.dim_spline > 0:
            proj_spline = self.w1_spline(timestamps)
            output_spline = self.w2_spline(proj_spline)
        else:
            output_spline = torch.zeros_like(timestamps[..., :0])

        p_weight = p.unsqueeze(0).unsqueeze(0)

        output_fourier_weighted = output_fourier * p_weight
        output_spline_weighted = output_spline * (1 - p_weight)

        output_fourier_padded = torch.cat([
            output_fourier_weighted,
            torch.zeros_like(output_spline_weighted, device=timestamps.device)
        ], dim=-1)

        output_spline_padded = torch.cat([
            torch.zeros_like(output_fourier_weighted, device=timestamps.device),
            output_spline_weighted
        ], dim=-1)

        output = output_fourier_padded + output_spline_padded

        if self.layer_norm:
            output = self.layernorm(output)

        if self.scale:
            output = self.scale_weight * output

        return output

    def frequency_regularization(self):

        if self.dim_fourier == 0 or self.freq_reg == 0.0:
            return 0.0

        freq_scale = F.softplus(self.freq_theta)
        return self.freq_reg * torch.mean(freq_scale ** 2)


class FourierSeries(nn.Module):
    """
    Args:
        dim_fourier (int): Dimension of both the input and output features.
        grid_size_fourier (int): Number of frequency components (excluding constant term). Defaults to 5.

    Attributes:
        dim_fourier (int): Dimension of the input/output features.
        grid_size_fourier (int): Number of frequency components for the Fourier series.
        fourier_weight (torch.nn.Parameter): Learnable Fourier coefficients of shape (2, dim_fourier, dim_fourier, grid_size_fourier).
            The first index corresponds to cosine coefficients, the second to sine coefficients.
        bias (torch.nn.Parameter): Bias term of shape (dim_fourier,).
    """

    def __init__(self, dim_fourier: int, grid_size_fourier: int = 5):
        super().__init__()
        self.dim_fourier = dim_fourier
        self.grid_size_fourier = grid_size_fourier

        # fourier_weight shape: (2, dim_fourier, dim_fourier, grid_size_fourier)
        #   - 2 corresponds to cosine and sine parts, respectively
        #   - The layer outputs dim_fourier features given dim_fourier inputs
        self.fourier_weight = torch.nn.Parameter(torch.randn(2, self.dim_fourier, self.dim_fourier, grid_size_fourier) /
                                                (np.sqrt(self.dim_fourier) * np.sqrt(self.grid_size_fourier)))

        # Bias term, one per output dimension
        self.bias = nn.Parameter(torch.zeros(self.dim_fourier))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        out_shape = original_shape[0:-1] + (self.dim_fourier,)

        # Flatten everything except the last dim
        x = x.reshape(-1, self.dim_fourier)  # (N, dim_fourier)

        # Frequency indices k = 1..grid_size_fourier
        k = torch.arange(1, self.grid_size_fourier + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.grid_size_fourier)  # shape: (1,1,1,K)

        # Reshape x to broadcast with k
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)  # (N,1,dim_fourier,1)

        # Compute cos(k * x) and sin(k * x)
        c = torch.cos(k * x_reshaped)  # (N,1,dim_fourier,K)
        s = torch.sin(k * x_reshaped)  # (N,1,dim_fourier,K)

        # Sum up the contributions with the learned Fourier coefficients
        # fourier_weight[0:1] -> cosine part, fourier_weight[1:2] -> sine part
        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))

        # Add bias
        y += self.bias

        # Reshape back to original shape
        y = y.reshape(out_shape)
        return y


class Spline(nn.Module):
    """
    Spline
    ------
    A learnable B-spline transformation layer designed to model smooth,
    nonlinear, and non-periodic temporal patterns. Unlike the Fourier basis,
    spline bases adapt more flexibly to local variations in time.

    Args:
        dim_spline (int): Input/output dimensionality.
        grid_size_spline (int): Number of internal spline knots.
        order_spline (int): Spline order (degree).
        grid_range (list): Numerical range of the spline grid.

    Attributes:
        grid (Tensor): Precomputed knot grid.
        base_weight (nn.Parameter): Linear transformation in the base branch.
        spline_weight (nn.Parameter): Learnable spline basis coefficients.
    """

    def __init__(self, dim_spline: int, grid_size_spline: int = 5, order_spline: int = 3, grid_range: list = [-1, 1]):
        super().__init__()
        self.dim_spline = dim_spline
        self.grid_size_spline = grid_size_spline
        self.order_spline = order_spline

        # Compute grid spacing
        h = (grid_range[1] - grid_range[0]) / float(self.grid_size_spline)

        # Build the grid for each dimension
        grid = torch.arange(-self.order_spline, self.grid_size_spline + self.order_spline + 1)
        grid = grid * h + grid_range[0]
        grid = grid.expand(self.dim_spline, -1).contiguous()
        self.register_buffer("grid", grid)

        # Base weight for the linear+activation branch
        self.base_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline))

        # Spline coefficients
        self.spline_weight = nn.Parameter(
            torch.Tensor(self.dim_spline, self.dim_spline, self.grid_size_spline + self.order_spline))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.dim_spline)  # flatten all but the last dimension

        # Base branch: tanh + linear
        base_output = nn.functional.linear(torch.tanh(x), self.base_weight)  # shape: (N, dim_spline)

        # If the input batch is empty, return a zero-like tensor
        if x.size(0) == 0:
            spline_output = torch.zeros_like(base_output)
        else:
            # Evaluate B-spline basis
            # b_splines(x).shape -> (N, dim_spline, grid_size_spline + order_spline)
            # Flatten for linear
            b_splines_val = self.b_splines(x).view(x.size(0), -1)

            # Reshape spline_weight for linear operation
            w = self.spline_weight.view(self.dim_spline, -1)
            spline_output = nn.functional.linear(b_splines_val, w)

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.dim_spline)
        return output

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # A: (dim_spline, batch_size, grid_size_spline + order_spline)
        A = self.b_splines(x).transpose(0, 1)
        # B: (dim_spline, batch_size, dim_spline)
        B = y.transpose(0, 1)

        # Compute pseudo-inverse of A
        A_pinv = torch.pinverse(A)  # shape: (dim_spline, grid_size_spline+order_spline, batch_size)

        # Solve the least squares problem using the pseudo-inverse
        # shape: (dim_spline, grid_size_spline+order_spline, batch_size)
        solution = torch.bmm(A_pinv, B)

        # Permute to (batch_size, dim_spline, grid_size_spline+order_spline) -> (dim_spline, dim_spline, grid_size_spline+order_spline)
        result = solution.permute(2, 0, 1).contiguous()
        return result

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # grid: (dim_spline, grid_size_spline + 2*order_spline + 1)
        grid = self.grid
        x = x.unsqueeze(-1)  # shape: (N, dim_spline, 1)

        # bases will mark where x lies between [grid_i, grid_{i+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # shape: (N, dim_spline, ?)

        # Recursively elevate the basis to higher spline orders
        for k in range(1, self.order_spline + 1):
            # Segment lengths in the left and right directions
            left_num = (x - grid[:, :-(k + 1)])
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)])

            right_num = (grid[:, k + 1:] - x)
            right_den = (grid[:, k + 1:] - grid[:, 1:-k])

            bases = (left_num / left_den) * bases[:, :, :-1] + (right_num / right_den) * bases[:, :, 1:]

        return bases.contiguous()


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim #time_dim的维度是100
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0, 1), inputs_key.transpose(0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs



