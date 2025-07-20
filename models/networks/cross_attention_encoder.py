# File: models/networks/cross_attention_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming your custom CrossAttention class is accessible,
# e.g., if it's in models/networks/attention.py, you'd do:
# from .attention import CrossAttention
# For simplicity, I'll assume CrossAttention can be imported from your project's structure,
# or you can copy-paste the CrossAttention class here for self-containment if preferred.
# For this example, let's assume it's in the same directory for clarity of import.

# Assuming the custom CrossAttention class is available in the scope or imported
# (You might need to adjust the import path based on your exact file structure)
# For demonstration purposes, I will include the CrossAttention class here for completeness.
# In your actual project, ensure you import it from where it's defined.

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1) # Output dimension matches query input dimension

    def forward(self, x1, x2, _, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output, 0 # Return output and a dummy value (original CrossAttention returns output, 0)


class CrossAttentionEncoder(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads, num_layers, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            # Each layer is an instance of your custom CrossAttention
            CrossAttention(in_dim1, in_dim2, k_dim, v_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # Layer norm applied to the 'query' side's dimension (in_dim1)
        self.norm = nn.LayerNorm(in_dim1) 

    def forward(self, query_features, kv_features):
        x = query_features
        for layer in self.layers:
            # Each layer takes x (query) and kv_features (key/value)
            # The output of CrossAttention matches the query's input dimension (in_dim1)
            x_attn, _ = layer(x, kv_features, _=None)
            # Apply residual connection, dropout, and layer normalization
            x = self.norm(x + self.dropout(x_attn)) 
        return x