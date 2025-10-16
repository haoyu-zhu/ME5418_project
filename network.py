import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class EncoderLayer(nn.Module):
    """
    单层 Encoder: Multi-Head Attention + FeedForward，两次残差连接。
    h1 = x + norm(attn(x))
    h2 = h1 + norm(ff(h1))
    """
    def __init__(self, embed_dim=128, n_heads=8, ff_hidden_dim=512, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ff_hidden_dim = ff_hidden_dim

        # 多头注意力
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)

        # 前馈层
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        # 归一化层（默认 LayerNorm，论文里 encoder 用 BatchNorm 也可切换）
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        # 归一化层（BatchNorm）
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)



    def forward(self, x):
        # Multi-Head Attention + 残差连接
        attn_out, _ = self.mha(x, x, x)
        #h1 = x + self.norm1(attn_out)
        h1 = self.bn1((x + attn_out).transpose(1, 2)).transpose(1, 2)

        # FeedForward + 残差连接
        ff_out = self.ff(h1)
        #h2 = h1 + self.norm2(ff_out)
        h2 = self.bn2((h1 + ff_out).transpose(1, 2)).transpose(1, 2)

        return h2


class GraphAttentionEncoder(nn.Module):
    """
    整个 Encoder 堆叠结构:
      输入 node_dim=3 → Linear 投射到 embed_dim=128
      堆叠 n_encode_layers 层 EncoderLayer
      输出 node_emb, graph_emb
    """
    def __init__(self, n_heads=8, embed_dim=128, n_encode_layers=3, node_dim=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(node_dim, embed_dim)  # (x, y, reward) → 128
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, ff_hidden_dim=512)
            for _ in range(n_encode_layers)
        ])

    def forward(self, x):
        """
        x: [B, N, node_dim]   -> 每个点的特征 (x, y, reward)
        return:
          node_emb: [B, N, D]
          graph_emb: [B, D] (mean pooling)
        """
        # Step 1: 线性映射到128维
        h = self.input_proj(x)  # [B, N, D]

        # Step 2: 堆叠多层注意力+前馈网络
        for layer in self.layers:
            h = layer(h)

        # Step 3: 图级embedding（所有节点取平均）
        graph_emb = h.mean(dim=1)

        return h, graph_emb



class Glimpse(nn.Module):
    """
    多头 glimpse：用 context 生成单个 query，与所有节点的 key/value 做注意力。
    不包含残差/归一化/FF（按论文做法）。
    """
    def __init__(self, embed_dim: int = 128, n_heads: int = 8):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dk = embed_dim // n_heads

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,
                ctx: torch.Tensor,       # [B, D]   (由 [h_g, h_{t-1}, D_t] 线性映射得到)
                node_emb: torch.Tensor,  # [B, N, D]
                feas_mask: Optional[torch.Tensor] = None  # [B, N] bool, True=可选
                ) -> torch.Tensor:       # 返回 refined context: [B, D]
        B, N, D = node_emb.shape
        # 1) 生成 q/k/v
        q = self.Wq(ctx).view(B, self.n_heads, self.dk)                       # [B,H,dk]
        k = self.Wk(node_emb).view(B, N, self.n_heads, self.dk).transpose(1, 2)  # [B,H,N,dk]
        v = self.Wv(node_emb).view(B, N, self.n_heads, self.dk).transpose(1, 2)  # [B,H,N,dk]

        # 2) 点积注意力 logits: [B,H,N]
        logits = torch.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(self.dk)
        if feas_mask is not None:
            logits = logits.masked_fill(~feas_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(logits, dim=-1)                                      # [B,H,N]
        head_ctx = torch.einsum("bhn,bhnd->bhd", attn, v)                     # [B,H,dk]
        out = head_ctx.reshape(B, D)                                          # [B,D]
        return self.Wo(out)                                                   # [B,D]


class OPDecoder(nn.Module):
    """
    按论文的解码器：
      C_t = [h_g, h_{t-1}, D_t]  --(Linear)--> ctx
      \tilde{C}_t = glimpse(ctx)
      u_tj = C * tanh( (q^T k_j)/sqrt(D) )     # 单头
      logits = mask(u_tj, infeasible=-inf)     # 掩码在截断之后
      probs  = softmax(logits)
    """
    def __init__(self, embed_dim: int = 128, n_heads: int = 8, tanh_clipping: float = 10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        # [h_g, h_{t-1}, D_t] -> D
        self.ctx_proj = nn.Linear(embed_dim * 2 + 1, embed_dim)

        # glimpse（多头）
        self.glimpse = Glimpse(embed_dim, n_heads)

        # 最终单头注意力（点积）
        self.Wq_fin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk_fin = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,
                node_emb: torch.Tensor,      # [B, N, D]
                graph_emb: torch.Tensor,     # [B, D]
                last_idx: torch.Tensor,      # [B] 当前节点索引（t=1时可用0或虚拟）
                remain_dist: torch.Tensor,   # [B] 剩余距离 D_t
                feas_mask: Optional[torch.Tensor]  # [B, N] bool, True=可选
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          logits: [B, N] (已 clip 并掩码)
          probs : [B, N]
        """
        device = node_emb.device
        B, N, D = node_emb.shape

        # 1) 取 h_{t-1}
        h_last = node_emb[torch.arange(B, device=device), last_idx]           # [B,D]

        # 2) 拼接 C_t 并投到 D 维 （注意，这里和论文不一样，映射后再注意力，论文中是直接注意力）
        ctx_raw = torch.cat([graph_emb, h_last, remain_dist.unsqueeze(-1)], dim=-1)  # [B, 2D+1]
        ctx = self.ctx_proj(ctx_raw)                                          # [B, D]

        # 3) glimpse：用 ctx 作为 query 与所有节点做多头注意力
        ctx_refined = self.glimpse(ctx, node_emb, feas_mask)                  # [B, D]

        # 4) 最终单头相似度 u_t = C * tanh(q^T k / sqrt(D))   (clip BEFORE masking!)
        q = self.Wq_fin(ctx_refined)                                          # [B,D]
        k = self.Wk_fin(node_emb)                                             # [B,N,D]
        scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(D)              # [B,N]

        if self.tanh_clipping > 0:
            scores = self.tanh_clipping * torch.tanh(scores)                  # clip to [-C, C]

        # 5) 掩码非法点（置 -inf）
        if feas_mask is not None:
            scores = scores.masked_fill(~feas_mask, float("-inf"))

        # 6) softmax → 概率
        probs = F.softmax(scores, dim=-1)                                     # [B,N]
        return scores, probs

# ======= 打包成“策略网络” =======
class AttentionNet(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, n_encode_layers=3, tanh_clipping=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = GraphAttentionEncoder(n_heads, embed_dim, n_encode_layers, node_dim=3)
        self.decoder = OPDecoder(embed_dim, n_heads, tanh_clipping)

    def forward(self, points, agent_idx, remaining_dist, mask=None):
        """
        points: [B,N,3] 或 [N,3]
        agent_idx: [B] 或 [1]
        remaining_dist: [B] / [B,1] / [1,1]
        mask: [B,N] / [N]，1/True=可选
        return: torch.distributions.Categorical
        """
        if points.ndim == 2:                      # 单样本 → 扩展 batch 维
            points = points.unsqueeze(0)
        B, N, _ = points.shape

        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.bool() if mask is not None else None

        agent_idx = agent_idx.view(B).long()
        remaining_dist = remaining_dist.view(B).float()

        node_emb, graph_emb = self.encoder(points)                  # [B,N,D], [B,D]
        logits, probs = self.decoder(node_emb, graph_emb, agent_idx, remaining_dist, mask)
        return torch.distributions.Categorical(probs=probs)

    # 兼容你训练脚本的调用
    def forward_dist(self, points, agent_idx, rem, mask):
        return self.forward(points, agent_idx, rem, mask)