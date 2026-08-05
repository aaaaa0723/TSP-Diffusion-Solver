import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    """時間嵌入層：保持不變，產生 sin/cos 時間特徵"""
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        device = t.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class SimpleGCNLayer(nn.Module):
    """手刻的極速 GCN 層：負責城市之間的情報交換"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, node_feat, adj_matrix):
        # node_feat: (batch, 20, in_dim)
        # adj_matrix: (batch, 20, 20) -> 這是城市之間的連接權重
        
        # 【核心魔法】矩陣相乘 (bmm) 瞬間完成所有鄰居特徵的加權總和
        aggregated_feat = torch.bmm(adj_matrix, node_feat)
        
        # 加上線性轉換與激發函數
        out = self.activation(self.linear(aggregated_feat))
        return out

class TSPDiffusionModel(nn.Module):
    def __init__(self, node_dim=2, time_dim=64, hidden_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # 節點特徵初始化
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # 建立兩層 GCN 來做深度情報交換
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        
        # 最終預測器：把融合後的節點特徵轉成邊緣機率
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, coords, dist_matrix, noisy_adj, t):
        batch_size, num_nodes, _ = coords.shape
        
        # 1. 時間特徵 (batch, time_dim)
        t_emb = self.time_mlp(t) 
        
        # 2. 節點初始特徵 (batch, 20, hidden_dim)
        x = self.node_encoder(coords) 
        
        # 3. GCN 訊息傳遞 (Message Passing)
        # 我們把「距離」跟「目前的雜訊狀態」相加或相乘當作圖的權重
        # 這裡示範簡單相加，讓模型知道誰離得近、誰目前機率高
        combined_weight = dist_matrix + noisy_adj 
        
        x = self.gcn1(x, combined_weight) # 第一回合情報交換
        x = self.gcn2(x, combined_weight) # 第二回合情報交換
        
        # 4. 把更新後的節點特徵配對 (i 節點 + j 節點 + 時間特徵)
        x_i = x.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        x_j = x.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, -1)
        t_emb_expand = t_emb.unsqueeze(1).unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        
        # 拼接在一起
        final_feat = torch.cat([x_i, x_j, t_emb_expand], dim=-1)
        
        # 5. 預測最終的 0/1 相鄰矩陣
        pred_adj = self.predictor(final_feat).squeeze(-1) 
        
        return pred_adj