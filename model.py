import torch
import torch.nn as nn

class SimpleGCNLayer(nn.Module):
    """手刻的極速 GCN 層：負責城市之間的情報交換"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, node_feat, adj_matrix):
        # node_feat: (batch, num_nodes, in_dim)
        # adj_matrix: (batch, num_nodes, num_nodes) -> 城市之間的連接權重
        
        # 【核心魔法】矩陣相乘 (bmm) 瞬間完成所有鄰居特徵的加權總和
        aggregated_feat = torch.bmm(adj_matrix, node_feat)
        
        # 加上線性轉換與激發函數
        out = self.activation(self.linear(aggregated_feat))
        return out

class TSPPureGNNModel(nn.Module):
    """第一階段驗證專用：純 GNN 模型 (無 Diffusion 機制)"""
    def __init__(self, node_dim=2, hidden_dim=128):
        super().__init__()
        
        # 節點特徵初始化
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # 建立兩層 GCN 來做深度情報交換
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        
        # 最終預測器：把融合後的節點特徵轉成邊緣機率
        self.edge_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.edge_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.edge_weight)

    def forward(self, coords, dist_matrix):
        # 輸入參數乾淨俐落，只留下 coords 跟 dist_matrix
        batch_size, num_nodes, _ = coords.shape
        
        # 1. 統一座標輸入尺度，實際公里距離仍由資料集保留
        coord_scale = coords.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        x = self.node_encoder(coords / coord_scale)
        
        # 2. GCN 訊息傳遞 (Message Passing)
        # 近距離邊具有較高關聯，並做 row normalization 保持聚合穩定
        scale = dist_matrix.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        weight = torch.exp(-dist_matrix / scale)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        x = self.gcn1(x, weight) # 第一回合情報交換
        x = self.gcn2(x, weight) # 第二回合情報交換
        
        # 3. Bilinear edge scorer 避免建立 (batch, N, N, 2 * hidden_dim)
        pred_adj = torch.einsum("bih,hk,bjk->bij", x, self.edge_weight, x)
        pred_adj = pred_adj + self.edge_bias

        return pred_adj