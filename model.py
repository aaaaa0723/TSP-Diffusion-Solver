import torch
import torch.nn as nn

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
        self.predictor = nn.Sequential(
            # 拔掉時間特徵後，輸入維度只剩下 i, j 兩個節點的特徵 (hidden_dim * 2)
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            # 【重點】輸出 1 維 Logits，不加 Sigmoid，交給外部的 BCEWithLogitsLoss 處理
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, coords, dist_matrix):
        # 輸入參數乾淨俐落，只留下 coords 跟 dist_matrix
        batch_size, num_nodes, _ = coords.shape
        
        # 1. 節點初始特徵 (batch, 20, hidden_dim)
        x = self.node_encoder(coords) 
        
        # 2. GCN 訊息傳遞 (Message Passing)
        # 第一階段沒了雜訊矩陣，我們直接把真實的「距離矩陣」當作圖的權重來傳遞情報
        weight = dist_matrix 
        
        x = self.gcn1(x, weight) # 第一回合情報交換
        x = self.gcn2(x, weight) # 第二回合情報交換
        
        # 3. 把更新後的節點特徵配對 (i 節點 + j 節點)
        x_i = x.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        x_j = x.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, -1)
        
        # 拼接在一起 (不用再拼接時間特徵了)
        final_feat = torch.cat([x_i, x_j], dim=-1)
        
        # 4. 預測最終的 0/1 相鄰矩陣
        pred_adj = self.predictor(final_feat).squeeze(-1) 
        
        return pred_adj