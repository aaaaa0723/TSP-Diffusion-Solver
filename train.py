import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 從旁邊的檔案匯入
from dataset import TSPDataset
from model import TSPPureGNNModel 

def train_model():
    # 1. 基本設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用裝置: {device}")
    
    epochs = 20
    batch_size = 64
    # 【關鍵修改 1】調降學習率，防止訓練尾聲發生梯度爆炸
    learning_rate = 5e-4 

    # 2. 載入與切割資料 (70/15/15)
    print("⏳ 準備載入與切割資料集...")
    dataset = TSPDataset('tsp_dataset_lite.npz')
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ 切割完成！訓練集: {train_size} 筆 | 驗證集: {val_size} 筆 | 測試集: {test_size} 筆")

    # 3. 初始化模型 
    model = TSPPureGNNModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = []
    val_loss_history = []

    # 4. 開始訓練
    print("🏃‍♂️ 訓練與監督正式開始！")
    
    for epoch in range(epochs):
        # ================== 訓練階段 ==================
        model.train() 
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_coords, batch_dists, batch_adjs in pbar:
            batch_coords = batch_coords.to(device)
            batch_dists = batch_dists.to(device)
            batch_adjs = batch_adjs.to(device, dtype=torch.float32) 

            optimizer.zero_grad()
            outputs = model(batch_coords, batch_dists)
            loss = criterion(outputs, batch_adjs)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ================== 驗證階段 ==================
        model.eval() 
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_coords, batch_dists, batch_adjs in val_loader:
                batch_coords = batch_coords.to(device)
                batch_dists = batch_dists.to(device)
                batch_adjs = batch_adjs.to(device, dtype=torch.float32) 
                
                outputs = model(batch_coords, batch_dists)
                val_loss = criterion(outputs, batch_adjs)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"✅ Epoch {epoch+1} 結束 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ==========================================
    # 存檔與畫圖
    # ==========================================
    print("\n💾 訓練完成，正在進行緊急存檔...")
    torch.save(model.state_dict(), "tsp_gnn_model.pth")
    print("🎉 模型權重已安全存檔！")

    print("🖼️ 準備繪製雙線 Loss 曲線...")
    plot_loss(train_loss_history, val_loss_history)


def plot_loss(train_loss, val_loss):
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    # 【關鍵修改 1】改用 MultipleLocator 來指定固定間隔
    from matplotlib.ticker import MultipleLocator 
    
    plt.figure(figsize=(8, 6))
    
    epochs_range = range(1, len(train_loss) + 1)
    
    plt.plot(epochs_range, train_loss, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(epochs_range, val_loss, marker='x', linestyle='--', color='r', label='Val Loss')
    
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    
    # 【關鍵修改 2】強制 X 軸刻度以 2 為單位跳動
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    
    plt.legend()
    plt.grid(True)
    
    plt.savefig('loss_curve.png') 
    print("📊 雙線 Loss 曲線已成功儲存為 loss_curve.png！")
if __name__ == "__main__":
    train_model()