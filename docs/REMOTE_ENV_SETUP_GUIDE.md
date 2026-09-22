# 遠端 Linux 環境設置教學（Conda + Python）

這份教學用來說明如何在遠端伺服器或雲端主機上，設置這個 TSP 專案所需的執行環境，並順利啟動它。

---

## 1. 前置條件

你需要：

- 一台 Linux 遠端主機（例如 Ubuntu / Debian / CentOS）
- SSH 權限
- sudo 權限（若需要安裝系統套件）
- 可連網，能從 GitHub 拉專案

如果你已經有遠端環境，直接從「2. 下載專案」開始即可。

---

## 2. 連上遠端伺服器

```bash
ssh user@your_server_ip
```

例如：

```bash
ssh labuser@203.0.113.10
```

登入後，先確認 Python / shell 狀態：

```bash
whoami
pwd
uname -a
```

---

## 3. 安裝 Miniconda（如果本機尚未安裝）

### 3.1 下載 Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

如果你需要 ARM 架構，則需改用對應版本；通常 x86_64 足夠。

### 3.2 安裝

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安裝過程中：

- 按 Enter 讀條款
- 輸入 `yes`
- 接受預設安裝路徑
- 之後可選擇是否初始化 conda（建議選 yes）

### 3.3 重新載入 shell

```bash
source ~/.bashrc
```

### 3.4 驗證

```bash
conda --version
```

如果正常輸出版本號，表示安裝成功。

---

## 4. 建立專案環境

在專案根目錄外部建立一個專用環境：

```bash
conda create -n tsp_env python=3.11 -y
```

進入環境：

```bash
conda activate tsp_env
```

確認版本：

```bash
python -V
```

---

## 5. 下載專案

```bash
git clone https://github.com/aaaaa0723/TSP-Diffusion-Solver.git
cd TSP-Diffusion-Solver
```

如果你想放在指定資料夾，也可以：

```bash
git clone https://github.com/aaaaa0723/TSP-Diffusion-Solver.git /workspace/tsp_solver
cd /workspace/tsp_solver
```

---

## 6. 安裝依賴套件

專案的依賴已經寫在 `requirements.txt` 。直接執行：

```bash
pip install -r requirements.txt
```

若要確認依賴是否都成功安裝：

```bash
python -c "import numpy, torch, matplotlib, optuna, ortools, seaborn, tqdm; print('OK')"
```

若輸出 `OK`，代表依賴已正確安裝。

---

## 7. 啟動專案環境的簡便方式

這個專案提供了啟動腳本：

```bash
bash activate_tsp_env.sh
```

它會做的事情是：

- 載入 Miniconda 的 shell 設定
- 啟動 `tsp_env`

如果你直接用 bash 方式啟動，也可以：

```bash
source /home/你的使用者名稱/miniconda3/etc/profile.d/conda.sh
conda activate tsp_env
```

---

## 8. 執行專案流程

### 8.1 產生資料集

```bash
python generate_tsp.py
```

這會產生：

- `tsp_dataset_lite.npz`

注意：這一步可能需要一些時間，因為資料生成包含 OR-Tools 求解。

### 8.2 訓練模型

```bash
python train.py
```

正式訓練會輸出：

- `tsp_gnn_model.pth`
- `loss_curve.png`

### 8.3 驗證與分析

```bash
python test_histogram.py
```

這會生成：

- `error_histogram.png`

```bash
python plot_qualitative.py
```

這會生成：

- `worst_case_comparison.png`

---

## 9. 常見問題與解法

### 問題 1：`conda` 指令不存在

解法：

```bash
source ~/.bashrc
```

如果仍沒有，確認是否已成功安裝 Miniconda。

---

### 問題 2：`pip install -r requirements.txt` 失敗

可能原因：

- Python 版本不符合
- 網路不穩
- 缺少編譯工具

建議：

```bash
conda activate tsp_env
python -V
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 問題 3：找不到 `ortools`

確認你是否在正確環境中：

```bash
conda activate tsp_env
python -c "import ortools; print(ortools.__version__)"
```

若沒有，重跑：

```bash
pip install ortools
```

---

### 問題 4：CUDA/torch 無法使用

這個專案會自行判斷：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

如果是 CPU 環境，程式仍可執行；只是速度會較慢。

若要在 GPU 環境中跑得更快，請確保：

- NVIDIA 驅動已安裝
- CUDA 版本與 PyTorch 相容
- 直接安裝適合 CUDA 架構的 PyTorch build

---

### 問題 5：資料集或模型檔不存在

如果執行訓練或分析時出現找不到檔案，先確認路徑是否正確：

```bash
ls -l
```

確認以下檔案存在：

- `tsp_dataset_lite.npz`
- `tsp_gnn_model.pth`

如果沒有，先跑：

```bash
python generate_tsp.py
python train.py
```

---

## 10. 建議的遠端操作流程

最推薦的實際操作順序：

```bash
ssh user@server
conda activate tsp_env
cd /path/to/TSP-Diffusion-Solver
pip install -r requirements.txt
python generate_tsp.py
python train.py
python test_histogram.py
python plot_qualitative.py
```

---

## 11. 一個最簡單的部署模板

如果你只想快速在遠端重現它，最簡單寫法如下：

```bash
cd ~
git clone https://github.com/aaaaa0723/TSP-Diffusion-Solver.git
cd TSP-Diffusion-Solver
conda create -n tsp_env python=3.11 -y
conda activate tsp_env
pip install -r requirements.txt
python generate_tsp.py
python train.py
```

---

## 12. 結論

這個專案的遠端環境設置相對簡單，核心只需要：

- Conda
- Python 3.11
- requirements.txt 內的套件
- 正確的資料生成與訓練步驟

只要依照上述步驟操作，通常就能在遠端主機上正常復現這套 TSP GNN baseline。

---

## 13. 補充：這個專案的實際設置建議

如果你準備要在遠端長期跑訓練，建議：

- 用 `conda` 建立獨立環境，不要污染 base
- 將專案放在固定路徑，例如 `/workspace/TSP-Diffusion-Solver`
- 每次執行前先確認 `conda activate tsp_env`
- 把資料集、checkpoint、圖片輸出保留在專案根目錄內
- 在 GPU 主機上訓練時，要確認 GPU driver / CUDA 相容性

這樣後續維護與重複執行會比較容易。
