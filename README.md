# ETH 統計分析框架

ETH 統計分析框架是一個全面的工具，用於分析以太坊價格變動對其他加密貨幣的影響。這個項目整合了三種不同的研究方法，以獲取有關加密貨幣市場的深入見解。

## 核心功能

### 1. 事件研究 (Event Study)

分析ETH價格下跌事件對其他加密貨幣的影響，從而識別出：
- 在ETH下跌期間表現穩定的幣種
- 在ETH下跌後反彈最好的幣種
- 異常收益率和統計顯著性的幣種

### 2. 聚類分析 (Clustering)

使用XGBoost進行加密貨幣聚類，基於：
- 與ETH的相關性
- 波動性指標
- 交易量特徵
- 價格動量

### 3. 假設檢驗 (Hypothesis Testing)

測試ETH下跌前後山寨幣收益分佈是否有顯著差異，並計算：
- 均值差異和顯著性水平 (p值)
- 效應大小 (Cohen's d)
- 多重檢驗校正
- 分佈比較和視覺化

## 技術特點

- **緩存系統**: 智能管理Binance API請求，避免重複獲取數據，提高效率
- **多線程處理**: 並行請求數據，同時遵守API請求限制
- **可視化**: 生成圖表和交互式圖形，直觀展示分析結果
- **參數化分析**: 支持不同的時間框架、閾值和窗口大小設置

## 安裝與設置

1. 安裝依賴項：

```bash
# 使用setup.py安裝
python setup.py install

# 或者手動安裝依賴
pip install -r requirements.txt
```

2. 配置API密鑰 (可選)：

如果需要獲取大量數據，建議使用自己的Binance API密鑰，可以在命令行中指定，或添加到代碼中。

## 使用方法

### 命令行界面

使用統一的命令行界面運行所有分析：

```bash
# 運行事件研究
python main/main.py --run event_study --timeframe 1m --days 30 --drop_threshold -0.01

# 運行聚類分析
python main/main.py --run clustering --timeframe 1h --days 90 --clusters 5

# 運行假設檢驗
python main/main.py --run hypothesis_testing --timeframe 1m --days 60 --pre_event_window 30 --post_event_window 30
```

### 緩存管理

```bash
# 顯示緩存統計信息
python main/main.py --run cache_info

# 清理過期緩存
python main/main.py --run clean_cache

# 強制清理所有緩存
python main/main.py --run clean_cache --force_clean
```

## 結果輸出

所有分析結果都保存在組織良好的目錄結構中：

```
results/
  ├── event_study/timeframe_days_drop_window/
  ├── clustering/timeframe_days_clusters/
  └── hypothesis_testing/timeframe_days_drop_window/
```

每個輸出目錄包含：
- 參數記錄 (JSON格式)
- 分析結果數據
- 視覺化圖表
- 詳細統計數據

## 項目結構

```
eth_statistical/
  ├── main/
  │   ├── event_study/       # 事件研究相關代碼
  │   ├── clustering/        # 聚類分析相關代碼
  │   ├── hypothesis_testing/ # 假設檢驗相關代碼
  │   ├── historical_data/   # 數據緩存目錄
  │   ├── config.py          # 配置參數
  │   ├── data_fetcher.py    # 數據獲取工具
  │   ├── utils.py           # 通用工具函數
  │   └── main.py            # 主入口點
  ├── results/               # 分析結果輸出目錄
  ├── setup.py               # 安裝腳本
  └── requirements.txt       # 依賴項清單
```

## 資料獲取優化

該框架優化了Binance API的使用，通過以下機制：
- 智能緩存避免重複請求
- 遵守API速率限制
- 並行處理提高效率
- 自動重試失敗的請求

