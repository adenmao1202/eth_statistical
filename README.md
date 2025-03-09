# 加密貨幣相關性分析工具 (Cryptocurrency Correlation Analyzer)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README_EN.md) | 繁體中文

這個工具使用Binance的歷史數據，以ETH為基準，比較所有其他加密貨幣合約價格的相關性。分析涵蓋了多個時間週期（1分鐘、5分鐘、15分鐘、1小時和4小時）。專為交易者和投資者設計，幫助發現市場相關性模式。

<p align="center">
  <img src="correlation_heatmap.png" alt="相關性熱圖範例" width="600">
</p>

## 功能特點

- 從Binance下載歷史價格數據
- 計算ETH與其他加密貨幣的價格相關性
- 支持多個時間週期的分析（1m、5m、15m、1h、4h）
- 視覺化相關性結果（條形圖和熱圖）
- 數據緩存功能，避免重複下載
- 多線程並行處理，提高分析速度
- 優化的數據處理和相關性計算

## 安裝步驟

1. 克隆此倉庫：

```bash
git clone https://github.com/zhiyu1105/crypto-correlation-analyzer.git
cd crypto-correlation-analyzer
```

2. 安裝依賴：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

直接運行主程序：

```bash
python crypto_correlation_analyzer.py
```

默認情況下，程序會：
- 分析過去30天的數據
- 顯示與ETH相關性最高的前20個加密貨幣
- 為每個時間週期生成相關性圖表
- 創建一個跨時間週期的相關性熱圖

### 快速分析模式

如果只需要快速分析特定時間週期，可以使用：

```bash
python quick_crypto_correlation.py
```

## 自定義分析

如需自定義分析參數，可以修改`crypto_correlation_analyzer.py`文件中的`main()`函數：

```python
# 設置日期範圍
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 修改天數

# 分析所有時間週期
results = analyzer.analyze_all_timeframes(start_date, end_date, top_n=20, use_cache=True)  # 修改top_n參數
```

## API密鑰（可選）

如果需要更高的API請求限制，可以在`main()`函數中設置Binance API密鑰：

```python
# 設置Binance API憑證
api_key = "YOUR_API_KEY"  # 替換為您的API密鑰
api_secret = "YOUR_API_SECRET"  # 替換為您的API密鑰
```

## 輸出文件

程序會生成以下文件：
- `historical_data/` 目錄：包含下載的歷史數據CSV文件
- `correlation_1m.csv`、`correlation_5m.csv`等：各時間週期的相關性數據
- `correlation_1m.png`、`correlation_5m.png`等：各時間週期的相關性條形圖
- `correlation_heatmap.png`：跨時間週期的相關性熱圖

## 項目結構

```
├── crypto_correlation_analyzer.py  # 主程序
├── quick_crypto_correlation.py     # 快速分析程序
├── requirements.txt               # 依賴包列表
├── README.md                      # 說明文檔
├── README_EN.md                   # 英文說明文檔
├── .gitignore                     # Git忽略文件
├── historical_data/               # 歷史數據目錄
├── correlation_*.csv              # 相關性數據文件
└── correlation_*.png              # 相關性圖表文件
```

## 注意事項

- 首次運行時，程序需要從Binance下載大量數據，可能需要較長時間
- 使用緩存功能（`use_cache=True`）可以加快後續運行速度
- Binance API有請求頻率限制，程序已添加延遲以避免超出限制
- 對於大量數據分析，建議使用較高性能的計算機

## 貢獻指南

歡迎提交問題和改進建議！請隨時提交 Issue 或 Pull Request。

## 許可證

本項目採用 [MIT 許可證](LICENSE)。
