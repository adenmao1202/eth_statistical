# ETH Event Study Analysis

## 概述

此模組提供了一個全面的框架，用於對加密貨幣市場進行穩健的事件研究。它旨在分析ETH價格下跌對各種加密貨幣的影響，跨多個時間段進行分析，確保統計嚴謹性並避免前瞻性偏差。

## 核心功能

- **跨時段驗證**：分析加密貨幣在多個時間段的表現，以識別具有一致行為的幣種。
- **假設檢驗整合**：整合正式的統計假設檢驗以驗證發現。
- **多維代幣分類**：依市值、交易量、波動性、相關性以及隨機籃子分組，以用於比較分析。
- **穩健事件檢測**：基於價格變動、交易量和連續下跌識別ETH價格下跌事件。
- **統計顯著性檢驗**：應用各種統計測試來驗證異常收益。
- **穩定性指標**：計算多種穩定性指標，以識別在不同市場條件下具有一致響應的代幣。
- **豐富的視覺化**：生成全面的視覺化和報告，以便直觀理解結果。

## 模組架構

### RobustEventStudy

整合代幣分類和時間序列驗證，實現無前瞻性偏差的全面事件研究。其主要功能包括：
- 對代幣進行多維度分類
- 進行跨時段的穩健事件研究
- 生成綜合的分析報告

### RobustEventAnalyzer

在多個時間段進行嚴格的統計分析，以識別穩定代幣。其核心特點：
- 高級統計檢驗
- 異常收益的深入分析
- 穩定性得分計算

### TimeSeriesValidator

評估加密貨幣在不同時間段的穩定性和表現。具體功能：
- 跨多個時間段評估加密貨幣的表現
- 計算穩定性指標
- 識別ETH價格下跌事件
- 生成穩定性報告和可視化圖表

### CoinClassifier

基於客觀的事件前特徵對加密貨幣進行分類，避免前瞻性偏差。分類方式包括：
- 依市值分類 (從外部API獲取市值數據)
- 依交易量分類 (基於歷史交易量分層)
- 依波動性分類 (基於價格波動率分層)
- 依ETH/BTC相關性分類 (分析與主要加密貨幣的價格相關性)
- 創建隨機代幣籃子 (用於基準比較)
- 自動識別穩定幣 (從分析中排除)

### HypothesisTesting

為加密貨幣回報進行統計假設檢驗，實現嚴謹的學術研究標準。功能包括：
- 多種統計檢驗（t檢驗、交叉截面標準差檢驗等）
- 效果大小計算
- 多重檢驗校正
- 檢驗結果可視化

## 使用方法

### 運行事件研究

```bash
python main.py --mode robust_event_study --classify_by_market_cap --classify_by_volume --classify_by_volatility --create_random_baskets --total_days 120 --period_length 30 --timeframe 5m --use_cache
```

### 命令行參數

- `--mode`：分析模式（`validation`, `event_study`, `robust_event_study`, `robust_event_analyzer`）
- `--timeframe`：數據時間框架（如 `1m`, `5m`, `15m`, `1h`）
- `--total_days`：要分析的總天數
- `--period_length`：每個驗證時段的天數
- `--request_delay`：API請求間的延遲（秒）
- `--use_cache`：使用緩存數據（當可用時）
- `--classify_by_market_cap`：按市值分類代幣
- `--classify_by_volume`：按交易量分類代幣
- `--classify_by_volatility`：按價格波動性分類代幣
- `--classify_by_sector`：按行業部門分類代幣
- `--create_random_baskets`：創建隨機代幣籃子作為比較
- `--drop_threshold`：ETH價格下跌閾值（負值）
- `--consecutive_drops`：需要連續下跌的蠟燭數量
- `--volume_factor`：ETH事件的交易量增加因子
- `--pre_event_window`：事件前窗口（分鐘）
- `--post_event_window`：事件後窗口（分鐘）
- `--significance_level`：統計顯著性水平
- `--output_suffix`：添加到輸出目錄名稱的後綴

## 優化執行速度建議

為了加快分析速度，可以嘗試以下方法：

1. **使用緩存功能**：添加 `--use_cache` 參數以重複利用已下載的數據
2. **減少分析時間范圍**：將 `--total_days` 設置為較小的值 (如 60 或 30 天)
3. **使用較大的時間框架**：將 `--timeframe` 從 5m 改為 15m、30m 或 1h
4. **減少分類方式**：僅使用必要的分類方法，如 `--classify_by_volume --classify_by_volatility`
5. **調整 request_delay**：在網絡連接良好的情況下，可以減少API請求之間的延遲

例如，以下命令可以顯著縮短執行時間：
```bash
python main.py --mode robust_event_study --classify_by_volume --total_days 60 --period_length 30 --timeframe 15m --use_cache --request_delay 0.05
```

## 分析模式

### RobustEventStudy

整合代幣分類和時間序列驗證進行全面事件研究，避免前瞻性偏差。

```bash
python main.py --mode robust_event_study
```

### RobustEventAnalyzer

跨多個時間段進行嚴格統計分析，識別穩定代幣。

```bash
python main.py --mode robust_event_analyzer
```

### ValidationOnly

專注於驗證加密貨幣在不同時間段的表現。

```bash
python main.py --mode validation
```

## 輸出結果

分析產生的輸出包括：

1. **穩定性分析**：包含每個代幣跨時段穩定性指標的CSV文件
2. **時段摘要**：每個時段事件和顯著代幣的摘要
3. **分析報告**：包含整體發現和統計數據的JSON報告
4. **視覺化**：顯示頂級穩定代幣、穩定性組件等的圖表
5. **詳細日誌**：保存在`event_study.log`中的分析過程詳細記錄

## 統計方法

該模組實現了多種統計檢驗方法：

- 配對和獨立樣本t檢驗
- 交叉截面標準差檢驗
- 效果大小計算
- 方向穩定性指標
- 幅度穩定性評估
- 多重檢驗校正

## 系統需求

- Python 3.7+
- pandas, numpy, matplotlib, seaborn, plotly
- scipy, statsmodels（用於統計檢驗）
- rich（用於控制台輸出）

## 重要說明

- **API密鑰**：需要在`config.py`文件中配置Binance API密鑰。
- **計算密集度**：對於大時間框架或大量代幣，分析可能計算密集。
- **緩存使用**：使用`--use_cache`選項可加速重複分析。
- **數據存儲**：歷史數據保存在`historical_data/`目錄，結果保存在`results/`目錄。
- **API限制**：程序會自動遵守幣安API的權重限制，當接近限制時會暫停請求一段時間。

## 事件研究流程

本模組實現了一個嚴謹的事件研究方法，遵循以下步驟：

1. **數據準備**：收集加密貨幣價格數據並處理缺失值
2. **事件定義**：基於可配置參數識別ETH價格下跌事件
3. **預期收益計算**：使用各種模型計算預期收益
4. **異常收益計算**：計算異常收益（實際收益與預期收益的差異）
5. **統計檢驗**：應用統計檢驗來驗證異常收益
6. **跨時段穩定性分析**：評估發現在多個時間段的穩定性
7. **結果分析和解釋**：生成全面的報告和視覺化

## 聯絡方式

如有問題或改進建議，請聯繫開發團隊。
