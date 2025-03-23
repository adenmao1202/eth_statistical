#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering Analysis Module
Uses XGBoost for cryptocurrency clustering analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
import os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_fetcher import DataFetcher


class ClusteringAnalyzer:
    """
    Class for cryptocurrency clustering analysis
    """
    
    def __init__(self, data_fetcher=None, reference_symbol: str = 'ETHUSDT'):
        """
        Initialize ClusteringAnalyzer
        
        Args:
            data_fetcher: Optional DataFetcher instance
            reference_symbol: Symbol used as reference (e.g., 'ETHUSDT')
        """
        self.data_fetcher = data_fetcher
        self.reference_symbol = reference_symbol
        self.clusters = None
        self.feature_df = None
        self.feature_importance = None
        
    def fetch_data_for_all_symbols(self, symbols: List[str], timeframe: str, 
                                 start_date: str, end_date: str, 
                                 use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols (List[str]): List of symbols to fetch
            timeframe (str): Timeframe (e.g., '1h')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            use_cache (bool): Whether to use cached data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their historical data
        """
        print(f"正在獲取 {len(symbols)} 個交易對的歷史數據...")
        
        # Calculate number of days between start and end dates
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_datetime - start_datetime).days + 1
        
        # Use fetch_multi_symbols from data_fetcher
        result = self.data_fetcher.fetch_multi_symbols(
            symbols=symbols,
            interval=timeframe,
            days=days,
            end_date=end_date,
            use_cache=use_cache
        )
        
        print(f"成功獲取 {len(result)} 個交易對的數據")
        return result
    
    def extract_features_from_price_data(self, symbols, data_dict):
        """
        Extract features from OHLCV data for clustering
        
        Args:
            symbols: List of cryptocurrency symbols
            data_dict: Dictionary with OHLCV data for each symbol
            
        Returns:
            DataFrame: Features for each symbol
        """
        print("Extracting features from price data...")
        
        # 創建一個空的DataFrame來存儲所有特徵
        all_features = {}
        
        # 循環處理每個交易對並提取特徵
        for symbol in symbols:
            if symbol not in data_dict:
                print(f"沒有 {symbol} 的數據，跳過")
                continue
                
            df = data_dict[symbol].copy()
            
            # 檢查數據質量
            if df.shape[0] < 10:
                print(f"警告: {symbol} 的數據點太少 ({df.shape[0]}行)，跳過")
                continue
                
            # 檢查並處理重複的時間戳
            if df.index.duplicated().any():
                dup_count = df.index.duplicated().sum()
                print(f"警告: {symbol} 包含 {dup_count} 個重複的時間戳，將被刪除")
                df = df[~df.index.duplicated(keep='first')]
            
            # 確保數據按時間排序
            df = df.sort_index()
            
            try:
                # 計算回報率
                df['returns'] = df['close'].pct_change()
                
                # 提取價格動態特徵
                features = {}
                
                # 最近價格變化
                features['recent_return_1d'] = df['returns'].tail(24).mean()  # 假設是小時數據
                features['recent_return_3d'] = df['returns'].tail(72).mean()  
                features['recent_return_7d'] = df['returns'].tail(168).mean()
                
                # 波動性指標
                features['volatility_1d'] = df['returns'].tail(24).std()
                features['volatility_3d'] = df['returns'].tail(72).std()
                features['volatility_7d'] = df['returns'].tail(168).std()
                
                # 交易量指標
                features['volume_change_1d'] = df['volume'].tail(24).mean() / df['volume'].tail(48).head(24).mean() - 1
                features['volume_change_3d'] = df['volume'].tail(72).mean() / df['volume'].tail(144).head(72).mean() - 1
                
                # 價格範圍
                features['price_range_1d'] = (df['high'].tail(24).max() - df['low'].tail(24).min()) / df['close'].iloc[-1]
                features['price_range_3d'] = (df['high'].tail(72).max() - df['low'].tail(72).min()) / df['close'].iloc[-1]
                
                # 趨勢指標
                features['trend_1d'] = np.corrcoef(np.arange(24), df['close'].tail(24).values)[0, 1]
                features['trend_3d'] = np.corrcoef(np.arange(72), df['close'].tail(72).values)[0, 1]
                
                # 相對強弱指標 (RSI)
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss.replace(0, 0.0001)  # 避免除以零
                features['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
                
                # 計算移動平均
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['ma50'] = df['close'].rolling(window=50).mean()
                
                # 價格相對於移動平均線
                features['price_to_ma20'] = df['close'].iloc[-1] / df['ma20'].iloc[-1] - 1
                features['price_to_ma50'] = df['close'].iloc[-1] / df['ma50'].iloc[-1] - 1
                
                # 移動平均線交叉
                features['ma_cross'] = (df['ma20'].iloc[-1] > df['ma50'].iloc[-1]) * 2 - 1  # +1表示金叉，-1表示死叉
                
                # 添加与ETH的相关性特征
                if self.reference_symbol in data_dict and symbol != self.reference_symbol:
                    eth_df = data_dict[self.reference_symbol].copy()
                    if len(eth_df) > 0 and len(df) > 0:
                        # 确保两个数据框有相同的时间索引
                        common_index = df.index.intersection(eth_df.index)
                        if len(common_index) > 5:  # 至少需要5个共同点来计算相关性
                            eth_returns = eth_df.loc[common_index, 'close'].pct_change().fillna(0)
                            symbol_returns = df.loc[common_index, 'close'].pct_change().fillna(0)
                            features['eth_correlation'] = eth_returns.corr(symbol_returns)
                        else:
                            features['eth_correlation'] = 0
                    else:
                        features['eth_correlation'] = 0
                else:
                    features['eth_correlation'] = 1 if symbol == self.reference_symbol else 0
                
                # 將特徵存儲在字典中
                all_features[symbol] = features
                
            except Exception as e:
                print(f"處理 {symbol} 時出錯: {str(e)}")
                traceback.print_exc()
        
        # 將所有特徵轉換為DataFrame
        features_df = pd.DataFrame(all_features).T
        
        # 檢查是否有NaN值並處理
        if features_df.isna().any().any():
            print(f"警告: 特徵中存在 {features_df.isna().sum().sum()} 個NaN值，將使用均值填充")
            features_df = features_df.fillna(features_df.mean())
            
        # 顯示提取的特徵
        print(f"成功為 {features_df.shape[0]} 個交易對提取了 {features_df.shape[1]} 個特徵")
        
        return features_df
    
    def normalize_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features for clustering
        
        Args:
            feature_df (pd.DataFrame): DataFrame with extracted features
            
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        print("Normalizing features...")
        
        # Create a copy of the dataframe
        normalized_df = feature_df.copy()
        
        # Columns that should be scaled
        scale_columns = [col for col in normalized_df.columns if col != 'ma_crossover']
        
        # Apply standard scaling
        scaler = StandardScaler()
        normalized_df[scale_columns] = scaler.fit_transform(normalized_df[scale_columns])
        
        return normalized_df
    
    def apply_clustering(self, normalized_df: pd.DataFrame, 
                        n_clusters: int = config.NUM_CLUSTERS) -> Tuple[pd.Series, Any]:
        """
        Apply KMeans clustering to the normalized features
        
        Args:
            normalized_df (pd.DataFrame): DataFrame with normalized features
            n_clusters (int): Number of clusters to form
            
        Returns:
            Tuple[pd.Series, Any]: Cluster assignments and KMeans model
        """
        print(f"Applying KMeans clustering with {n_clusters} clusters...")
        
        # Create and fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_df)
        
        # Convert to Series
        cluster_series = pd.Series(labels, index=normalized_df.index)
        
        # Save cluster centers
        self.cluster_centers = kmeans.cluster_centers_
        
        return cluster_series, kmeans
    
    def train_xgboost_classifier(self, features, labels):
        """
        Train XGBoost classifier on clustered data to explain what features are important
        for cluster assignment
        
        Args:
            features: DataFrame with features
            labels: Cluster labels
            
        Returns:
            tuple: (feature_importance_df, trained_model)
        """
        # 確保labels是pandas Series
        if isinstance(labels, np.ndarray):
            y = pd.Series(labels)
        else:
            y = labels.copy()
            
        X = features.copy()
        
        # 檢查標籤是否從0開始且連續
        unique_labels = y.unique()
        expected_labels = list(range(len(unique_labels)))
        
        if not np.array_equal(sorted(unique_labels), expected_labels):
            print(f"警告: 標籤不連續。預期標籤: {expected_labels}, 實際標籤: {sorted(unique_labels)}")
            
            # 創建標籤映射
            label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
            y = y.map(label_map)
            
            # 驗證是否修復成功
            if not np.array_equal(sorted(y.unique()), expected_labels):
                print("標籤修復失敗，嘗試直接修復...")
                # 直接修復 - 基於唯一值的索引
                new_y = pd.Series(np.zeros_like(y.values))
                for i, label in enumerate(sorted(unique_labels)):
                    new_y[y.index[y == label]] = i
                y = new_y
                
            print(f"修復後的標籤: {sorted(y.unique())}")
        
        # 檢查每個類別的樣本數
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        
        print(f"每個類別的樣本數: {class_counts.to_dict()}")
        
        # 如果某些類別的樣本過少，使用非分層抽樣
        if min_samples < 2:
            print(f"警告: 類別 {class_counts.idxmin()} 只有 {min_samples} 個樣本，將使用非分層抽樣")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            # 使用分層抽樣來保持每個類別的比例
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        
        # 檢查訓練集中的唯一標籤
        train_labels = set(y_train)
        if len(train_labels) < 2:
            print(f"警告: 訓練集中只有一個類別 {train_labels}，無法訓練模型")
            return pd.DataFrame({'feature': X.columns, 'importance': 0.0}), None
            
        # 訓練 XGBoost 分類器
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1, 
            n_estimators=100, 
            objective='multi:softprob',
            num_class=len(train_labels),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # 評估模型
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"XGBoost 分類器準確率: {accuracy:.4f}")
        
        # 計算特徵重要性
        importance = model.feature_importances_
        
        # 創建特徵重要性的DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 10 重要特徵:")
        print(feature_importance.head(10))
        
        return feature_importance, model
    
    def visualize_clusters_2d(self, features, labels, output_dir=None):
        """
        Visualize clusters in 2D using PCA
        
        Args:
            features: Feature DataFrame
            labels: Cluster labels (can be numpy array or pandas Series)
            output_dir: Directory to save the plot
        """
        # 使用helper函數標準化標籤
        labels_series = self._normalize_labels(labels, features.index)
        labels_array = labels_series.values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # Create a DataFrame for plotting
        pca_df = pd.DataFrame(features_pca, columns=['PC1', 'PC2'])
        pca_df['cluster'] = labels_array
        pca_df['symbol'] = features.index
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Get a color map with distinct colors
        n_clusters = len(set(labels_array))
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        
        # Plot each cluster
        for i in range(n_clusters):
            cluster_data = pca_df[pca_df['cluster'] == i]
            plt.scatter(
                cluster_data['PC1'], 
                cluster_data['PC2'],
                label=f'Cluster {i}',
                color=cmap(i),
                alpha=0.7,
                s=100
            )
            
        # Add labels for each point
        for i, row in pca_df.iterrows():
            plt.annotate(
                row['symbol'],
                (row['PC1'], row['PC2']),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8
            )
            
        # Add explained variance
        explained_var = pca.explained_variance_ratio_
        plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
        plt.title('Cryptocurrency Clusters in 2D Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'clusters_2d.png'))
        else:
            plt.savefig('clusters_2d.png')
        plt.close()
        
        # Save PCA results
        if output_dir:
            pca_df.to_csv(os.path.join(output_dir, 'pca_results.csv'))
            
            # Save explained variance
            with open(os.path.join(output_dir, 'pca_explained_variance.txt'), 'w') as f:
                f.write(f"PC1: {explained_var[0]:.4f} ({explained_var[0]:.2%})\n")
                f.write(f"PC2: {explained_var[1]:.4f} ({explained_var[1]:.2%})\n")
                f.write(f"Total: {sum(explained_var[:2]):.4f} ({sum(explained_var[:2]):.2%})\n")
    
    def visualize_feature_importance(self, feature_importance, output_dir=None):
        """
        Visualize feature importance from XGBoost model
        
        Args:
            feature_importance: Series with feature importances
            output_dir: Directory to save the plot
        """
        plt.figure(figsize=(12, 8))
        feature_importance.sort_values().plot(kind='barh')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('XGBoost Feature Importance for Clustering')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        else:
            plt.savefig('feature_importance.png')
        plt.close()
    
    def analyze_clusters(self, feature_df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
        """
        Analyze the characteristics of each cluster
        
        Args:
            feature_df (pd.DataFrame): DataFrame with extracted features
            clusters (pd.Series): Cluster assignments
            
        Returns:
            pd.DataFrame: Cluster analysis summary
        """
        print("Analyzing cluster characteristics...")
        
        # Add cluster labels to feature dataframe
        feature_df_with_clusters = feature_df.copy()
        feature_df_with_clusters['cluster'] = clusters
        
        # Initialize list to store cluster summaries
        cluster_summaries = []
        
        # Analyze each cluster
        for cluster_id in sorted(clusters.unique()):
            # Get coins in this cluster
            cluster_coins = feature_df_with_clusters[feature_df_with_clusters['cluster'] == cluster_id]
            
            # Calculate summary statistics
            summary = {
                'cluster_id': cluster_id,
                'num_coins': len(cluster_coins),
                'avg_volatility': cluster_coins['volatility_1d'].mean(),
                'avg_eth_correlation': cluster_coins['eth_correlation'].mean(),
                'avg_recent_return': cluster_coins['recent_return_1d'].mean(),
                'avg_volume_change': cluster_coins['volume_change_1d'].mean(),
                'avg_rsi': cluster_coins['rsi'].mean(),
                'example_coins': ', '.join(cluster_coins.index[:5].tolist())
            }
            
            cluster_summaries.append(summary)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(cluster_summaries)
        
        # Print summary
        print("\n=== Cluster Analysis Summary ===")
        for _, row in summary_df.iterrows():
            print(f"\nCluster {int(row['cluster_id'])} ({int(row['num_coins'])} coins):")
            print(f"  Average Volatility: {row['avg_volatility']:.4f}")
            print(f"  Average ETH Correlation: {row['avg_eth_correlation']:.4f}")
            print(f"  Average Recent Return: {row['avg_recent_return']:.4f}")
            print(f"  Average RSI: {row['avg_rsi']:.2f}")
            print(f"  Example coins: {row['example_coins']}")
        
        # Save summary to CSV
        summary_df.to_csv('cluster_summary.csv', index=False)
        
        return summary_df
    
    def evaluate_clustering(self, normalized_df: pd.DataFrame, clusters) -> Dict[str, float]:
        """
        Evaluate clustering performance using various metrics
        
        Args:
            normalized_df (pd.DataFrame): DataFrame with normalized features
            clusters: Cluster assignments (can be pandas Series or numpy array)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        print("Evaluating clustering performance...")
        
        # 使用helper函數標準化標籤
        clusters_series = self._normalize_labels(clusters, normalized_df.index)
        clusters_array = clusters_series.values
        
        # Calculate silhouette score
        silhouette = silhouette_score(normalized_df, clusters_array)
        
        # Calculate Calinski-Harabasz Index
        calinski_harabasz = calinski_harabasz_score(normalized_df, clusters_array)
        
        # Calculate inertia (for K-means)
        if hasattr(self, 'kmeans'):
            inertia = self.kmeans.inertia_
        else:
            inertia = None
        
        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette,
            'calinski_harabasz_index': calinski_harabasz,
            'inertia': inertia
        }
        
        print("\n=== Clustering Evaluation Metrics ===")
        print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
        if inertia is not None:
            print(f"Inertia: {inertia:.4f} (lower is better)")
        
        return metrics
    
    def run_clustering_analysis(self, symbols, data_dict, n_clusters=5, output_dir=None):
        """
        Run complete clustering analysis workflow
        
        Args:
            symbols: List of cryptocurrency symbols
            data_dict: Dictionary with OHLCV data for each symbol
            n_clusters: Number of clusters to form
            output_dir: Directory to save results
            
        Returns:
            tuple: (labels, feature_importance)
        """
        print("\n" + "="*50)
        print(f"Beginning clustering analysis for {len(symbols)} cryptocurrencies")
        print("="*50)
        
        # 提取特徵
        try:
            features = self.extract_features_from_price_data(symbols, data_dict)
            print(f"Extracted {features.shape[1]} features for {features.shape[0]} cryptocurrencies")
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            traceback.print_exc()
            return None, None
        
        # 對特徵進行聚類
        try:
            labels, centers, metrics = self.cluster_coins(features, n_clusters)
            print(f"Cryptocurrency clustering complete. Formed {n_clusters} clusters")
            
            # 轉換為 Series 以便進一步處理
            if isinstance(labels, np.ndarray):
                labels_series = pd.Series(labels, index=features.index)
            else:
                labels_series = labels.copy()
            
            print("\nClusters distribution:")
            cluster_counts = labels_series.value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                print(f"Cluster {cluster}: {count} coins ({count/len(labels_series)*100:.1f}%)")
                
            # 顯示每個集群中的加密貨幣
            for cluster in sorted(labels_series.unique()):
                coins_in_cluster = labels_series[labels_series == cluster].index.tolist()
                print(f"\nCluster {cluster} ({len(coins_in_cluster)} coins):")
                print(", ".join(coins_in_cluster))
        except Exception as e:
            print(f"Error during clustering: {str(e)}")
            traceback.print_exc()
            return None, None
          
        # 訓練 XGBoost 分類器
        try:
            feature_importance, model = self.train_xgboost_classifier(features, labels_series)
            print("XGBoost model training completed.")
            
            # 如果模型訓練成功，繼續可視化
            if model is not None:
                # 解釋模型 - 根據特徵重要性分析每個集群
                print("\nAnalyzing clusters based on feature importance...")
                self.explain_model(model, features, labels_series, feature_importance)
            else:
                print("跳過模型相關可視化，因為模型訓練失敗")
        except Exception as e:
            print(f"Error during classifier training: {str(e)}")
            traceback.print_exc()
            feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0.0})
            
        # 評估聚類質量並打印指標
        print("\nClustering quality metrics:")
        print(f"Silhouette Score: {metrics.get('silhouette', 'N/A'):.4f}")
        if 'calinski_harabasz' in metrics:
            print(f"Calinski-Harabasz Index: {metrics.get('calinski_harabasz', 'N/A'):.1f}")
        
        # 創建2D可視化
        try:
            self.visualize_clusters_2d(features, labels_series, output_dir)
        except Exception as e:
            print(f"Error during 2D visualization: {str(e)}")
            traceback.print_exc()
            
        # 保存結果到文件
        if output_dir:
            try:
                # 確保輸出目錄存在
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存聚類結果
                result_file = os.path.join(output_dir, "cluster_results.json")
                cluster_results = {}
                
                for symbol, cluster in labels_series.items():
                    cluster_results[symbol] = int(cluster)
                    
                with open(result_file, 'w') as f:
                    json.dump(cluster_results, f, indent=2)
                print(f"Saved cluster results to {result_file}")
                
                # 保存特徵重要性
                importance_file = os.path.join(output_dir, "feature_importance.csv")
                feature_importance.to_csv(importance_file, index=False)
                print(f"Saved feature importance to {importance_file}")
                
                # 保存聚類指標
                metrics_file = os.path.join(output_dir, "clustering_metrics.json")
                # 將numpy值轉換為Python原生類型以進行JSON序列化
                metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in metrics.items()}
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                print(f"Saved clustering metrics to {metrics_file}")
                
                # 保存標籤映射（如果存在）
                if hasattr(self, 'label_mapping') and self.label_mapping is not None:
                    mapping_file = os.path.join(output_dir, "label_mapping.json")
                    with open(mapping_file, 'w') as f:
                        json.dump(self.label_mapping, f, indent=2)
                    print(f"Saved label mapping to {mapping_file}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                traceback.print_exc()
                
        return labels_series, feature_importance

    def explain_model(self, model, features, labels, output_dir=None):
        """
        Generate model explanations using SHAP values
        
        Args:
            model: Trained XGBoost model
            features: Feature DataFrame
            labels: Cluster labels (can be numpy array or pandas Series)
            output_dir: Directory to save plots
        """
        # 使用helper函數標準化標籤
        labels_series = self._normalize_labels(labels, features.index)
        labels_array = labels_series.values
            
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            
            # Get SHAP values
            shap_values = explainer.shap_values(features)
            
            # Summary plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
            else:
                plt.savefig('shap_summary.png')
            plt.close()
            
            # SHAP dependency plots for top features
            feature_importance = pd.Series(
                np.abs(shap_values).mean(axis=0), 
                index=features.columns
            ).sort_values(ascending=False)
            
            top_features = feature_importance.index[:5]  # Top 5 features
            
            for feature in top_features:
                plt.figure(figsize=(12, 8))
                shap.dependence_plot(
                    feature, shap_values, features, 
                    interaction_index=None, show=False
                )
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}.png'))
                else:
                    plt.savefig(f'shap_dependence_{feature}.png')
                plt.close()
                
            print(f"模型解釋圖生成完成，已保存到 {output_dir if output_dir else '當前目錄'}")
            
        except ImportError:
            print("無法導入SHAP庫。跳過模型解釋。")
        except Exception as e:
            print(f"生成模型解釋時出錯: {e}")

    def cluster_coins(self, features, n_clusters=5):
        """
        Cluster cryptocurrencies based on extracted features
        
        Args:
            features: DataFrame with extracted features
            n_clusters: Number of clusters to form
            
        Returns:
            tuple: (labels, cluster_centers, scores)
        """
        print(f"Clustering {features.shape[0]} coins into {n_clusters} groups...")
        
        # Standardize features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        original_labels = kmeans.fit_predict(features_scaled)
        
        # 確保標籤是從0到n_clusters-1的連續整數
        unique_labels = np.unique(original_labels)
        
        # 檢查標籤值是否連續
        if len(unique_labels) != n_clusters or list(unique_labels) != list(range(n_clusters)):
            print(f"警告: 從KMeans獲得的標籤不符合預期: {unique_labels}")
            
            # 重新映射標籤
            label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
            labels = np.array([label_map[l] for l in original_labels])
            
            # 檢查是否修復成功
            if list(np.unique(labels)) != list(range(len(np.unique(labels)))):
                print(f"警告: 映射後的標籤仍不連續: {np.unique(labels)}")
                print("使用強制修復方法...")
                
                # 強制修復 - 直接使用索引作為標籤
                new_labels = np.zeros_like(original_labels)
                for i, old_label in enumerate(np.unique(original_labels)):
                    new_labels[original_labels == old_label] = i
                
                # 檢查修復後的標籤
                if list(np.unique(new_labels)) != list(range(len(np.unique(new_labels)))):
                    print(f"錯誤: 無法修復標籤: {np.unique(new_labels)}")
                else:
                    labels = new_labels
                    print(f"成功修復標籤，新的標籤值: {np.unique(labels)}")
            else:
                print(f"成功重新映射標籤: {np.unique(labels)}")
                
            # 保存標籤映射以便後續使用
            self.label_mapping = {
                'original_to_new': {str(old): int(new) for old, new in label_map.items()},
                'new_to_original': {str(new): int(old) for old, new in label_map.items()}
            }
        else:
            # 標籤已經是連續的
            labels = original_labels
            self.label_mapping = None
        
        # 保存kmeans模型以便後續計算慣性等評估指標
        self.kmeans = kmeans
        
        # Get cluster centers and transform back to original scale
        centers = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers)
        
        # Calculate silhouette score
        silhouette = silhouette_score(features_scaled, labels)
        
        print(f"Clustering complete. Silhouette score: {silhouette:.4f}")
        
        # 返回labels作為numpy數組
        return labels, centers_original, {'silhouette': silhouette}
        
    def generate_cluster_profiles(self, features, labels, n_clusters):
        """
        Generate profiles for each cluster
        
        Args:
            features: DataFrame with features
            labels: Cluster labels (can be numpy array or pandas Series)
            n_clusters: Number of clusters
            
        Returns:
            dict: Cluster profiles
        """
        profiles = {}
        
        # 使用helper函數標準化標籤
        labels_series = self._normalize_labels(labels, features.index)
        
        for i in range(n_clusters):
            # 獲取此集群中的幣種索引
            cluster_coins = features.index[labels_series == i].tolist()
            
            # 如果沒有幣種，跳過
            if not cluster_coins:
                profiles[f"cluster_{i}"] = {
                    "members": [],
                    "size": 0,
                    "description": "Empty cluster"
                }
                continue
                
            # 獲取此集群的特徵
            cluster_features = features.loc[features.index.isin(cluster_coins)]
            
            # 計算平均特徵值
            mean_values = cluster_features.mean()
            
            # 判斷主要特徵
            high_volatility = mean_values['volatility_1d'] > features['volatility_1d'].mean()
            high_volume = mean_values['avg_volume'] > features['avg_volume'].mean()
            positive_momentum = mean_values['momentum_7d'] > 0
            
            # 產生描述
            description = []
            if high_volatility:
                description.append("高波動性")
            else:
                description.append("低波動性")
                
            if high_volume:
                description.append("高交易量")
            else:
                description.append("低交易量")
                
            if positive_momentum:
                description.append("正向動量")
            else:
                description.append("負向動量")
                
            # 構建檔案
            profile = {
                "members": cluster_coins,
                "size": len(cluster_coins),
                "description": "、".join(description),
                "mean_values": {k: float(v) for k, v in mean_values.items()}
            }
            
            profiles[f"cluster_{i}"] = profile
            
        return profiles

    def _normalize_labels(self, labels, features_index):
        """
        將標籤轉換為pandas Series格式，使其與features索引對齊
        
        Args:
            labels: 標籤數據，可以是numpy數組或pandas Series
            features_index: 特徵DataFrame的索引
            
        Returns:
            pd.Series: 標準化後的標籤Series
        """
        if isinstance(labels, pd.Series):
            return labels
        else:
            return pd.Series(labels, index=features_index)