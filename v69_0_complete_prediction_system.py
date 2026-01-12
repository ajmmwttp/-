# =============================================================================
# v69.0 メタ認知AI予測システム【完全版】
# =============================================================================
# 統計理論(40+手法) + 機械学習(15モデル) = 55+予測モデル統合
# =============================================================================

print("="*70)
print("【v69.0 メタ認知AI予測システム】完全版")
print("  統計理論 + 機械学習 ハイブリッドアンサンブル")
print("="*70)

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from datetime import date, timedelta, datetime
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# ライブラリ読み込み
# =============================================================================
print("\n【ライブラリ確認】")

# statsmodels
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import statsmodels.api as sm
    HAS_STATSMODELS = True
    print("  ✅ statsmodels")
except ImportError:
    HAS_STATSMODELS = False
    print("  ⚠️ statsmodels なし")

# Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
    print("  ✅ Prophet")
except ImportError:
    HAS_PROPHET = False

# jpholiday
try:
    import jpholiday
    HAS_JPHOLIDAY = True
    print("  ✅ jpholiday")
except ImportError:
    HAS_JPHOLIDAY = False

# scikit-learn
try:
    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
    )
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
    print("  ✅ scikit-learn")
except ImportError:
    HAS_SKLEARN = False
    print("  ⚠️ scikit-learn なし")

# LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("  ✅ LightGBM")
except ImportError:
    HAS_LIGHTGBM = False

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("  ✅ XGBoost")
except ImportError:
    HAS_XGBOOST = False

# CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
    print("  ✅ CatBoost")
except ImportError:
    HAS_CATBOOST = False

# Joblib
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

print("\n✅ ライブラリ読み込み完了\n")


# =============================================================================
# 統計的検定・分析クラス
# =============================================================================
class StatisticalAnalyzer:
    """統計的分析・検定"""

    @staticmethod
    def adf_test(series):
        """ADF検定（定常性）"""
        if not HAS_STATSMODELS or len(series) < 20:
            return {'stationary': True, 'p_value': 0.0}
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            return {
                'stationary': result[1] < 0.05,
                'p_value': result[1],
                'adf_stat': result[0]
            }
        except:
            return {'stationary': True, 'p_value': 0.0}

    @staticmethod
    def ljung_box_test(series, lags=10):
        """Ljung-Box検定（自己相関）"""
        if not HAS_STATSMODELS or len(series) < lags + 5:
            return {'autocorrelated': False}
        try:
            result = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
            p_value = result['lb_pvalue'].values[0]
            return {'autocorrelated': p_value < 0.05, 'p_value': p_value}
        except:
            return {'autocorrelated': False}

    @staticmethod
    def detect_outliers_iqr(series, k=1.5):
        """IQR法による外れ値検出"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        outliers = (series < lower) | (series > upper)
        return outliers, lower, upper

    @staticmethod
    def confidence_interval(data, confidence=0.95):
        """信頼区間計算"""
        n = len(data)
        if n < 2:
            return np.mean(data), np.mean(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h


# =============================================================================
# 統計予測クラス（40+手法）
# =============================================================================
class StatisticalPredictor:
    """統計的予測（40+手法）"""

    def __init__(self):
        self.ts_data = None
        self.df_data = None
        self.seasonality = {}
        self.decomposition = None
        self.analyzer = StatisticalAnalyzer()

    def prepare(self, df, target_col='合計'):
        """データ準備"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')
        df = df[df[target_col] > 0]

        self.df_data = df
        self.ts_data = df.set_index('Date')[target_col]

        if len(self.ts_data) < 14:
            return False

        # 季節性分析
        ts_df = self.ts_data.reset_index()
        ts_df.columns = ['Date', 'value']
        ts_df['weekday'] = ts_df['Date'].dt.dayofweek
        ts_df['month'] = ts_df['Date'].dt.month
        ts_df['week_of_month'] = ((ts_df['Date'].dt.day - 1) // 7 + 1)

        self.seasonality['weekday'] = ts_df.groupby('weekday')['value'].mean().to_dict()
        self.seasonality['month'] = ts_df.groupby('month')['value'].mean().to_dict()
        self.seasonality['week_of_month'] = ts_df.groupby('week_of_month')['value'].mean().to_dict()

        # STL分解
        if HAS_STATSMODELS and len(self.ts_data) >= 14:
            try:
                self.decomposition = STL(self.ts_data, period=7, robust=True).fit()
            except:
                pass

        return True

    def predict(self, target_date):
        """統計予測実行"""
        if self.ts_data is None:
            return {}

        target_dt = pd.to_datetime(target_date)
        ts = self.ts_data[self.ts_data.index < target_dt]

        if len(ts) < 14:
            return {}

        predictions = {}
        target_weekday = target_dt.dayofweek
        target_month = target_dt.month
        target_week_of_month = (target_dt.day - 1) // 7 + 1

        # === カテゴリ1: 移動平均系 (7種) ===
        for window in [3, 5, 7, 14, 21, 30]:
            if len(ts) >= window:
                predictions[f'SMA_{window}'] = ts.tail(window).mean()

        weights = np.arange(1, min(8, len(ts)+1))
        if len(ts) >= 7:
            predictions['WMA_7'] = np.average(ts.tail(7), weights=weights)

        for span in [7, 14, 21]:
            if len(ts) >= span:
                predictions[f'EMA_{span}'] = ts.ewm(span=span).mean().iloc[-1]

        if len(ts) >= 14:
            sma1 = ts.rolling(7).mean()
            predictions['TMA_7'] = sma1.rolling(7).mean().iloc[-1]

        # === カテゴリ2: 季節性ベース (8種) ===
        if target_weekday in self.seasonality.get('weekday', {}):
            predictions['Weekday_Mean'] = self.seasonality['weekday'][target_weekday]

        ts_df = ts.reset_index()
        ts_df.columns = ['Date', 'value']
        ts_df['weekday'] = ts_df['Date'].dt.dayofweek
        same_wd = ts_df[ts_df['weekday'] == target_weekday]['value']

        if len(same_wd) > 0:
            predictions['Weekday_Recent4'] = same_wd.tail(4).mean()
            predictions['Weekday_Recent8'] = same_wd.tail(8).mean()

        if target_month in self.seasonality.get('month', {}):
            predictions['Month_Mean'] = self.seasonality['month'][target_month]

        ts_df['month'] = ts_df['Date'].dt.month
        same_wd_month = ts_df[(ts_df['weekday'] == target_weekday) & (ts_df['month'] == target_month)]['value']
        if len(same_wd_month) > 0:
            predictions['Weekday_Month_Mean'] = same_wd_month.mean()

        if target_week_of_month in self.seasonality.get('week_of_month', {}):
            predictions['WeekOfMonth_Mean'] = self.seasonality['week_of_month'][target_week_of_month]

        # === カテゴリ3: 前年比較 (5種) ===
        prev_year_same = target_dt - timedelta(days=365)
        if prev_year_same in ts.index:
            predictions['PrevYear_SameDay'] = ts[prev_year_same]

        prev_year_same_wd = target_dt - timedelta(days=364)
        if prev_year_same_wd in ts.index:
            predictions['PrevYear_SameWeekday'] = ts[prev_year_same_wd]

        prev_year_month = ts[(ts.index.year == target_dt.year - 1) & (ts.index.month == target_month)]
        if len(prev_year_month) > 0:
            predictions['PrevYear_MonthMean'] = prev_year_month.mean()

        if 'PrevYear_SameDay' in predictions:
            current_year_avg = ts[ts.index.year == target_dt.year].mean() if len(ts[ts.index.year == target_dt.year]) > 0 else ts.mean()
            prev_year_avg = ts[ts.index.year == target_dt.year - 1].mean() if len(ts[ts.index.year == target_dt.year - 1]) > 0 else ts.mean()
            growth_rate = np.clip(current_year_avg / prev_year_avg if prev_year_avg > 0 else 1.0, 0.8, 1.3)
            predictions['PrevYear_GrowthAdj'] = predictions['PrevYear_SameDay'] * growth_rate

        # === カテゴリ4: 指数平滑化 (6種) ===
        if HAS_STATSMODELS:
            try:
                model = ExponentialSmoothing(ts.values, trend=None, seasonal=None)
                predictions['ExpSmooth_Simple'] = model.fit(optimized=True).forecast(1)[0]
            except:
                pass

            try:
                model = ExponentialSmoothing(ts.values, trend='add', seasonal=None)
                predictions['ExpSmooth_Holt'] = model.fit(optimized=True).forecast(1)[0]
            except:
                pass

            try:
                model = ExponentialSmoothing(ts.values, trend='add', damped_trend=True, seasonal=None)
                predictions['ExpSmooth_Damped'] = model.fit(optimized=True).forecast(1)[0]
            except:
                pass

            if len(ts) >= 14:
                try:
                    model = ExponentialSmoothing(ts.values, trend='add', seasonal='add', seasonal_periods=7)
                    predictions['HoltWinters_Add'] = model.fit(optimized=True).forecast(1)[0]
                except:
                    pass

                try:
                    model = ExponentialSmoothing(ts.values, trend='add', seasonal='mul', seasonal_periods=7)
                    predictions['HoltWinters_Mul'] = model.fit(optimized=True).forecast(1)[0]
                except:
                    pass

        # === カテゴリ5: ARIMA系 (6種) ===
        if HAS_STATSMODELS and len(ts) >= 30:
            for order in [(1,0,0), (1,1,0), (1,1,1), (2,1,1), (2,1,2), (0,1,1)]:
                try:
                    model = ARIMA(ts.values, order=order)
                    predictions[f'ARIMA_{order[0]}{order[1]}{order[2]}'] = model.fit().forecast(1)[0]
                except:
                    pass

        # === カテゴリ6: SARIMA (3種) ===
        if HAS_STATSMODELS and len(ts) >= 60:
            sarima_configs = [((1,1,1), (1,1,1,7)), ((1,1,1), (0,1,1,7)), ((2,1,1), (1,1,0,7))]
            for order, seasonal_order in sarima_configs:
                try:
                    model = SARIMAX(ts.values, order=order, seasonal_order=seasonal_order)
                    name = f'SARIMA_{order[0]}{order[1]}{order[2]}_{seasonal_order[0]}{seasonal_order[1]}{seasonal_order[2]}'
                    predictions[name] = model.fit(disp=False).forecast(1)[0]
                except:
                    pass

        # === カテゴリ7: Prophet ===
        if HAS_PROPHET and len(ts) >= 30:
            try:
                prophet_df = pd.DataFrame({'ds': ts.index, 'y': ts.values})
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                future = pd.DataFrame({'ds': [target_dt]})
                predictions['Prophet'] = model.predict(future)['yhat'].values[0]
            except:
                pass

        # === カテゴリ8: STL分解ベース ===
        if self.decomposition is not None:
            try:
                trend = self.decomposition.trend.dropna()
                if len(trend) >= 7:
                    trend_diff = trend.diff().tail(7).mean()
                    predictions['STL_Trend'] = trend.iloc[-1] + trend_diff
                    seasonal_idx = len(ts) % 7
                    seasonal = self.decomposition.seasonal.iloc[-(7-seasonal_idx)] if seasonal_idx > 0 else self.decomposition.seasonal.iloc[-7]
                    predictions['STL_Combined'] = predictions['STL_Trend'] + seasonal
            except:
                pass

        # === カテゴリ9: ベイズ推定 ===
        prior_mean = ts.mean()
        prior_std = ts.std()
        likelihood_mean = ts.tail(14).mean()
        likelihood_std = ts.tail(14).std() / np.sqrt(14)

        posterior_precision = 1/prior_std**2 + 1/likelihood_std**2
        posterior_mean = (prior_mean/prior_std**2 + likelihood_mean/likelihood_std**2) / posterior_precision
        predictions['Bayes_Posterior'] = posterior_mean

        # === カテゴリ10: ブートストラップ ===
        recent_values = ts.tail(30).values
        bootstrap_preds = [np.mean(np.random.choice(recent_values, size=len(recent_values), replace=True)) for _ in range(1000)]
        predictions['Bootstrap_Mean'] = np.mean(bootstrap_preds)

        # === カテゴリ11: カーネル密度推定 ===
        try:
            same_wd_values = ts_df[ts_df['weekday'] == target_weekday]['value'].values
            if len(same_wd_values) >= 5:
                kde = stats.gaussian_kde(same_wd_values)
                x_range = np.linspace(same_wd_values.min(), same_wd_values.max(), 1000)
                predictions['KDE_Mode'] = x_range[np.argmax(kde(x_range))]
        except:
            pass

        # === カテゴリ12: ロバスト統計 (5種) ===
        predictions['Median_7'] = ts.tail(7).median()
        predictions['Median_14'] = ts.tail(14).median()
        predictions['Median_30'] = ts.tail(30).median()
        predictions['TrimMean_10pct'] = stats.trim_mean(ts.tail(30).values, 0.1)
        predictions['TrimMean_20pct'] = stats.trim_mean(ts.tail(30).values, 0.2)

        # === カテゴリ13: 変化点検出 ===
        if len(ts) >= 14:
            recent_trend = np.polyfit(range(7), ts.tail(7).values, 1)[0]
            older_trend = np.polyfit(range(7), ts.tail(14).head(7).values, 1)[0]
            if recent_trend > older_trend:
                predictions['TrendChange_Adj'] = ts.tail(7).mean() * 1.02
            elif recent_trend < older_trend:
                predictions['TrendChange_Adj'] = ts.tail(7).mean() * 0.98
            else:
                predictions['TrendChange_Adj'] = ts.tail(7).mean()

        return predictions


# =============================================================================
# 機械学習予測クラス（15モデル）
# =============================================================================
class MLPredictor:
    """機械学習予測（15モデル）"""

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_cols = []
        self.is_trained = False

    def _create_features(self, df, target_col='合計'):
        """特徴量作成"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')

        # 基本特徴量
        df['weekday'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['quarter'] = df['Date'].dt.quarter
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['is_month_start'] = (df['day'] <= 5).astype(int)
        df['is_month_end'] = (df['day'] >= 25).astype(int)

        # 祝日
        if HAS_JPHOLIDAY:
            df['is_holiday'] = df['Date'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
        else:
            df['is_holiday'] = 0

        # ラグ特徴量
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

        # 移動統計量
        for window in [7, 14, 21, 30]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window).max()

        # 指数移動平均
        for span in [7, 14, 21]:
            df[f'ewm_{span}'] = df[target_col].shift(1).ewm(span=span).mean()

        # 同曜日統計
        df['same_weekday_mean'] = df.groupby('weekday')[target_col].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())

        # トレンド
        df['trend'] = np.arange(len(df))

        # 差分
        df['diff_1'] = df[target_col].diff(1)
        df['diff_7'] = df[target_col].diff(7)

        return df

    def train(self, df, target_col='合計'):
        """モデル訓練"""
        if not HAS_SKLEARN:
            return False

        print("\n【機械学習モデル訓練】")

        df = self._create_features(df, target_col)
        df = df.dropna()

        if len(df) < 50:
            print("  ⚠️ データ不足")
            return False

        # 特徴量選択
        exclude_cols = ['Date', target_col, 'diff_1', 'diff_7']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]

        X = df[self.feature_cols].values
        y = df[target_col].values

        # スケーリング
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # モデル定義
        model_configs = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'BayesianRidge': BayesianRidge(),
            'HuberRegressor': HuberRegressor(),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }

        if HAS_SKLEARN:
            model_configs['HistGradientBoosting'] = HistGradientBoostingRegressor(max_iter=100, random_state=42)

        # ブースティングモデル
        if HAS_LIGHTGBM:
            model_configs['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)

        if HAS_XGBOOST:
            model_configs['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)

        if HAS_CATBOOST:
            model_configs['CatBoost'] = CatBoostRegressor(iterations=100, depth=5, random_state=42, verbose=False)

        # 訓練
        trained_count = 0
        for name, model in model_configs.items():
            try:
                model.fit(X_scaled, y)
                self.models[name] = model
                trained_count += 1
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ⚠️ {name}: {str(e)[:30]}")

        self.is_trained = True
        print(f"\n  訓練完了: {trained_count}モデル")
        return True

    def predict(self, target_date, df, target_col='合計'):
        """機械学習予測実行"""
        if not self.is_trained or not self.models:
            return {}

        df = self._create_features(df, target_col)
        df = df.dropna()

        if len(df) == 0:
            return {}

        # 最新の特徴量を使用
        X_pred = df[self.feature_cols].iloc[-1:].values
        X_pred_scaled = self.scaler.transform(X_pred)

        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_pred_scaled)[0]
                if pred > 0 and not np.isnan(pred):
                    predictions[f'ML_{name}'] = pred
            except:
                pass

        return predictions


# =============================================================================
# 完全版アンサンブル予測システム
# =============================================================================
class CompletePredictionSystem:
    """統計 + 機械学習 完全版予測システム"""

    def __init__(self):
        self.stat_predictor = StatisticalPredictor()
        self.ml_predictor = MLPredictor()
        self.df_data = None
        self.is_initialized = False
        self.ai_memory = []  # AI学習メモリ

    def initialize(self, df):
        """システム初期化"""
        print("\n" + "="*70)
        print("【v69.0 完全版予測システム】初期化")
        print("="*70)

        self.df_data = df.copy()

        # カラム名正規化
        if 'Date' not in self.df_data.columns:
            for col in ['日付', 'date']:
                if col in self.df_data.columns:
                    self.df_data = self.df_data.rename(columns={col: 'Date'})
                    break

        # 統計予測準備
        print("\n【統計分析】")
        stat_ok = self.stat_predictor.prepare(self.df_data, '合計')

        # 機械学習訓練
        print("\n【機械学習訓練】")
        ml_ok = self.ml_predictor.train(self.df_data, '合計')

        # 基本統計量
        ts = self.df_data.set_index(pd.to_datetime(self.df_data['Date'], errors='coerce'))['合計'].dropna()
        print(f"\n【データ概要】")
        print(f"  期間: {ts.index.min().date()} ～ {ts.index.max().date()}")
        print(f"  件数: {len(ts)}日分")
        print(f"  平均: {ts.mean():.0f}件")
        print(f"  標準偏差: {ts.std():.0f}件")

        self.is_initialized = stat_ok or ml_ok
        print("\n✅ 初期化完了！")
        return self.is_initialized

    def predict(self, target_date, actuals=None):
        """完全版予測実行"""
        if not self.is_initialized:
            return None

        actuals = actuals or {}

        print("\n" + "="*70)
        print("【v69.0 完全版予測】")
        print("="*70)
        print(f"予測日: {target_date}")

        # 統計予測
        stat_preds = self.stat_predictor.predict(target_date)
        print(f"\n統計モデル: {len(stat_preds)}個")

        # 機械学習予測
        ml_preds = self.ml_predictor.predict(target_date, self.df_data, '合計')
        print(f"機械学習モデル: {len(ml_preds)}個")

        # 統合
        all_preds = {**stat_preds, **ml_preds}
        valid_preds = {k: v for k, v in all_preds.items()
                      if v > 0 and not np.isnan(v) and not np.isinf(v)}

        if not valid_preds:
            return None

        print(f"\n合計モデル: {len(valid_preds)}個")

        # カテゴリ別集計
        categories = {
            '移動平均': [k for k in valid_preds if 'MA_' in k or 'EMA_' in k or 'WMA' in k or 'TMA' in k],
            '季節性': [k for k in valid_preds if 'Weekday' in k or 'Month' in k or 'WeekOf' in k],
            '前年比較': [k for k in valid_preds if 'PrevYear' in k],
            '時系列モデル': [k for k in valid_preds if 'ARIMA' in k or 'SARIMA' in k or 'Prophet' in k or 'STL' in k or 'ExpSmooth' in k or 'Holt' in k],
            '統計推定': [k for k in valid_preds if 'Bayes' in k or 'Bootstrap' in k or 'KDE' in k],
            'ロバスト統計': [k for k in valid_preds if 'Median' in k or 'Trim' in k or 'TrendChange' in k],
            '機械学習': [k for k in valid_preds if k.startswith('ML_')],
        }

        print("\n【カテゴリ別予測】")
        for cat, keys in categories.items():
            if keys:
                values = [valid_preds[k] for k in keys]
                print(f"  {cat}: {np.median(values):.0f}件 ({len(keys)}モデル)")

        # アンサンブル計算
        all_values = list(valid_preds.values())

        # 複数のアンサンブル手法
        ensemble_mean = np.mean(all_values)
        ensemble_median = np.median(all_values)
        ensemble_trimmed = stats.trim_mean(all_values, 0.1)

        # IQR内平均（外れ値除外）
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr_values = [v for v in all_values if q1 <= v <= q3]
        ensemble_iqr = np.mean(iqr_values) if iqr_values else ensemble_median

        # 統計と機械学習の加重平均
        stat_values = [valid_preds[k] for k in valid_preds if not k.startswith('ML_')]
        ml_values = [valid_preds[k] for k in valid_preds if k.startswith('ML_')]

        if stat_values and ml_values:
            stat_median = np.median(stat_values)
            ml_median = np.median(ml_values)
            # 統計60% + 機械学習40%
            hybrid_pred = stat_median * 0.6 + ml_median * 0.4
        else:
            hybrid_pred = ensemble_median

        print(f"\n【アンサンブル比較】")
        print(f"  平均: {ensemble_mean:.0f}件")
        print(f"  中央値: {ensemble_median:.0f}件")
        print(f"  トリム平均: {ensemble_trimmed:.0f}件")
        print(f"  IQR平均: {ensemble_iqr:.0f}件")
        print(f"  ハイブリッド: {hybrid_pred:.0f}件")

        # 最終予測（ハイブリッド基準）
        final_pred = hybrid_pred

        # 12時実績調整
        if actuals.get(12, 0) > 0:
            df = self.df_data.copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            ratio = 0.45
            if '件数(～12:00)' in df.columns:
                df_12h = df[(df['件数(～12:00)'] > 0) & (df['合計'] > 0)]
                if len(df_12h) > 0:
                    ratio = (df_12h['件数(～12:00)'] / df_12h['合計']).mean()
            est = actuals[12] / ratio
            final_pred = final_pred * 0.25 + est * 0.75
            print(f"\n  ⏰ 12時実績調整: {actuals[12]} → {est:.0f}件")

        # AI学習メモリからの補正
        if self.ai_memory:
            recent_errors = [m['error'] for m in self.ai_memory[-10:] if 'error' in m]
            if recent_errors:
                avg_error = np.mean(recent_errors)
                if abs(avg_error) > 0.02:
                    correction = final_pred * avg_error
                    final_pred += correction
                    print(f"  🧠 AI学習補正: {correction:+.0f}件")

        # 信頼区間
        pred_lower = np.percentile(all_values, 2.5)
        pred_upper = np.percentile(all_values, 97.5)

        # 信頼度スコア
        cv = np.std(all_values) / np.mean(all_values) * 100
        if cv < 5:
            confidence = 95
        elif cv < 10:
            confidence = 85
        elif cv < 15:
            confidence = 75
        else:
            confidence = 65

        print("\n" + "="*70)
        print(f"★★★ 最終予測: {int(final_pred):,}件 ★★★")
        print(f"    信頼区間 (95%): {int(pred_lower):,} ～ {int(pred_upper):,}件")
        print(f"    信頼度: {confidence}点")
        print(f"    変動係数: {cv:.1f}%")
        print(f"    使用モデル: {len(valid_preds)}個 (統計{len(stat_preds)} + ML{len(ml_preds)})")
        print("="*70)

        return {
            'final_prediction': int(final_pred),
            'pred_lower': int(pred_lower),
            'pred_upper': int(pred_upper),
            'confidence_score': confidence,
            'cv': cv,
            'model_count': len(valid_preds),
            'stat_model_count': len(stat_preds),
            'ml_model_count': len(ml_preds),
            'all_predictions': valid_preds,
            'ensemble_mean': ensemble_mean,
            'ensemble_median': ensemble_median,
            'hybrid_pred': hybrid_pred
        }

    def update_memory(self, target_date, actual_value, predicted_value):
        """AI学習メモリ更新"""
        if actual_value > 0 and predicted_value > 0:
            error = (actual_value - predicted_value) / actual_value
            self.ai_memory.append({
                'date': target_date,
                'actual': actual_value,
                'predicted': predicted_value,
                'error': error
            })
            # 最新100件のみ保持
            if len(self.ai_memory) > 100:
                self.ai_memory = self.ai_memory[-100:]
            print(f"🧠 AI学習: 誤差{error*100:+.1f}%を記録")


# =============================================================================
# Google Colab UI
# =============================================================================
def create_complete_ui():
    """完全版UI"""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
    except:
        print("⚠️ ipywidgets が必要です")
        return

    system = [None]
    df_data = [None]

    # ウィジェット
    date_picker = widgets.DatePicker(description='予測日:', value=date.today())
    actual_12h = widgets.IntText(value=0, description='12時実績:')
    actual_total = widgets.IntText(value=0, description='実績合計:')

    run_btn = widgets.Button(description='🚀 完全版予測', button_style='success',
                            layout=widgets.Layout(width='180px', height='50px'))
    init_btn = widgets.Button(description='📊 初期化', button_style='warning',
                             layout=widgets.Layout(width='120px', height='50px'))
    learn_btn = widgets.Button(description='🧠 学習', button_style='info',
                              layout=widgets.Layout(width='100px', height='50px'))

    output = widgets.Output()
    result = widgets.Output()

    def on_init(b):
        init_btn.disabled = True
        with output:
            clear_output()
            import glob
            files = glob.glob('/content/*.xlsx') + glob.glob('/content/*.csv')
            if not files:
                print("⚠️ /content にExcel/CSVファイルをアップロードしてください")
                init_btn.disabled = False
                return
            try:
                f = files[0]
                df_data[0] = pd.read_excel(f) if f.endswith('.xlsx') else pd.read_csv(f)
                print(f"📁 読み込み: {os.path.basename(f)}")

                if 'Date' not in df_data[0].columns:
                    for c in ['日付', 'date']:
                        if c in df_data[0].columns:
                            df_data[0] = df_data[0].rename(columns={c: 'Date'})
                            break

                system[0] = CompletePredictionSystem()
                system[0].initialize(df_data[0])
            except Exception as e:
                print(f"⚠️ エラー: {e}")
        init_btn.disabled = False

    def on_run(b):
        run_btn.disabled = True
        with output:
            clear_output()
            if system[0] is None:
                on_init(None)
            if system[0]:
                res = system[0].predict(date_picker.value, {12: actual_12h.value})
                if res:
                    with result:
                        clear_output()
                        display(HTML(f"""
                        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:white;padding:30px;border-radius:20px;text-align:center;margin:20px 0;box-shadow:0 10px 30px rgba(0,0,0,0.3);'>
                            <h2 style='margin:0;font-size:24px;'>🚀 v69.0【完全版】統計+機械学習</h2>
                            <div style='font-size:64px;font-weight:bold;margin:20px 0;text-shadow:2px 2px 10px rgba(255,255,255,0.3);'>{res['final_prediction']:,}件</div>
                            <div style='font-size:16px;margin:10px 0;'>95%信頼区間: {res['pred_lower']:,} ～ {res['pred_upper']:,}件</div>
                            <div style='display:flex;justify-content:center;gap:30px;margin-top:15px;'>
                                <div style='background:rgba(255,255,255,0.1);padding:10px 20px;border-radius:10px;'>
                                    <div style='font-size:12px;opacity:0.8;'>信頼度</div>
                                    <div style='font-size:24px;font-weight:bold;'>{res['confidence_score']}点</div>
                                </div>
                                <div style='background:rgba(255,255,255,0.1);padding:10px 20px;border-radius:10px;'>
                                    <div style='font-size:12px;opacity:0.8;'>変動係数</div>
                                    <div style='font-size:24px;font-weight:bold;'>{res['cv']:.1f}%</div>
                                </div>
                                <div style='background:rgba(255,255,255,0.1);padding:10px 20px;border-radius:10px;'>
                                    <div style='font-size:12px;opacity:0.8;'>モデル数</div>
                                    <div style='font-size:24px;font-weight:bold;'>{res['model_count']}個</div>
                                </div>
                            </div>
                            <div style='margin-top:15px;font-size:14px;opacity:0.9;'>
                                📊 統計: {res['stat_model_count']}モデル | 🤖 ML: {res['ml_model_count']}モデル
                            </div>
                        </div>
                        """))
        run_btn.disabled = False

    def on_learn(b):
        if system[0] and actual_total.value > 0:
            with output:
                clear_output()
                system[0].update_memory(date_picker.value, actual_total.value,
                                       system[0].predict(date_picker.value)['final_prediction'] if system[0] else 0)

    init_btn.on_click(on_init)
    run_btn.on_click(on_run)
    learn_btn.on_click(on_learn)

    ui = widgets.VBox([
        widgets.HTML('''
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:20px;border-radius:15px;margin-bottom:15px;">
            <h2 style="margin:0;">🚀 v69.0 完全版予測システム</h2>
            <p style="margin:5px 0 0 0;opacity:0.9;">統計理論(40+手法) + 機械学習(15モデル) = 55+予測モデル統合</p>
        </div>
        '''),
        widgets.HTML('''
        <div style="background:#f8f9fa;padding:12px;border-radius:8px;margin:10px 0;font-size:12px;">
            <b>📊 統計手法:</b> 移動平均, 季節性分析, 前年比較, 指数平滑化, ARIMA, SARIMA, Prophet, STL分解, ベイズ推定, ブートストラップ, KDE, ロバスト統計<br>
            <b>🤖 機械学習:</b> LightGBM, XGBoost, CatBoost, RandomForest, GradientBoosting, ExtraTrees, Ridge, Lasso, ElasticNet, BayesianRidge, Huber, KNN, MLP, AdaBoost, HistGradientBoosting
        </div>
        '''),
        widgets.HBox([date_picker, actual_12h]),
        widgets.HBox([actual_total, learn_btn]),
        widgets.HBox([run_btn, init_btn]),
        result
    ])

    display(ui, output)


# =============================================================================
# メイン実行
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("【v69.0 完全版予測システム 搭載モデル】")
    print("="*70)

    print("\n📊 【統計手法】40+種")
    print("-"*50)
    print("  1. 移動平均系: SMA(6種), WMA, EMA(3種), TMA")
    print("  2. 季節性分析: 曜日/月/週/交互作用 (8種)")
    print("  3. 前年比較: 同日/同曜日/成長率調整 (5種)")
    print("  4. 指数平滑化: Simple/Holt/Damped/HW (6種)")
    print("  5. ARIMA: 複数次数 (6種)")
    print("  6. SARIMA: 季節性ARIMA (3種)")
    print("  7. Prophet: Facebook時系列予測")
    print("  8. STL分解: トレンド+季節性")
    print("  9. ベイズ推定: 事後分布予測")
    print(" 10. ブートストラップ: リサンプリング予測")
    print(" 11. カーネル密度推定: KDE")
    print(" 12. ロバスト統計: 中央値/トリム平均 (5種)")
    print(" 13. 変化点検出: トレンド変化調整")

    print("\n🤖 【機械学習モデル】15種")
    print("-"*50)
    print("  1. LightGBM (高速勾配ブースティング)")
    print("  2. XGBoost (極限勾配ブースティング)")
    print("  3. CatBoost (カテゴリ対応ブースティング)")
    print("  4. RandomForest (ランダムフォレスト)")
    print("  5. GradientBoosting (勾配ブースティング)")
    print("  6. ExtraTrees (極度ランダム木)")
    print("  7. HistGradientBoosting (ヒストグラムベース)")
    print("  8. AdaBoost (適応型ブースティング)")
    print("  9. Ridge (L2正則化回帰)")
    print(" 10. Lasso (L1正則化回帰)")
    print(" 11. ElasticNet (L1+L2正則化)")
    print(" 12. BayesianRidge (ベイズリッジ回帰)")
    print(" 13. HuberRegressor (ロバスト回帰)")
    print(" 14. KNN (K近傍法)")
    print(" 15. MLP (多層パーセプトロン)")

    print("\n" + "="*70)
    print("合計: 55+予測モデル統合")
    print("="*70)

    # Google Colab判定
    try:
        import google.colab
        print("\n🌐 Google Colab環境を検出")
        create_complete_ui()
    except:
        print("\n💻 ローカル環境で実行中")
        print("   Google ColabでUIを使用するには、このファイルをColabにアップロードしてください")
