# =============================================================================
# v68.0 メタ認知AI予測システム【統計理論強化版】
# =============================================================================
# 統計的推論 + ベイズ推定 + 時系列分解 + 回帰分析 = 40+統計手法
# =============================================================================

print("="*70)
print("【v68.0 メタ認知AI予測システム】統計理論強化版")
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
# オプショナルライブラリ
# =============================================================================
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import statsmodels.api as sm
    HAS_STATSMODELS = True
    print("✅ statsmodels")
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
    print("✅ Prophet")
except ImportError:
    HAS_PROPHET = False

try:
    import jpholiday
    HAS_JPHOLIDAY = True
    print("✅ jpholiday")
except ImportError:
    HAS_JPHOLIDAY = False

try:
    from sklearn.linear_model import Ridge, BayesianRidge, HuberRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
    print("✅ scikit-learn")
except ImportError:
    HAS_SKLEARN = False

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
                'adf_stat': result[0],
                'critical_1pct': result[4]['1%'],
                'critical_5pct': result[4]['5%']
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
            return {
                'autocorrelated': p_value < 0.05,
                'p_value': p_value,
                'lb_stat': result['lb_stat'].values[0]
            }
        except:
            return {'autocorrelated': False}

    @staticmethod
    def calculate_acf_pacf(series, nlags=20):
        """ACF/PACF計算"""
        if not HAS_STATSMODELS or len(series) < nlags + 5:
            return None, None
        try:
            acf_vals = acf(series.dropna(), nlags=nlags)
            pacf_vals = pacf(series.dropna(), nlags=nlags)
            return acf_vals, pacf_vals
        except:
            return None, None

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
    def detect_outliers_zscore(series, threshold=3):
        """Z-score法による外れ値検出"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = z_scores > threshold
        return outliers

    @staticmethod
    def confidence_interval(data, confidence=0.95):
        """信頼区間計算"""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean - h, mean + h

    @staticmethod
    def prediction_interval(predictions, residuals, confidence=0.95):
        """予測区間計算"""
        n = len(residuals)
        se = np.std(residuals) * np.sqrt(1 + 1/n)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        return t_val * se


# =============================================================================
# 高度統計予測クラス
# =============================================================================
class AdvancedStatisticalPredictor:
    """高度統計予測（40+手法）"""

    def __init__(self):
        self.ts_data = None
        self.df_data = None
        self.is_ready = False
        self.analyzer = StatisticalAnalyzer()
        self.decomposition = None
        self.seasonality = {}

    def prepare(self, df, target_col='合計'):
        """データ準備と統計分析"""
        print("\n" + "="*70)
        print("【統計分析・準備】")
        print("="*70)

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')
        df = df[df[target_col] > 0]

        self.df_data = df
        self.ts_data = df.set_index('Date')[target_col]

        if len(self.ts_data) < 30:
            print("⚠️ データ不足")
            return False

        print(f"  データ件数: {len(self.ts_data)}日分")
        print(f"  期間: {self.ts_data.index.min().date()} ～ {self.ts_data.index.max().date()}")

        # 基本統計量
        print(f"\n【基本統計量】")
        print(f"  平均: {self.ts_data.mean():.1f}")
        print(f"  中央値: {self.ts_data.median():.1f}")
        print(f"  標準偏差: {self.ts_data.std():.1f}")
        print(f"  変動係数: {self.ts_data.std()/self.ts_data.mean()*100:.1f}%")
        print(f"  歪度: {self.ts_data.skew():.2f}")
        print(f"  尖度: {self.ts_data.kurtosis():.2f}")

        # 定常性検定
        adf_result = self.analyzer.adf_test(self.ts_data)
        print(f"\n【定常性検定 (ADF)】")
        print(f"  定常性: {'あり' if adf_result['stationary'] else 'なし'}")
        print(f"  p値: {adf_result.get('p_value', 'N/A'):.4f}")

        # 自己相関検定
        lb_result = self.analyzer.ljung_box_test(self.ts_data)
        print(f"\n【自己相関検定 (Ljung-Box)】")
        print(f"  自己相関: {'あり' if lb_result['autocorrelated'] else 'なし'}")

        # 季節性分析
        self._analyze_seasonality()

        # 外れ値検出
        outliers, lower, upper = self.analyzer.detect_outliers_iqr(self.ts_data)
        n_outliers = outliers.sum()
        print(f"\n【外れ値検出 (IQR法)】")
        print(f"  外れ値数: {n_outliers}件 ({n_outliers/len(self.ts_data)*100:.1f}%)")
        print(f"  正常範囲: {lower:.0f} ～ {upper:.0f}")

        self.is_ready = True
        print("\n✅ 統計分析完了")
        return True

    def _analyze_seasonality(self):
        """季節性分析"""
        print(f"\n【季節性分析】")

        # 曜日別平均
        ts_df = self.ts_data.reset_index()
        ts_df.columns = ['Date', 'value']
        ts_df['weekday'] = ts_df['Date'].dt.dayofweek
        ts_df['month'] = ts_df['Date'].dt.month
        ts_df['week_of_month'] = ((ts_df['Date'].dt.day - 1) // 7 + 1)

        weekday_stats = ts_df.groupby('weekday')['value'].agg(['mean', 'std', 'count'])
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']

        print("  曜日別平均:")
        for wd, row in weekday_stats.iterrows():
            print(f"    {weekday_names[wd]}: {row['mean']:.0f} (±{row['std']:.0f})")

        self.seasonality['weekday'] = ts_df.groupby('weekday')['value'].mean().to_dict()

        # 月別平均
        month_stats = ts_df.groupby('month')['value'].agg(['mean', 'std'])
        self.seasonality['month'] = ts_df.groupby('month')['value'].mean().to_dict()

        # 月内週別
        self.seasonality['week_of_month'] = ts_df.groupby('week_of_month')['value'].mean().to_dict()

        # STL分解
        if HAS_STATSMODELS and len(self.ts_data) >= 14:
            try:
                self.decomposition = STL(self.ts_data, period=7, robust=True).fit()
                trend_strength = 1 - np.var(self.decomposition.resid) / np.var(self.decomposition.trend + self.decomposition.resid)
                seasonal_strength = 1 - np.var(self.decomposition.resid) / np.var(self.decomposition.seasonal + self.decomposition.resid)
                print(f"\n  STL分解:")
                print(f"    トレンド強度: {max(0, trend_strength):.2%}")
                print(f"    季節性強度: {max(0, seasonal_strength):.2%}")
            except:
                pass

    def predict(self, target_date):
        """統計予測（40+手法）"""
        if not self.is_ready:
            return {}, {}

        target_dt = pd.to_datetime(target_date)
        ts = self.ts_data[self.ts_data.index < target_dt]

        if len(ts) < 14:
            return {}, {}

        predictions = {}
        confidence_intervals = {}
        errors = {}

        # ===================================================================
        # カテゴリ1: 移動平均系（7手法）
        # ===================================================================
        # 1-1. 単純移動平均
        for window in [3, 5, 7, 14, 21, 30]:
            predictions[f'SMA_{window}'] = ts.tail(window).mean()

        # 1-2. 加重移動平均
        weights = np.arange(1, 8)
        predictions['WMA_7'] = np.average(ts.tail(7), weights=weights)

        # 1-3. 指数移動平均
        for span in [7, 14, 21]:
            predictions[f'EMA_{span}'] = ts.ewm(span=span).mean().iloc[-1]

        # 1-4. 三角移動平均
        n = 7
        sma1 = ts.rolling(n).mean()
        predictions['TMA_7'] = sma1.rolling(n).mean().iloc[-1]

        # ===================================================================
        # カテゴリ2: 季節性ベース（8手法）
        # ===================================================================
        target_weekday = target_dt.dayofweek
        target_month = target_dt.month
        target_week_of_month = (target_dt.day - 1) // 7 + 1

        # 2-1. 曜日平均
        if target_weekday in self.seasonality.get('weekday', {}):
            predictions['Weekday_Mean'] = self.seasonality['weekday'][target_weekday]

        # 2-2. 直近N週の同曜日
        ts_df = ts.reset_index()
        ts_df.columns = ['Date', 'value']
        ts_df['weekday'] = ts_df['Date'].dt.dayofweek
        same_wd = ts_df[ts_df['weekday'] == target_weekday]['value']

        if len(same_wd) > 0:
            predictions['Weekday_Recent4'] = same_wd.tail(4).mean()
            predictions['Weekday_Recent8'] = same_wd.tail(8).mean()

            # 信頼区間付き
            if len(same_wd) >= 4:
                ci_low, ci_high = self.analyzer.confidence_interval(same_wd.tail(8).values)
                confidence_intervals['Weekday_Recent8'] = (ci_low, ci_high)

        # 2-3. 月別平均
        if target_month in self.seasonality.get('month', {}):
            predictions['Month_Mean'] = self.seasonality['month'][target_month]

        # 2-4. 曜日×月の交互作用
        ts_df['month'] = ts_df['Date'].dt.month
        same_wd_month = ts_df[(ts_df['weekday'] == target_weekday) & (ts_df['month'] == target_month)]['value']
        if len(same_wd_month) > 0:
            predictions['Weekday_Month_Mean'] = same_wd_month.mean()

        # 2-5. 月内週別
        if target_week_of_month in self.seasonality.get('week_of_month', {}):
            predictions['WeekOfMonth_Mean'] = self.seasonality['week_of_month'][target_week_of_month]

        # ===================================================================
        # カテゴリ3: 前年比較（5手法）
        # ===================================================================
        # 3-1. 前年同日
        prev_year_same = target_dt - timedelta(days=365)
        if prev_year_same in ts.index:
            predictions['PrevYear_SameDay'] = ts[prev_year_same]

        # 3-2. 前年同週同曜日
        prev_year_same_wd = target_dt - timedelta(days=364)
        if prev_year_same_wd in ts.index:
            predictions['PrevYear_SameWeekday'] = ts[prev_year_same_wd]

        # 3-3. 前年同月平均
        prev_year_month = ts[(ts.index.year == target_dt.year - 1) & (ts.index.month == target_month)]
        if len(prev_year_month) > 0:
            predictions['PrevYear_MonthMean'] = prev_year_month.mean()

        # 3-4. 前年同日±3日平均
        for delta in range(-3, 4):
            check_date = prev_year_same + timedelta(days=delta)
            if check_date in ts.index:
                nearby = [ts[check_date + timedelta(days=d)] for d in range(-3, 4)
                         if check_date + timedelta(days=d) in ts.index]
                if nearby:
                    predictions['PrevYear_Nearby'] = np.mean(nearby)
                break

        # 3-5. 年次成長率調整
        if 'PrevYear_SameDay' in predictions:
            current_year_avg = ts[ts.index.year == target_dt.year].mean() if len(ts[ts.index.year == target_dt.year]) > 0 else ts.mean()
            prev_year_avg = ts[ts.index.year == target_dt.year - 1].mean() if len(ts[ts.index.year == target_dt.year - 1]) > 0 else ts.mean()
            growth_rate = current_year_avg / prev_year_avg if prev_year_avg > 0 else 1.0
            growth_rate = np.clip(growth_rate, 0.8, 1.3)
            predictions['PrevYear_GrowthAdj'] = predictions['PrevYear_SameDay'] * growth_rate

        # ===================================================================
        # カテゴリ4: 指数平滑化（6手法）
        # ===================================================================
        if HAS_STATSMODELS:
            # 4-1. 単純指数平滑化
            try:
                model = ExponentialSmoothing(ts.values, trend=None, seasonal=None)
                fit = model.fit(optimized=True)
                predictions['ExpSmooth_Simple'] = fit.forecast(1)[0]
            except:
                pass

            # 4-2. ダブル指数平滑化（Holtの線形トレンド）
            try:
                model = ExponentialSmoothing(ts.values, trend='add', seasonal=None)
                fit = model.fit(optimized=True)
                predictions['ExpSmooth_Holt'] = fit.forecast(1)[0]
            except:
                pass

            # 4-3. 減衰トレンド
            try:
                model = ExponentialSmoothing(ts.values, trend='add', damped_trend=True, seasonal=None)
                fit = model.fit(optimized=True)
                predictions['ExpSmooth_Damped'] = fit.forecast(1)[0]
            except:
                pass

            # 4-4-6. Holt-Winters（加法/乗法）
            if len(ts) >= 14:
                try:
                    model = ExponentialSmoothing(ts.values, trend='add', seasonal='add', seasonal_periods=7)
                    fit = model.fit(optimized=True)
                    predictions['HoltWinters_Add'] = fit.forecast(1)[0]
                except:
                    pass

                try:
                    model = ExponentialSmoothing(ts.values, trend='add', seasonal='mul', seasonal_periods=7)
                    fit = model.fit(optimized=True)
                    predictions['HoltWinters_Mul'] = fit.forecast(1)[0]
                except:
                    pass

        # ===================================================================
        # カテゴリ5: ARIMA系（6手法）
        # ===================================================================
        if HAS_STATSMODELS and len(ts) >= 30:
            arima_orders = [(1,0,0), (1,1,0), (1,1,1), (2,1,1), (2,1,2), (0,1,1)]
            for order in arima_orders:
                try:
                    model = ARIMA(ts.values, order=order)
                    fit = model.fit()
                    predictions[f'ARIMA_{order[0]}{order[1]}{order[2]}'] = fit.forecast(1)[0]
                except:
                    pass

        # ===================================================================
        # カテゴリ6: SARIMA（3手法）
        # ===================================================================
        if HAS_STATSMODELS and len(ts) >= 60:
            sarima_configs = [
                ((1,1,1), (1,1,1,7)),
                ((1,1,1), (0,1,1,7)),
                ((2,1,1), (1,1,0,7))
            ]
            for order, seasonal_order in sarima_configs:
                try:
                    model = SARIMAX(ts.values, order=order, seasonal_order=seasonal_order)
                    fit = model.fit(disp=False)
                    name = f'SARIMA_{order[0]}{order[1]}{order[2]}_{seasonal_order[0]}{seasonal_order[1]}{seasonal_order[2]}'
                    predictions[name] = fit.forecast(1)[0]
                except:
                    pass

        # ===================================================================
        # カテゴリ7: Prophet
        # ===================================================================
        if HAS_PROPHET and len(ts) >= 30:
            try:
                prophet_df = pd.DataFrame({'ds': ts.index, 'y': ts.values})
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                future = pd.DataFrame({'ds': [target_dt]})
                forecast = model.predict(future)
                predictions['Prophet'] = forecast['yhat'].values[0]
                confidence_intervals['Prophet'] = (forecast['yhat_lower'].values[0], forecast['yhat_upper'].values[0])
            except:
                pass

        # ===================================================================
        # カテゴリ8: STL分解ベース
        # ===================================================================
        if self.decomposition is not None:
            try:
                # トレンド外挿
                trend = self.decomposition.trend.dropna()
                if len(trend) >= 7:
                    trend_diff = trend.diff().tail(7).mean()
                    predictions['STL_Trend'] = trend.iloc[-1] + trend_diff

                # 季節成分
                seasonal_idx = len(ts) % 7
                seasonal = self.decomposition.seasonal.iloc[-(7-seasonal_idx)] if seasonal_idx > 0 else self.decomposition.seasonal.iloc[-7]

                # トレンド + 季節性
                if 'STL_Trend' in predictions:
                    predictions['STL_Combined'] = predictions['STL_Trend'] + seasonal
            except:
                pass

        # ===================================================================
        # カテゴリ9: 回帰分析ベース
        # ===================================================================
        if HAS_SKLEARN and len(ts) >= 30:
            # 特徴量作成
            ts_df = ts.reset_index()
            ts_df.columns = ['Date', 'value']
            ts_df['weekday'] = ts_df['Date'].dt.dayofweek
            ts_df['month'] = ts_df['Date'].dt.month
            ts_df['day'] = ts_df['Date'].dt.day
            ts_df['week_of_month'] = ((ts_df['day'] - 1) // 7 + 1)
            ts_df['day_of_year'] = ts_df['Date'].dt.dayofyear
            ts_df['trend'] = np.arange(len(ts_df))

            # ラグ特徴量
            for lag in [1, 7, 14]:
                ts_df[f'lag_{lag}'] = ts_df['value'].shift(lag)

            ts_df = ts_df.dropna()

            if len(ts_df) >= 20:
                feature_cols = ['weekday', 'month', 'week_of_month', 'day_of_year', 'trend', 'lag_1', 'lag_7', 'lag_14']
                X = ts_df[feature_cols].values
                y = ts_df['value'].values

                # 予測用特徴量
                X_pred = np.array([[
                    target_weekday, target_month, target_week_of_month,
                    target_dt.timetuple().tm_yday,
                    len(ts_df) + 1,
                    ts.iloc[-1], ts.iloc[-7] if len(ts) >= 7 else ts.iloc[-1],
                    ts.iloc[-14] if len(ts) >= 14 else ts.iloc[-1]
                ]])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_pred_scaled = scaler.transform(X_pred)

                # Ridge回帰
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(X_scaled, y)
                    predictions['Ridge_Regression'] = model.predict(X_pred_scaled)[0]
                except:
                    pass

                # ベイズリッジ回帰
                try:
                    model = BayesianRidge()
                    model.fit(X_scaled, y)
                    pred, std = model.predict(X_pred_scaled, return_std=True)
                    predictions['BayesRidge'] = pred[0]
                    confidence_intervals['BayesRidge'] = (pred[0] - 1.96*std[0], pred[0] + 1.96*std[0])
                except:
                    pass

                # Huber回帰（ロバスト）
                try:
                    model = HuberRegressor()
                    model.fit(X_scaled, y)
                    predictions['Huber_Regression'] = model.predict(X_pred_scaled)[0]
                except:
                    pass

        # ===================================================================
        # カテゴリ10: ベイズ推定
        # ===================================================================
        # 事前分布: 過去データの統計量
        # 尤度: 直近データ
        # 事後分布: 更新された予測

        prior_mean = ts.mean()
        prior_std = ts.std()
        likelihood_mean = ts.tail(14).mean()
        likelihood_std = ts.tail(14).std() / np.sqrt(14)

        # ベイズ更新
        posterior_precision = 1/prior_std**2 + 1/likelihood_std**2
        posterior_mean = (prior_mean/prior_std**2 + likelihood_mean/likelihood_std**2) / posterior_precision
        posterior_std = 1 / np.sqrt(posterior_precision)

        predictions['Bayes_Posterior'] = posterior_mean
        confidence_intervals['Bayes_Posterior'] = (
            posterior_mean - 1.96 * posterior_std,
            posterior_mean + 1.96 * posterior_std
        )

        # ===================================================================
        # カテゴリ11: ブートストラップ予測
        # ===================================================================
        n_bootstrap = 1000
        recent_values = ts.tail(30).values
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(recent_values, size=len(recent_values), replace=True)
            bootstrap_preds.append(np.mean(sample))

        predictions['Bootstrap_Mean'] = np.mean(bootstrap_preds)
        confidence_intervals['Bootstrap'] = (
            np.percentile(bootstrap_preds, 2.5),
            np.percentile(bootstrap_preds, 97.5)
        )

        # ===================================================================
        # カテゴリ12: カーネル密度推定
        # ===================================================================
        try:
            same_wd_values = ts_df[ts_df['weekday'] == target_weekday]['value'].values
            if len(same_wd_values) >= 5:
                kde = stats.gaussian_kde(same_wd_values)
                x_range = np.linspace(same_wd_values.min(), same_wd_values.max(), 1000)
                predictions['KDE_Mode'] = x_range[np.argmax(kde(x_range))]
        except:
            pass

        # ===================================================================
        # カテゴリ13: 中央値ベース（ロバスト）
        # ===================================================================
        predictions['Median_7'] = ts.tail(7).median()
        predictions['Median_14'] = ts.tail(14).median()
        predictions['Median_30'] = ts.tail(30).median()

        # トリム平均
        predictions['TrimMean_10pct'] = stats.trim_mean(ts.tail(30).values, 0.1)
        predictions['TrimMean_20pct'] = stats.trim_mean(ts.tail(30).values, 0.2)

        # ===================================================================
        # カテゴリ14: 変化点検出ベース
        # ===================================================================
        # 直近のトレンド変化を考慮
        if len(ts) >= 14:
            recent_trend = np.polyfit(range(7), ts.tail(7).values, 1)[0]
            older_trend = np.polyfit(range(7), ts.tail(14).head(7).values, 1)[0]

            if recent_trend > older_trend:
                # 上昇トレンド加速
                predictions['TrendChange_Adj'] = ts.tail(7).mean() * 1.02
            elif recent_trend < older_trend:
                # 下降トレンド加速
                predictions['TrendChange_Adj'] = ts.tail(7).mean() * 0.98
            else:
                predictions['TrendChange_Adj'] = ts.tail(7).mean()

        return predictions, confidence_intervals


# =============================================================================
# ハイブリッド統計アンサンブル
# =============================================================================
class StatisticalEnsemblePredictor:
    """統計的アンサンブル予測"""

    def __init__(self):
        self.stat_predictor = AdvancedStatisticalPredictor()
        self.is_initialized = False
        self.df_data = None

    def initialize(self, df):
        """初期化"""
        print("\n" + "="*70)
        print("【v68.0 統計理論強化版】初期化")
        print("="*70)

        self.df_data = df.copy()

        if 'Date' not in self.df_data.columns:
            for col in ['日付', 'date']:
                if col in self.df_data.columns:
                    self.df_data = self.df_data.rename(columns={col: 'Date'})
                    break

        self.stat_predictor.prepare(self.df_data, '合計')
        self.is_initialized = True
        print("\n✅ 初期化完了！")

    def predict(self, target_date, actuals=None):
        """統計的アンサンブル予測"""
        if not self.is_initialized:
            return None

        actuals = actuals or {}

        print("\n" + "="*70)
        print("【v68.0 統計予測実行】")
        print("="*70)
        print(f"予測日: {target_date}\n")

        predictions, confidence_intervals = self.stat_predictor.predict(target_date)

        if not predictions:
            return None

        # 有効な予測のみ
        valid_preds = {k: v for k, v in predictions.items()
                      if v > 0 and not np.isnan(v) and not np.isinf(v)}

        print(f"【予測モデル数】{len(valid_preds)}個")

        # カテゴリ別集計
        categories = {
            '移動平均': [k for k in valid_preds if 'MA_' in k or 'EMA_' in k or 'WMA' in k or 'TMA' in k],
            '季節性': [k for k in valid_preds if 'Weekday' in k or 'Month' in k or 'WeekOf' in k],
            '前年比較': [k for k in valid_preds if 'PrevYear' in k],
            '指数平滑化': [k for k in valid_preds if 'ExpSmooth' in k or 'Holt' in k],
            'ARIMA系': [k for k in valid_preds if 'ARIMA' in k or 'SARIMA' in k],
            'Prophet': [k for k in valid_preds if 'Prophet' in k],
            'STL': [k for k in valid_preds if 'STL' in k],
            '回帰分析': [k for k in valid_preds if 'Regression' in k or 'Ridge' in k or 'Bayes' in k or 'Huber' in k],
            'ベイズ推定': [k for k in valid_preds if 'Bayes_Posterior' in k or 'Bootstrap' in k],
            'ロバスト統計': [k for k in valid_preds if 'Median' in k or 'Trim' in k or 'KDE' in k],
        }

        print("\n【カテゴリ別予測】")
        for cat, keys in categories.items():
            if keys:
                values = [valid_preds[k] for k in keys]
                print(f"  {cat}: {np.median(values):.0f}件 ({len(keys)}モデル)")

        # 統計的アンサンブル
        all_values = list(valid_preds.values())

        # 複数のアンサンブル手法
        ensemble_mean = np.mean(all_values)
        ensemble_median = np.median(all_values)
        ensemble_trimmed = stats.trim_mean(all_values, 0.1)

        # IQR内の値のみで平均（外れ値除外）
        q1, q3 = np.percentile(all_values, [25, 75])
        iqr_values = [v for v in all_values if q1 <= v <= q3]
        ensemble_iqr = np.mean(iqr_values) if iqr_values else ensemble_median

        # 最終予測（中央値ベース - ロバスト）
        final_pred = ensemble_median

        print(f"\n【アンサンブル比較】")
        print(f"  平均: {ensemble_mean:.0f}件")
        print(f"  中央値: {ensemble_median:.0f}件")
        print(f"  トリム平均: {ensemble_trimmed:.0f}件")
        print(f"  IQR平均: {ensemble_iqr:.0f}件")

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
            final_pred = final_pred * 0.30 + est * 0.70
            print(f"\n  ⏰ 12時実績調整: {actuals[12]} → {est:.0f}件")

        # 信頼区間（パーセンタイル法）
        pred_lower = np.percentile(all_values, 2.5)
        pred_upper = np.percentile(all_values, 97.5)

        # 予測の不確実性
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
        print(f"    使用モデル: {len(valid_preds)}個")
        print("="*70)

        return {
            'final_prediction': int(final_pred),
            'pred_lower': int(pred_lower),
            'pred_upper': int(pred_upper),
            'confidence_score': confidence,
            'cv': cv,
            'model_count': len(valid_preds),
            'all_predictions': valid_preds,
            'ensemble_mean': ensemble_mean,
            'ensemble_median': ensemble_median,
            'confidence_intervals': confidence_intervals
        }


# =============================================================================
# Colab UI
# =============================================================================
def create_statistical_ui():
    """統計版UI"""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
    except:
        return

    system = [None]
    df_data = [None]

    date_picker = widgets.DatePicker(description='予測日:', value=date.today())
    actual_12h = widgets.IntText(value=0, description='12時実績:')
    run_btn = widgets.Button(description='📊 統計予測', button_style='info',
                            layout=widgets.Layout(width='180px', height='50px'))
    init_btn = widgets.Button(description='📈 統計分析', button_style='warning',
                             layout=widgets.Layout(width='150px', height='50px'))
    output = widgets.Output()
    result = widgets.Output()

    def on_init(b):
        init_btn.disabled = True
        with output:
            clear_output()
            import glob
            files = glob.glob('/content/*.xlsx') or glob.glob('/content/*.csv')
            if not files:
                print("⚠️ ファイルをアップロード")
                init_btn.disabled = False
                return
            try:
                df_data[0] = pd.read_excel(files[0]) if files[0].endswith('.xlsx') else pd.read_csv(files[0])
                if 'Date' not in df_data[0].columns:
                    for c in ['日付', 'date']:
                        if c in df_data[0].columns:
                            df_data[0] = df_data[0].rename(columns={c: 'Date'})
                            break
                system[0] = StatisticalEnsemblePredictor()
                system[0].initialize(df_data[0])
            except Exception as e:
                print(f"⚠️ {e}")
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
                        <div style='background:linear-gradient(135deg,#2c3e50,#3498db);color:white;padding:25px;border-radius:15px;text-align:center;margin:20px 0;'>
                            <h2>📊 v68.0【統計理論強化版】</h2>
                            <div style='font-size:56px;font-weight:bold;margin:20px 0;'>{res['final_prediction']:,}件</div>
                            <div>95%信頼区間: {res['pred_lower']:,} ～ {res['pred_upper']:,}件</div>
                            <div>信頼度: {res['confidence_score']}点 | 変動係数: {res['cv']:.1f}%</div>
                            <div style='margin-top:10px;'>📈 統計モデル: {res['model_count']}個</div>
                        </div>
                        """))
        run_btn.disabled = False

    init_btn.on_click(on_init)
    run_btn.on_click(on_run)

    ui = widgets.VBox([
        widgets.HTML('<h2 style="color:#2c3e50;">📊 v68.0 統計理論強化版</h2>'),
        widgets.HTML('''
        <div style="background:#ecf0f1;padding:10px;border-radius:5px;margin:10px 0;font-size:12px;">
            <b>統計手法:</b> 移動平均(7種), 季節性分析(8種), 前年比較(5種), 指数平滑化(6種),
            ARIMA(6種), SARIMA(3種), Prophet, STL分解, 回帰分析(3種), ベイズ推定,
            ブートストラップ, カーネル密度推定, ロバスト統計(5種)
        </div>
        '''),
        date_picker, actual_12h,
        widgets.HBox([run_btn, init_btn]),
        result
    ])
    display(ui, output)


# =============================================================================
# メイン
# =============================================================================
if __name__ == "__main__":
    print("\n【v68.0 搭載統計手法】")
    print("-"*60)
    print("1. 移動平均系: SMA, WMA, EMA, TMA (7種)")
    print("2. 季節性分析: 曜日/月/週/交互作用 (8種)")
    print("3. 前年比較: 同日/同曜日/成長率調整 (5種)")
    print("4. 指数平滑化: Simple/Holt/Damped/HW (6種)")
    print("5. ARIMA: 複数次数 (6種)")
    print("6. SARIMA: 季節性ARIMA (3種)")
    print("7. Prophet: Facebookの時系列予測")
    print("8. STL分解: トレンド+季節性")
    print("9. 回帰分析: Ridge/BayesRidge/Huber (3種)")
    print("10. ベイズ推定: 事後分布予測")
    print("11. ブートストラップ: リサンプリング予測")
    print("12. カーネル密度推定: KDE")
    print("13. ロバスト統計: 中央値/トリム平均 (5種)")
    print("14. 変化点検出: トレンド変化調整")
    print("-"*60)
    print("合計: 40+統計手法")

    try:
        import google.colab
        create_statistical_ui()
    except:
        pass
