# =============================================================================
# v66.9 メタ認知AI予測システム【ハイブリッド版】
# =============================================================================
# 15個の機械学習モデル + 10個の統計モデル = 25モデルアンサンブル
# =============================================================================

print("="*70)
print("【v66.9 メタ認知AI予測システム】ハイブリッド25モデル版")
print("="*70)

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta, datetime
import warnings
import traceback
warnings.filterwarnings("ignore")

# =============================================================================
# オプショナルライブラリのインポート
# =============================================================================
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HISTGB = True
except ImportError:
    HAS_HISTGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM")
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
    print("✅ XGBoost")
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
    print("✅ CatBoost")
except ImportError:
    HAS_CAT = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
    print("✅ statsmodels (ARIMA, Holt-Winters, STL)")
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

print("\n✅ ライブラリ読み込み完了\n")


# =============================================================================
# AI記憶ノートクラス
# =============================================================================
AI_MEMORY_PATH = './ai_memory_note_v66.csv'

class AIMemoryNote:
    """AI記憶ノート - 2年分学習対応版"""

    def __init__(self, memory_path=AI_MEMORY_PATH):
        self.memory_path = memory_path
        self.df_memory = self._load_memory()
        self.bias_info = {}

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                df = pd.read_csv(self.memory_path)
                df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
                print(f"  📚 AI記憶ノート: {len(df)}件の記録を読み込み")
                return df
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    def save_prediction(self, prediction_date, prediction_data):
        try:
            new_record = {
                'prediction_date': pd.Timestamp(prediction_date).isoformat(),
                'prediction_timestamp': datetime.now().isoformat(),
                'final_prediction': prediction_data.get('final_prediction'),
                'pred_lower': prediction_data.get('pred_lower'),
                'pred_upper': prediction_data.get('pred_upper'),
                'confidence_score': prediction_data.get('confidence_score'),
                'ml_prediction': prediction_data.get('ml_prediction'),
                'stat_prediction': prediction_data.get('stat_prediction'),
                'actual_value': prediction_data.get('actual_value'),
                'error': prediction_data.get('error'),
                'error_pct': prediction_data.get('error_pct'),
                'bias_correction_applied': prediction_data.get('bias_correction_applied', False),
                'bias_correction_amount': prediction_data.get('bias_correction_amount', 0),
            }
            df_new = pd.DataFrame([new_record])
            self.df_memory = pd.concat([self.df_memory, df_new], ignore_index=True)
            return True
        except:
            return False

    def save_to_file(self):
        self.df_memory.to_csv(self.memory_path, index=False)

    def detect_bias(self, recent_n=10):
        if self.df_memory is None or len(self.df_memory) == 0:
            self.bias_info = {'detected': False, 'bias': 0, 'std': 0}
            return self.bias_info

        if 'actual_value' not in self.df_memory.columns or 'error' not in self.df_memory.columns:
            self.bias_info = {'detected': False, 'bias': 0, 'std': 0}
            return self.bias_info

        df_completed = self.df_memory[self.df_memory['actual_value'].notna()].copy()

        if len(df_completed) < 3:
            self.bias_info = {'detected': False, 'bias': 0, 'std': 0}
            return self.bias_info

        recent_errors = df_completed.tail(recent_n)['error'].dropna()

        if len(recent_errors) < 3:
            self.bias_info = {'detected': False, 'bias': 0, 'std': 0}
            return self.bias_info

        bias = recent_errors.mean()
        std = recent_errors.std() if len(recent_errors) > 1 else 0
        is_biased = abs(bias) > max(10, 0.5 * std) if std > 0 else abs(bias) > 10

        self.bias_info = {
            'detected': is_biased,
            'bias': bias,
            'std': std,
            'sample_count': len(recent_errors),
            'direction': '過大予測' if bias > 0 else '過小予測'
        }
        return self.bias_info

    def apply_correction(self, prediction, special_day_score=0):
        if not self.bias_info.get('detected', False):
            return prediction, 0

        bias = self.bias_info['bias']
        if special_day_score > 0.5:
            correction_rate = 0.3
        elif special_day_score > 0.2:
            correction_rate = 0.5
        else:
            correction_rate = 0.7

        correction = -bias * correction_rate
        return prediction + correction, correction

    def get_accuracy_stats(self):
        if self.df_memory is None or len(self.df_memory) == 0:
            return None
        if 'actual_value' not in self.df_memory.columns:
            return None
        df_completed = self.df_memory[self.df_memory['actual_value'].notna()].copy()
        if len(df_completed) == 0:
            return None

        return {
            'total': len(df_completed),
            'mae': df_completed['error'].abs().mean() if 'error' in df_completed.columns else 0,
            'mae_pct': df_completed['error_pct'].abs().mean() if 'error_pct' in df_completed.columns else 0,
            'within_5pct': (df_completed['error_pct'].abs() <= 5).sum() if 'error_pct' in df_completed.columns else 0,
            'within_10pct': (df_completed['error_pct'].abs() <= 10).sum() if 'error_pct' in df_completed.columns else 0,
        }


# =============================================================================
# 特徴量エンジニアリング
# =============================================================================
def create_features_enhanced(df, target_col='合計'):
    """強化版特徴量生成"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_monday'] = (df['weekday'] == 0).astype(int)
    df['is_friday'] = (df['weekday'] == 4).astype(int)

    if HAS_JPHOLIDAY:
        df['is_holiday'] = df['Date'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
        df['is_golden_week'] = df['Date'].apply(
            lambda x: 1 if (x.month == 4 and x.day >= 29) or (x.month == 5 and x.day <= 5) else 0
        )
        df['is_obon'] = df['Date'].apply(lambda x: 1 if x.month == 8 and 10 <= x.day <= 16 else 0)
        df['is_year_end'] = df['Date'].apply(
            lambda x: 1 if (x.month == 12 and x.day >= 28) or (x.month == 1 and x.day <= 3) else 0
        )
    else:
        df['is_holiday'] = 0
        df['is_golden_week'] = 0
        df['is_obon'] = 0
        df['is_year_end'] = 0

    if target_col in df.columns:
        for lag in [1, 2, 3, 5, 7, 14, 21, 28, 30, 60, 90]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

        for w in [1, 2, 3, 4, 8, 12]:
            df[f'lag_{w*7}_weekday'] = df[target_col].shift(w * 7)

        df['lag_365'] = df[target_col].shift(365)
        df['lag_364'] = df[target_col].shift(364)

        for window in [3, 5, 7, 14, 21, 30, 60]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).max()

        for span in [3, 7, 14, 21, 30]:
            df[f'ema_{span}'] = df[target_col].shift(1).ewm(span=span, adjust=False).mean()

        df['diff_1'] = df[target_col].diff(1)
        df['diff_7'] = df[target_col].diff(7)
        df['pct_change_1'] = df[target_col].pct_change(1)
        df['pct_change_7'] = df[target_col].pct_change(7)

        df['weekday_mean'] = df.groupby('weekday')[target_col].transform('mean')
        df['month_mean'] = df.groupby('month')[target_col].transform('mean')
        df['weekday_month_mean'] = df.groupby(['weekday', 'month'])[target_col].transform('mean')

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# =============================================================================
# 統計時系列モデルクラス
# =============================================================================
class StatisticalTimeSeriesModels:
    """統計的時系列モデル群（10モデル）"""

    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.training_data = None
        self.model_scores = {}

    def train(self, df, target_col='合計'):
        """統計モデルの訓練/準備"""
        print("\n" + "="*70)
        print("【統計時系列モデル訓練】10モデル")
        print("="*70)

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')
        df = df[df[target_col] > 0]

        self.training_data = df.set_index('Date')[target_col]

        if len(self.training_data) < 30:
            print("⚠️ データ不足（30件未満）")
            return False

        print(f"  訓練データ: {len(self.training_data)}件")

        # 統計モデルは予測時に訓練するため、ここでは準備のみ
        self.is_trained = True
        print("✅ 統計モデル準備完了")
        return True

    def predict(self, target_date, n_steps=1):
        """統計モデルで予測"""
        if not self.is_trained or self.training_data is None:
            return {}, {}

        target_dt = pd.to_datetime(target_date)
        predictions = {}
        errors = {}

        # データを予測日より前のものに限定
        ts_data = self.training_data[self.training_data.index < target_dt]

        if len(ts_data) < 30:
            return {}, {}

        # =================================================================
        # 1. 単純移動平均 (SMA)
        # =================================================================
        try:
            sma_7 = ts_data.tail(7).mean()
            sma_14 = ts_data.tail(14).mean()
            sma_30 = ts_data.tail(30).mean()
            predictions['SMA_7'] = sma_7
            predictions['SMA_14'] = sma_14
            predictions['SMA_30'] = sma_30
        except Exception as e:
            errors['SMA'] = str(e)

        # =================================================================
        # 2. 加重移動平均 (WMA)
        # =================================================================
        try:
            weights = np.arange(1, 8)
            wma = np.average(ts_data.tail(7), weights=weights)
            predictions['WMA_7'] = wma
        except Exception as e:
            errors['WMA'] = str(e)

        # =================================================================
        # 3. 指数移動平均 (EMA)
        # =================================================================
        try:
            ema_7 = ts_data.ewm(span=7, adjust=False).mean().iloc[-1]
            ema_14 = ts_data.ewm(span=14, adjust=False).mean().iloc[-1]
            predictions['EMA_7'] = ema_7
            predictions['EMA_14'] = ema_14
        except Exception as e:
            errors['EMA'] = str(e)

        # =================================================================
        # 4. Holt-Winters指数平滑化
        # =================================================================
        if HAS_STATSMODELS and len(ts_data) >= 14:
            try:
                # 単純指数平滑化
                model_ses = ExponentialSmoothing(
                    ts_data.values,
                    trend=None,
                    seasonal=None
                ).fit(optimized=True)
                predictions['ExpSmooth_Simple'] = model_ses.forecast(n_steps)[0]
            except Exception as e:
                errors['ExpSmooth_Simple'] = str(e)

            try:
                # ダブル指数平滑化（トレンド付き）
                model_des = ExponentialSmoothing(
                    ts_data.values,
                    trend='add',
                    seasonal=None
                ).fit(optimized=True)
                predictions['ExpSmooth_Double'] = model_des.forecast(n_steps)[0]
            except Exception as e:
                errors['ExpSmooth_Double'] = str(e)

            # 季節性付き（週次）
            if len(ts_data) >= 14:
                try:
                    model_hw = ExponentialSmoothing(
                        ts_data.values,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=7
                    ).fit(optimized=True)
                    predictions['HoltWinters'] = model_hw.forecast(n_steps)[0]
                except Exception as e:
                    errors['HoltWinters'] = str(e)

        # =================================================================
        # 5. ARIMA
        # =================================================================
        if HAS_STATSMODELS and len(ts_data) >= 30:
            try:
                model_arima = ARIMA(ts_data.values, order=(1, 1, 1)).fit()
                predictions['ARIMA_111'] = model_arima.forecast(n_steps)[0]
            except Exception as e:
                errors['ARIMA_111'] = str(e)

            try:
                model_arima2 = ARIMA(ts_data.values, order=(2, 1, 2)).fit()
                predictions['ARIMA_212'] = model_arima2.forecast(n_steps)[0]
            except Exception as e:
                errors['ARIMA_212'] = str(e)

        # =================================================================
        # 6. SARIMA（季節性ARIMA）
        # =================================================================
        if HAS_STATSMODELS and len(ts_data) >= 60:
            try:
                model_sarima = SARIMAX(
                    ts_data.values,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7)
                ).fit(disp=False)
                predictions['SARIMA'] = model_sarima.forecast(n_steps)[0]
            except Exception as e:
                errors['SARIMA'] = str(e)

        # =================================================================
        # 7. Prophet
        # =================================================================
        if HAS_PROPHET and len(ts_data) >= 30:
            try:
                prophet_df = pd.DataFrame({
                    'ds': ts_data.index,
                    'y': ts_data.values
                })
                model_prophet = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                model_prophet.fit(prophet_df)

                future = pd.DataFrame({'ds': [target_dt]})
                forecast = model_prophet.predict(future)
                predictions['Prophet'] = forecast['yhat'].values[0]
            except Exception as e:
                errors['Prophet'] = str(e)

        # =================================================================
        # 8. STL分解 + 予測
        # =================================================================
        if HAS_STATSMODELS and len(ts_data) >= 14:
            try:
                stl = STL(ts_data, period=7, robust=True).fit()
                # トレンド + 季節性の最後の値を使用
                trend_last = stl.trend.iloc[-1]
                seasonal_idx = len(ts_data) % 7
                seasonal_component = stl.seasonal.iloc[-(7-seasonal_idx)] if seasonal_idx > 0 else stl.seasonal.iloc[-7]
                predictions['STL'] = trend_last + seasonal_component
            except Exception as e:
                errors['STL'] = str(e)

        # =================================================================
        # 9. 同曜日平均（曜日ベース予測）
        # =================================================================
        try:
            ts_df = ts_data.reset_index()
            ts_df.columns = ['Date', 'value']
            ts_df['weekday'] = ts_df['Date'].dt.dayofweek
            target_weekday = target_dt.dayofweek

            same_weekday = ts_df[ts_df['weekday'] == target_weekday]['value']
            if len(same_weekday) > 0:
                predictions['WeekdayMean'] = same_weekday.mean()
                # 直近4週の同曜日
                predictions['WeekdayMean_Recent'] = same_weekday.tail(4).mean()
        except Exception as e:
            errors['WeekdayMean'] = str(e)

        # =================================================================
        # 10. 前年同日/同週
        # =================================================================
        try:
            # 前年同日
            prev_year_date = target_dt - timedelta(days=365)
            if prev_year_date in ts_data.index:
                predictions['PrevYear_SameDay'] = ts_data[prev_year_date]

            # 前年同週同曜日
            prev_year_same_weekday = target_dt - timedelta(days=364)
            if prev_year_same_weekday in ts_data.index:
                predictions['PrevYear_SameWeekday'] = ts_data[prev_year_same_weekday]
            else:
                # 近い日付を探す
                for delta in range(-3, 4):
                    check_date = prev_year_same_weekday + timedelta(days=delta)
                    if check_date in ts_data.index:
                        predictions['PrevYear_SameWeekday'] = ts_data[check_date]
                        break
        except Exception as e:
            errors['PrevYear'] = str(e)

        return predictions, errors


# =============================================================================
# 機械学習モデルクラス（15モデル）
# =============================================================================
class EnhancedMLPredictor:
    """15個のMLモデルによる高精度予測"""

    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_cols = None
        self.is_trained = False
        self.model_scores = {}
        self.model_weights = {}

    def _initialize_models(self):
        """15個のモデルを初期化"""
        models = {}

        # 1. LightGBM
        if HAS_LGB:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=8,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbose=-1
            )

        # 2. XGBoost
        if HAS_XGB:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
            )

        # 3. CatBoost
        if HAS_CAT:
            models['CatBoost'] = CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=6,
                random_state=42, verbose=0
            )

        # 4. RandomForest
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )

        # 5. Ridge
        models['Ridge'] = Ridge(alpha=1.0)

        # 6. GradientBoosting
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42
        )

        # 7. ExtraTrees
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )

        # 8. HistGradientBoosting
        if HAS_HISTGB:
            models['HistGradientBoosting'] = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=10,
                min_samples_leaf=20, random_state=42
            )

        # 9. SVR
        models['SVR'] = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)

        # 10. Lasso
        models['Lasso'] = Lasso(alpha=0.1, max_iter=5000)

        # 11. ElasticNet
        models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)

        # 12. BayesianRidge
        models['BayesianRidge'] = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
        )

        # 13. MLP
        models['MLP'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            alpha=0.001, learning_rate='adaptive', max_iter=500,
            early_stopping=True, validation_fraction=0.1, random_state=42
        )

        # 14. KNN
        models['KNN'] = KNeighborsRegressor(
            n_neighbors=10, weights='distance', algorithm='auto', n_jobs=-1
        )

        # 15. AdaBoost
        models['AdaBoost'] = AdaBoostRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        )

        return models

    def train(self, df, target_col='合計'):
        """モデル訓練"""
        print("\n" + "="*70)
        print("【機械学習モデル訓練】15モデル")
        print("="*70)

        df_feat = create_features_enhanced(df, target_col)
        df_train = df_feat.dropna(subset=[target_col])

        exclude_cols = ['Date', target_col, 'year']
        self.feature_cols = [c for c in df_train.columns
                            if c not in exclude_cols
                            and df_train[c].dtype in ['int64', 'float64', 'int32', 'float32']]

        X = df_train[self.feature_cols].fillna(0)
        y = df_train[target_col]

        if len(X) < 50:
            print("⚠️ データ不足（50件未満）")
            return False

        print(f"  訓練データ: {len(X)}件, 特徴量: {len(self.feature_cols)}個")

        X_scaled = self.scaler.fit_transform(X)
        self.models = self._initialize_models()

        print(f"\n  📊 {len(self.models)}個のモデルを訓練中...")

        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            try:
                print(f"    🔄 {name:20s}...", end=" ")
                cv_scores = cross_val_score(
                    model, X_scaled, y, cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1 if 'n_jobs' not in model.get_params() else 1
                )
                cv_mae = -cv_scores.mean()
                model.fit(X_scaled, y)
                self.model_scores[name] = {'mae': cv_mae}
                print(f"✅ MAE: {cv_mae:,.1f}")
            except Exception as e:
                print(f"❌")

        total_inv_mae = sum(1 / (s['mae'] + 1) for s in self.model_scores.values())
        self.model_weights = {
            name: (1 / (score['mae'] + 1)) / total_inv_mae
            for name, score in self.model_scores.items()
        }

        self.is_trained = True
        print(f"\n✅ {len(self.models)}個のMLモデル訓練完了！")
        return True

    def predict(self, df, target_date, target_col='合計'):
        """予測実行"""
        if not self.is_trained:
            return None, {}

        df_feat = create_features_enhanced(df, target_col)
        target_dt = pd.to_datetime(target_date)
        target_row = df_feat[df_feat['Date'] == target_dt]

        if len(target_row) == 0:
            last_row = df_feat.iloc[-1:].copy()
            last_row['Date'] = target_dt
            last_row['year'] = target_dt.year
            last_row['month'] = target_dt.month
            last_row['day'] = target_dt.day
            last_row['weekday'] = target_dt.dayofweek
            target_row = last_row

        X_pred = target_row[self.feature_cols].fillna(0)
        for col in self.feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[self.feature_cols]
        X_pred_scaled = self.scaler.transform(X_pred)

        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_pred_scaled)[0]
                predictions[name] = max(0, pred)
            except:
                pass

        if len(predictions) == 0:
            return None, {}

        ensemble = sum(
            predictions[k] * self.model_weights.get(k, 1/len(predictions))
            for k in predictions
        )

        return ensemble, {'predictions': predictions, 'model_count': len(predictions)}


# =============================================================================
# ハイブリッドアンサンブルクラス
# =============================================================================
class HybridEnsemblePredictor:
    """ML + 統計モデルのハイブリッドアンサンブル"""

    def __init__(self):
        self.ml_predictor = EnhancedMLPredictor()
        self.stat_predictor = StatisticalTimeSeriesModels()
        self.memory_note = AIMemoryNote()
        self.is_initialized = False
        self.df_data = None

    def initialize(self, df_orig):
        """システム初期化"""
        print("\n" + "="*70)
        print("【v66.9 ハイブリッドシステム初期化】")
        print("  ML 15モデル + 統計 10モデル = 25モデル")
        print("="*70)

        self.df_data = df_orig.copy()

        # ML モデル訓練
        self.ml_predictor.train(self.df_data, '合計')

        # 統計モデル準備
        self.stat_predictor.train(self.df_data, '合計')

        self.is_initialized = True
        print("\n✅ ハイブリッドシステム初期化完了！")

    def predict(self, target_date, actuals=None):
        """ハイブリッド予測実行"""
        if not self.is_initialized:
            print("⚠️ システムが初期化されていません")
            return None

        if actuals is None:
            actuals = {}

        print("\n" + "="*70)
        print("【v66.9 ハイブリッド予測】25モデルアンサンブル")
        print("="*70)
        print(f"予測日: {target_date}\n")

        df = self.df_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['合計'] = pd.to_numeric(df['合計'], errors='coerce').fillna(0)
        df = df[df['合計'] > 0]

        all_predictions = {}

        # =================================================================
        # ML予測（15モデル）
        # =================================================================
        print("【機械学習予測】15モデル")
        ml_pred, ml_info = self.ml_predictor.predict(df, target_date, '合計')
        if ml_pred:
            print(f"  MLアンサンブル: {ml_pred:,.0f}件 ({ml_info['model_count']}モデル)")
            all_predictions['ML_Ensemble'] = ml_pred

            # 個別MLモデルの予測も追加
            if 'predictions' in ml_info:
                for name, pred in ml_info['predictions'].items():
                    all_predictions[f'ML_{name}'] = pred

        # =================================================================
        # 統計予測（10モデル）
        # =================================================================
        print("\n【統計時系列予測】10モデル")
        stat_preds, stat_errors = self.stat_predictor.predict(target_date)

        stat_count = 0
        for name, pred in stat_preds.items():
            if pred > 0 and not np.isnan(pred):
                all_predictions[f'Stat_{name}'] = pred
                stat_count += 1
                if stat_count <= 5:
                    print(f"  {name}: {pred:,.0f}件")

        if stat_count > 5:
            print(f"  ... 他 {stat_count - 5} モデル")

        print(f"  統計モデル合計: {stat_count}モデル")

        # =================================================================
        # ハイブリッドアンサンブル
        # =================================================================
        print("\n【ハイブリッドアンサンブル】")

        valid_predictions = {k: v for k, v in all_predictions.items()
                           if v > 0 and not np.isnan(v)}

        if len(valid_predictions) == 0:
            print("⚠️ 有効な予測がありません")
            return None

        # 重み付け戦略
        # MLモデル: 60%, 統計モデル: 40%
        ml_preds = {k: v for k, v in valid_predictions.items() if k.startswith('ML_')}
        stat_preds_final = {k: v for k, v in valid_predictions.items() if k.startswith('Stat_')}

        if ml_preds and stat_preds_final:
            ml_ensemble = np.median(list(ml_preds.values()))
            stat_ensemble = np.median(list(stat_preds_final.values()))
            hybrid_ensemble = ml_ensemble * 0.60 + stat_ensemble * 0.40
        elif ml_preds:
            hybrid_ensemble = np.median(list(ml_preds.values()))
        elif stat_preds_final:
            hybrid_ensemble = np.median(list(stat_preds_final.values()))
        else:
            hybrid_ensemble = np.median(list(valid_predictions.values()))

        print(f"  有効モデル数: {len(valid_predictions)}")
        print(f"  ハイブリッド予測: {hybrid_ensemble:,.0f}件")

        # =================================================================
        # メタ認知補正
        # =================================================================
        self.memory_note.detect_bias(recent_n=30)
        corrected_pred, correction = self.memory_note.apply_correction(hybrid_ensemble)

        if correction != 0:
            print(f"\n【メタ認知補正】")
            print(f"  補正量: {correction:+,.0f}件")
            print(f"  補正後: {corrected_pred:,.0f}件")

        # =================================================================
        # 12時実績調整
        # =================================================================
        final_pred = corrected_pred
        if actuals.get(12, 0) > 0:
            ratio = 0.45
            if '件数(～12:00)' in df.columns:
                df_12h = df[(df['件数(～12:00)'] > 0) & (df['合計'] > 0)].copy()
                if len(df_12h) > 0:
                    df_12h['件数(～12:00)'] = pd.to_numeric(df_12h['件数(～12:00)'], errors='coerce')
                    ratio = (df_12h['件数(～12:00)'] / df_12h['合計']).mean()

            if ratio > 0:
                est_12h = actuals[12] / ratio
                final_pred = final_pred * 0.35 + est_12h * 0.65
                print(f"\n  ⏰ 12時実績調整: {actuals[12]}件 → 推定{est_12h:,.0f}件")

        # =================================================================
        # 信頼区間と信頼度
        # =================================================================
        all_values = list(valid_predictions.values())
        pred_std = np.std(all_values)
        pred_lower = max(0, final_pred - 1.96 * pred_std)
        pred_upper = final_pred + 1.96 * pred_std

        # 予測のばらつきに基づく信頼度
        cv = pred_std / (np.mean(all_values) + 1e-8)
        if cv < 0.05:
            confidence = 90
        elif cv < 0.10:
            confidence = 80
        elif cv < 0.15:
            confidence = 70
        else:
            confidence = 60

        print("\n" + "="*70)
        print(f"★★★ 最終予測: {int(final_pred):,}件 ★★★")
        print(f"    信頼区間: {int(pred_lower):,} ～ {int(pred_upper):,}件")
        print(f"    信頼度: {confidence}点")
        print(f"    使用モデル数: {len(valid_predictions)}個")
        print("="*70)

        return {
            'final_prediction': int(final_pred),
            'pred_lower': int(pred_lower),
            'pred_upper': int(pred_upper),
            'confidence_score': confidence,
            'ml_prediction': ml_pred,
            'stat_predictions': stat_preds,
            'all_predictions': valid_predictions,
            'model_count': len(valid_predictions),
            'bias_correction': correction
        }


# =============================================================================
# Google Colab用UIクラス
# =============================================================================
def create_colab_ui():
    """Google Colab用のUI作成"""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
    except ImportError:
        print("⚠️ ipywidgetsが必要です")
        return

    # グローバル変数
    global_system = [None]  # リストで参照を保持
    global_df = [None]

    date_picker = widgets.DatePicker(description='予測日:', value=date.today())
    actual_12h = widgets.IntText(value=0, description='12時実績:', layout=widgets.Layout(width='150px'))

    run_btn = widgets.Button(
        description='🚀 予測実行',
        button_style='success',
        layout=widgets.Layout(width='200px', height='50px')
    )
    init_btn = widgets.Button(
        description='🎓 25モデル学習',
        button_style='warning',
        layout=widgets.Layout(width='180px', height='50px')
    )
    output = widgets.Output()
    result_output = widgets.Output()

    def on_init(b):
        init_btn.disabled = True
        init_btn.description = '学習中...'

        with output:
            clear_output()
            import glob
            files = glob.glob('/content/*.xlsx')
            if not files:
                files = glob.glob('/content/*.csv')

            if not files:
                print("⚠️ データファイルをアップロードしてください")
                init_btn.disabled = False
                init_btn.description = '🎓 25モデル学習'
                return

            try:
                if files[0].endswith('.xlsx'):
                    global_df[0] = pd.read_excel(files[0], engine='openpyxl')
                else:
                    global_df[0] = pd.read_csv(files[0])

                if 'Date' not in global_df[0].columns:
                    for col in ['日付', 'date']:
                        if col in global_df[0].columns:
                            global_df[0] = global_df[0].rename(columns={col: 'Date'})
                            break

                global_system[0] = HybridEnsemblePredictor()
                global_system[0].initialize(global_df[0])

            except Exception as e:
                print(f"⚠️ エラー: {e}")
                traceback.print_exc()

        init_btn.disabled = False
        init_btn.description = '🎓 25モデル学習'

    def on_run(b):
        run_btn.disabled = True
        run_btn.description = '処理中...'

        with output:
            clear_output()

            if global_df[0] is None:
                import glob
                files = glob.glob('/content/*.xlsx')
                if not files:
                    files = glob.glob('/content/*.csv')

                if not files:
                    print("⚠️ データファイルをアップロードしてください")
                    run_btn.disabled = False
                    run_btn.description = '🚀 予測実行'
                    return

                try:
                    if files[0].endswith('.xlsx'):
                        global_df[0] = pd.read_excel(files[0], engine='openpyxl')
                    else:
                        global_df[0] = pd.read_csv(files[0])

                    if 'Date' not in global_df[0].columns:
                        for col in ['日付', 'date']:
                            if col in global_df[0].columns:
                                global_df[0] = global_df[0].rename(columns={col: 'Date'})
                                break
                except Exception as e:
                    print(f"⚠️ エラー: {e}")
                    run_btn.disabled = False
                    run_btn.description = '🚀 予測実行'
                    return

            if global_system[0] is None:
                global_system[0] = HybridEnsemblePredictor()
                global_system[0].initialize(global_df[0])

            actuals = {12: actual_12h.value}
            results = global_system[0].predict(date_picker.value, actuals)

            if results:
                with result_output:
                    clear_output()

                    correction_info = ""
                    if results.get('bias_correction', 0) != 0:
                        correction_info = f"<div style='margin-top:10px;'>🧠 AI補正: {results['bias_correction']:+,.0f}件</div>"

                    display(HTML(f"""
                    <div style='background:linear-gradient(135deg,#11998e,#38ef7d);color:white;padding:25px;border-radius:15px;text-align:center;margin-top:20px;'>
                        <h2 style='margin:0;'>🧠 v66.9【ハイブリッド25モデル版】</h2>
                        <div style='font-size:56px;font-weight:bold;margin:20px 0;'>{results['final_prediction']:,}件</div>
                        <div style='font-size:18px;'>信頼区間: {results['pred_lower']:,} ～ {results['pred_upper']:,}件</div>
                        <div style='font-size:16px;margin-top:10px;'>信頼度: {results['confidence_score']}点</div>
                        <div style='font-size:14px;margin-top:5px;'>📊 使用モデル: {results['model_count']}個 (ML 15 + 統計 10)</div>
                        {correction_info}
                    </div>
                    """))

        run_btn.disabled = False
        run_btn.description = '🚀 予測実行'

    init_btn.on_click(on_init)
    run_btn.on_click(on_run)

    ui = widgets.VBox([
        widgets.HTML('<h2 style="color:#11998e;">🧠 v66.9 ハイブリッドAI予測【25モデル版】</h2>'),
        widgets.HTML('<p>ML 15モデル + 統計 10モデルのハイブリッドアンサンブル</p>'),
        widgets.HTML('''
        <div style="background:#f0f0f0;padding:10px;border-radius:5px;margin:10px 0;">
            <b>MLモデル:</b> LightGBM, XGBoost, CatBoost, RandomForest, Ridge,
            GradientBoosting, ExtraTrees, HistGradientBoosting, SVR, Lasso,
            ElasticNet, BayesianRidge, MLP, KNN, AdaBoost<br>
            <b>統計モデル:</b> SMA, WMA, EMA, Holt-Winters, ARIMA, SARIMA,
            Prophet, STL, 曜日平均, 前年同日
        </div>
        '''),
        date_picker,
        widgets.HBox([actual_12h]),
        widgets.HBox([run_btn, init_btn]),
        result_output
    ])

    display(ui, output)


# =============================================================================
# メイン
# =============================================================================
def demo_usage():
    """使用例のデモ"""
    print("\n" + "="*70)
    print("【v66.9 使用方法】")
    print("="*70)
    print("""
# 1. データの読み込み
import pandas as pd
df = pd.read_excel('実績表.xlsx')

# 2. Date列の確認・変換
if '日付' in df.columns:
    df = df.rename(columns={'日付': 'Date'})

# 3. システム初期化
system = HybridEnsemblePredictor()
system.initialize(df)

# 4. 予測実行
from datetime import date
result = system.predict(date.today())

# 5. 結果確認
print(f"予測: {result['final_prediction']:,}件")
print(f"使用モデル: {result['model_count']}個")
""")
    print("="*70)


if __name__ == "__main__":
    demo_usage()

    print("\n【搭載モデル一覧】25個")
    print("-"*50)
    print("【ML 15モデル】")
    ml_models = [
        "1. LightGBM", "2. XGBoost", "3. CatBoost",
        "4. RandomForest", "5. Ridge", "6. GradientBoosting",
        "7. ExtraTrees", "8. HistGradientBoosting", "9. SVR",
        "10. Lasso", "11. ElasticNet", "12. BayesianRidge",
        "13. MLP", "14. KNN", "15. AdaBoost"
    ]
    for m in ml_models:
        print(f"  {m}")

    print("\n【統計 10モデル】")
    stat_models = [
        "1. SMA (単純移動平均)", "2. WMA (加重移動平均)",
        "3. EMA (指数移動平均)", "4. 単純指数平滑化",
        "5. ダブル指数平滑化", "6. Holt-Winters",
        "7. ARIMA", "8. SARIMA", "9. Prophet",
        "10. STL分解"
    ]
    for m in stat_models:
        print(f"  {m}")
    print("-"*50)

    # Google Colab環境の場合はUIを表示
    try:
        import google.colab
        print("\n🎨 Google Colab UI を起動中...")
        create_colab_ui()
    except ImportError:
        pass
