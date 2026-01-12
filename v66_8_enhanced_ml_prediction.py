# =============================================================================
# v66.8 メタ認知AI予測システム【10個追加ML強化版】
# =============================================================================
# 15個の機械学習モデルによる高精度アンサンブル予測
# =============================================================================

print("="*70)
print("【v66.8 メタ認知AI予測システム】10個追加ML強化版")
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
    BaggingRegressor,
    StackingRegressor,
    VotingRegressor
)
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    SGDRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta, datetime
import warnings
import traceback
warnings.filterwarnings("ignore")

# オプショナルライブラリ
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
    import jpholiday
    HAS_JPHOLIDAY = True
    print("✅ jpholiday")
except ImportError:
    HAS_JPHOLIDAY = False

print("\n✅ ライブラリ読み込み完了")
print(f"   HistGradientBoosting: {'✅' if HAS_HISTGB else '❌'}")
print()


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
        """予測を保存"""
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
        """ファイルに保存"""
        self.df_memory.to_csv(self.memory_path, index=False)

    def detect_bias(self, recent_n=10):
        """バイアス検出"""
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
        """自己補正を適用"""
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
        corrected_prediction = prediction + correction

        return corrected_prediction, correction

    def get_accuracy_stats(self):
        """精度統計を取得"""
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
            'median_error': df_completed['error_pct'].median() if 'error_pct' in df_completed.columns else 0,
            'within_5pct': (df_completed['error_pct'].abs() <= 5).sum() if 'error_pct' in df_completed.columns else 0,
            'within_10pct': (df_completed['error_pct'].abs() <= 10).sum() if 'error_pct' in df_completed.columns else 0,
        }


# =============================================================================
# 特徴量エンジニアリング（強化版）
# =============================================================================
def create_features_enhanced(df, target_col='合計'):
    """強化版特徴量生成"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # 基本時間特徴量
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear

    # 周期的エンコーディング
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # バイナリ特徴量
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_monday'] = (df['weekday'] == 0).astype(int)
    df['is_friday'] = (df['weekday'] == 4).astype(int)
    df['is_quarter_start'] = ((df['month'] % 3 == 1) & (df['day'] <= 5)).astype(int)
    df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df['day'] >= 25)).astype(int)

    # 祝日特徴量
    if HAS_JPHOLIDAY:
        df['is_holiday'] = df['Date'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
        df['is_golden_week'] = df['Date'].apply(
            lambda x: 1 if (x.month == 4 and x.day >= 29) or (x.month == 5 and x.day <= 5) else 0
        )
        df['is_obon'] = df['Date'].apply(
            lambda x: 1 if x.month == 8 and 10 <= x.day <= 16 else 0
        )
        df['is_year_end'] = df['Date'].apply(
            lambda x: 1 if (x.month == 12 and x.day >= 28) or (x.month == 1 and x.day <= 3) else 0
        )
    else:
        df['is_holiday'] = 0
        df['is_golden_week'] = 0
        df['is_obon'] = 0
        df['is_year_end'] = 0

    # ラグ特徴量
    if target_col in df.columns:
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 60, 90]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

        # 同曜日ラグ
        for w in [1, 2, 3, 4, 8, 12]:
            df[f'lag_{w*7}_weekday'] = df[target_col].shift(w * 7)

        # 年間ラグ
        df['lag_365'] = df[target_col].shift(365)
        df['lag_364'] = df[target_col].shift(364)  # 同曜日前年

        # ローリング統計量
        for window in [3, 5, 7, 14, 21, 30, 60]:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).max()
            df[f'rolling_median_{window}'] = df[target_col].shift(1).rolling(window, min_periods=1).median()

        # 指数移動平均
        for span in [3, 7, 14, 21, 30]:
            df[f'ema_{span}'] = df[target_col].shift(1).ewm(span=span, adjust=False).mean()

        # トレンド特徴量
        df['diff_1'] = df[target_col].diff(1)
        df['diff_7'] = df[target_col].diff(7)
        df['pct_change_1'] = df[target_col].pct_change(1)
        df['pct_change_7'] = df[target_col].pct_change(7)

        # グループ統計量
        df['weekday_mean'] = df.groupby('weekday')[target_col].transform('mean')
        df['weekday_std'] = df.groupby('weekday')[target_col].transform('std')
        df['month_mean'] = df.groupby('month')[target_col].transform('mean')
        df['month_std'] = df.groupby('month')[target_col].transform('std')
        df['week_of_month_mean'] = df.groupby('week_of_month')[target_col].transform('mean')

        # 相互作用特徴量
        df['weekday_month_mean'] = df.groupby(['weekday', 'month'])[target_col].transform('mean')

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# =============================================================================
# 強化版機械学習予測クラス（15モデル）
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

        # ===== 既存5モデル =====
        # 1. LightGBM
        if HAS_LGB:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )

        # 2. XGBoost
        if HAS_XGB:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )

        # 3. CatBoost
        if HAS_CAT:
            models['CatBoost'] = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=0
            )

        # 4. RandomForest
        models['RandomForest'] = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 5. Ridge
        models['Ridge'] = Ridge(alpha=1.0)

        # ===== 追加10モデル =====
        # 6. GradientBoosting
        models['GradientBoosting'] = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )

        # 7. ExtraTrees
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 8. HistGradientBoosting
        if HAS_HISTGB:
            models['HistGradientBoosting'] = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_depth=10,
                min_samples_leaf=20,
                random_state=42
            )

        # 9. SVR (Support Vector Regression)
        models['SVR'] = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        )

        # 10. Lasso
        models['Lasso'] = Lasso(alpha=0.1, max_iter=5000)

        # 11. ElasticNet
        models['ElasticNet'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=5000
        )

        # 12. BayesianRidge
        models['BayesianRidge'] = BayesianRidge(
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )

        # 13. MLPRegressor (Neural Network)
        models['MLP'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

        # 14. KNeighborsRegressor
        models['KNN'] = KNeighborsRegressor(
            n_neighbors=10,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            n_jobs=-1
        )

        # 15. AdaBoost
        models['AdaBoost'] = AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

        return models

    def train(self, df, target_col='合計'):
        """モデル訓練"""
        print("\n" + "="*70)
        print("【強化版機械学習訓練】15モデル")
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

        print(f"  訓練データ: {len(X)}件")
        print(f"  特徴量: {len(self.feature_cols)}個")

        X_scaled = self.scaler.fit_transform(X)

        # モデル初期化
        self.models = self._initialize_models()

        print(f"\n  📊 {len(self.models)}個のモデルを訓練中...")

        # 時系列クロスバリデーション
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            try:
                print(f"    🔄 {name:20s}...", end=" ")

                # クロスバリデーションスコア
                cv_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1 if 'n_jobs' not in model.get_params() else 1
                )
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()

                # モデル訓練
                model.fit(X_scaled, y)

                self.model_scores[name] = {
                    'mae': cv_mae,
                    'std': cv_std
                }

                print(f"✅ MAE: {cv_mae:,.1f} (±{cv_std:,.1f})")

            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")

        # モデル重みを計算（MAEの逆数で重み付け）
        total_inv_mae = sum(1 / (s['mae'] + 1) for s in self.model_scores.values())
        self.model_weights = {
            name: (1 / (score['mae'] + 1)) / total_inv_mae
            for name, score in self.model_scores.items()
        }

        # 重みでソートして表示
        print("\n  📊 モデル重み（精度順）:")
        sorted_weights = sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True)
        for name, weight in sorted_weights[:5]:
            print(f"    {name:20s}: {weight*100:5.2f}%")

        self.is_trained = True
        print(f"\n✅ {len(self.models)}個のモデル訓練完了！")

        return True

    def predict(self, df, target_date, target_col='合計'):
        """予測実行"""
        if not self.is_trained:
            return None, {}

        df_feat = create_features_enhanced(df, target_col)
        target_dt = pd.to_datetime(target_date)
        target_row = df_feat[df_feat['Date'] == target_dt]

        if len(target_row) == 0:
            # 予測日のデータがない場合、最後の行を使用して特徴量を生成
            last_row = df_feat.iloc[-1:].copy()
            last_row['Date'] = target_dt

            # 日付関連特徴量を更新
            last_row['year'] = target_dt.year
            last_row['month'] = target_dt.month
            last_row['day'] = target_dt.day
            last_row['weekday'] = target_dt.dayofweek
            last_row['week_of_month'] = (target_dt.day - 1) // 7 + 1

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
            except Exception as e:
                pass

        if len(predictions) == 0:
            return None, {}

        # 加重平均アンサンブル
        ensemble = sum(
            predictions[k] * self.model_weights.get(k, 1/len(predictions))
            for k in predictions
        )

        # 予測の不確実性（標準偏差）
        pred_std = np.std(list(predictions.values()))

        return ensemble, {
            'predictions': predictions,
            'std': pred_std,
            'model_count': len(predictions)
        }

    def get_model_performance_summary(self):
        """モデルパフォーマンスサマリー"""
        if not self.model_scores:
            return None

        summary = []
        for name, score in sorted(self.model_scores.items(), key=lambda x: x[1]['mae']):
            summary.append({
                'model': name,
                'mae': score['mae'],
                'std': score['std'],
                'weight': self.model_weights.get(name, 0) * 100
            })

        return pd.DataFrame(summary)


# =============================================================================
# 統計予測関数群
# =============================================================================
def predict_by_similar_days(df_orig, prediction_date, recent_years=3, top_n=20):
    """多次元類似日検索（強化版）"""
    try:
        if df_orig is None or len(df_orig) == 0 or '合計' not in df_orig.columns:
            return 1000, {}

        pred_date = pd.to_datetime(prediction_date)
        df = df_orig.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df[df['合計'] > 0].copy()

        if len(df) == 0:
            return 1000, {}

        # 特徴量計算
        df['weekday'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 25).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        target_features = {
            'weekday': pred_date.dayofweek,
            'month': pred_date.month,
            'week_of_month': (pred_date.day - 1) // 7 + 1,
            'is_month_start': 1 if pred_date.day <= 3 else 0,
            'is_month_end': 1 if pred_date.day >= 25 else 0,
            'month_sin': np.sin(2 * np.pi * pred_date.month / 12),
            'month_cos': np.cos(2 * np.pi * pred_date.month / 12),
            'weekday_sin': np.sin(2 * np.pi * pred_date.dayofweek / 7),
            'weekday_cos': np.cos(2 * np.pi * pred_date.dayofweek / 7),
        }

        feature_weights = {
            'weekday': 5.0, 'month': 3.0, 'week_of_month': 2.0,
            'is_month_start': 4.0, 'is_month_end': 2.5,
            'month_sin': 2.0, 'month_cos': 2.0,
            'weekday_sin': 3.0, 'weekday_cos': 3.0
        }

        feature_cols = list(feature_weights.keys())
        weights_array = np.array([feature_weights[col] for col in feature_cols])

        cutoff_date = pred_date - pd.DateOffset(years=recent_years)
        df_recent = df[(df['Date'] >= cutoff_date) & (df['Date'] < pred_date)].copy()
        if len(df_recent) < 10:
            df_recent = df[df['Date'] < pred_date].copy()
        if len(df_recent) == 0:
            return df['合計'].mean(), {}

        X_history = df_recent[feature_cols].values
        X_target = np.array([[target_features[col] for col in feature_cols]])

        scaler = StandardScaler()
        X_history_scaled = scaler.fit_transform(X_history)
        X_target_scaled = scaler.transform(X_target)

        weighted_diff = (X_history_scaled - X_target_scaled) * weights_array
        distances = np.sqrt((weighted_diff ** 2).sum(axis=1))
        df_recent['distance'] = distances

        actual_top_n = min(top_n, len(df_recent))
        similar_days = df_recent.nsmallest(actual_top_n, 'distance').copy()

        inv_distances = 1 / (similar_days['distance'].values + 0.01)
        similarity_weights = inv_distances / inv_distances.sum()
        weighted_prediction = (similar_days['合計'].values * similarity_weights).sum()

        # 年次トレンド補正
        yearly_avg = df_recent.groupby(df_recent['Date'].dt.year)['合計'].mean()
        if len(yearly_avg) >= 2:
            growth_rates = yearly_avg.pct_change().dropna()
            avg_growth = np.clip(growth_rates.mean(), -0.15, 0.25)
        else:
            avg_growth = 0.0

        years_diff = pred_date.year - similar_days['Date'].dt.year.mean()
        trend_adj = np.clip((1 + avg_growth) ** years_diff, 0.7, 1.5)
        prediction = weighted_prediction * trend_adj

        return prediction, {'similar_count': actual_top_n, 'avg_growth_rate': avg_growth}
    except Exception as e:
        return 1000, {}


def predict_by_pattern_matching(df_orig, prediction_date, pattern_days=7, top_n=10):
    """パターンマッチング予測"""
    try:
        if df_orig is None or len(df_orig) == 0:
            return 1000, {}

        pred_date = pd.to_datetime(prediction_date)
        df = df_orig.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        df = df[df['合計'] > 0].copy()

        if len(df) < pattern_days * 3:
            return df['合計'].mean() if len(df) > 0 else 1000, {}

        recent_data = df[df['Date'] < pred_date].tail(pattern_days)
        if len(recent_data) < pattern_days:
            return df['合計'].mean(), {}

        recent_values = recent_data['合計'].values
        recent_std = max(recent_values.std(), 1e-8)
        recent_pattern = (recent_values - recent_values.mean()) / recent_std

        pattern_similarities = []
        for i in range(pattern_days, len(df) - pattern_days - 1):
            hist_values = df.iloc[i-pattern_days:i]['合計'].values
            if len(hist_values) == pattern_days:
                hist_std = max(hist_values.std(), 1e-8)
                hist_pattern = (hist_values - hist_values.mean()) / hist_std

                norm_r, norm_h = np.linalg.norm(recent_pattern), np.linalg.norm(hist_pattern)
                if norm_r > 1e-8 and norm_h > 1e-8:
                    similarity = np.dot(recent_pattern, hist_pattern) / (norm_r * norm_h)
                else:
                    similarity = 0

                pattern_similarities.append({
                    'similarity': similarity,
                    'next_value': df.iloc[i]['合計'],
                    'pattern_mean': hist_values.mean()
                })

        if len(pattern_similarities) == 0:
            return df['合計'].mean(), {}

        pattern_df = pd.DataFrame(pattern_similarities).nlargest(top_n, 'similarity')

        similarities = pattern_df['similarity'].values
        positive_sim = (similarities + 1) / 2
        weights = positive_sim / positive_sim.sum() if positive_sim.sum() > 0 else np.ones(len(positive_sim)) / len(positive_sim)

        recent_mean = recent_values.mean()
        scale_factors = np.clip(recent_mean / (pattern_df['pattern_mean'].values + 1e-8), 0.5, 2.0)
        adjusted_values = pattern_df['next_value'].values * scale_factors
        prediction = (adjusted_values * weights).sum()

        return prediction, {'top_similarity': pattern_df['similarity'].max()}
    except Exception as e:
        return 1000, {}


def predict_by_previous_year(df_orig, prediction_date):
    """前年同日ベース予測"""
    try:
        if df_orig is None or len(df_orig) == 0 or '合計' not in df_orig.columns:
            return 1000, {}

        pred_date = pd.to_datetime(prediction_date)
        df = df_orig.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['year'], df['month'] = df['Date'].dt.year, df['Date'].dt.month

        prev_year_data = df[(df['year'] == pred_date.year - 1) & (df['month'] == pred_date.month) & (df['合計'] > 0)]
        if len(prev_year_data) == 0:
            return df['合計'].mean() if len(df) > 0 else 1000, {}

        prev_year_avg = prev_year_data['合計'].mean()

        current_year_data = df[(df['year'] == pred_date.year) & (df['合計'] > 0)]
        prev_year_all = df[(df['year'] == pred_date.year - 1) & (df['合計'] > 0)]

        growth = np.clip(
            current_year_data['合計'].mean() / prev_year_all['合計'].mean(),
            0.8, 1.3
        ) if len(current_year_data) > 0 and len(prev_year_all) > 0 else 1.0

        prediction = prev_year_avg * growth

        return prediction, {'prev_year_avg': prev_year_avg, 'month_growth': growth}
    except Exception as e:
        return 1000, {}


def predict_by_recent_trend(df_orig, prediction_date):
    """直近トレンド予測"""
    try:
        if df_orig is None or len(df_orig) == 0 or '合計' not in df_orig.columns:
            return 1000, {}

        pred_date = pd.to_datetime(prediction_date)
        df = df_orig.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.dayofweek
        df['year'] = df['Date'].dt.year

        recent_data = df[(df['Date'] >= pred_date - timedelta(days=30)) & (df['Date'] < pred_date) & (df['合計'] > 0)]
        if len(recent_data) < 5:
            recent_data = df[(df['Date'] < pred_date) & (df['合計'] > 0)].tail(20)
        if len(recent_data) == 0:
            return df['合計'].mean() if len(df) > 0 else 1000, {}

        recent_avg = recent_data['合計'].mean()

        historical_data = df[
            (df['month'] == pred_date.month) &
            (df['weekday'] == pred_date.dayofweek) &
            (df['year'] < pred_date.year) &
            (df['合計'] > 0)
        ]
        if len(historical_data) == 0:
            historical_data = df[(df['weekday'] == pred_date.dayofweek) & (df['合計'] > 0)]
        if len(historical_data) == 0:
            return recent_avg, {}

        historical_avg = historical_data['合計'].mean()
        overall_avg = df[df['合計'] > 0]['合計'].mean()

        trend_ratio = np.clip(recent_avg / overall_avg, 0.7, 1.5) if overall_avg > 0 else 1.0
        prediction = historical_avg * trend_ratio

        return prediction, {'recent_avg': recent_avg, 'trend_ratio': trend_ratio}
    except Exception as e:
        return 1000, {}


# =============================================================================
# メインアンサンブル予測関数
# =============================================================================
def ensemble_all_predictions(ml_pred, ml_info, stat_preds, stat_infos):
    """ML予測と統計予測の統合アンサンブル"""

    # 統計予測のアンサンブル
    stat_weights = {
        'similar': 0.35,
        'pattern': 0.20,
        'prev_year': 0.25,
        'trend': 0.20
    }

    stat_ensemble = sum(
        stat_preds[k] * stat_weights[k]
        for k in stat_preds.keys()
    )

    # ML予測と統計予測の統合
    if ml_pred is not None:
        # ML予測の信頼度に基づいて重み付け
        ml_weight = 0.65
        stat_weight = 0.35

        final_ensemble = ml_pred * ml_weight + stat_ensemble * stat_weight
    else:
        final_ensemble = stat_ensemble

    # 予測の不確実性を計算
    all_preds = list(stat_preds.values())
    if ml_pred is not None:
        all_preds.append(ml_pred)

    uncertainty = np.std(all_preds) / (np.mean(all_preds) + 1e-8)

    return final_ensemble, {
        'ml_pred': ml_pred,
        'stat_ensemble': stat_ensemble,
        'uncertainty': uncertainty
    }


# =============================================================================
# バックテスト学習
# =============================================================================
def run_backtest_learning(df_orig, memory_note, years=2):
    """2年分のバックテストでAI記憶ノートを学習"""
    print("\n" + "="*70)
    print("【AI記憶ノート】2年分バックテスト学習")
    print("="*70)

    df = df_orig.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    if '合計' not in df.columns:
        print("⚠️ '合計'列が見つかりません")
        return memory_note

    df['合計'] = pd.to_numeric(df['合計'], errors='coerce').fillna(0)
    df = df[df['合計'] > 0]

    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=365 * years)

    min_history_date = df['Date'].min() + timedelta(days=90)
    if start_date < min_history_date:
        start_date = min_history_date

    print(f"  学習期間: {start_date.date()} ～ {end_date.date()}")

    backtest_dates = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]['Date'].tolist()

    print(f"  バックテスト日数: {len(backtest_dates)}日")
    print("\n  🔄 バックテスト実行中...")

    success_count = 0
    total_error = 0

    for i, target_date in enumerate(backtest_dates):
        try:
            df_history = df[df['Date'] < target_date].copy()

            if len(df_history) < 30:
                continue

            # 簡易予測
            pred_date = pd.to_datetime(target_date)
            same_cond = (df_history['Date'].dt.weekday == pred_date.dayofweek) & \
                        (df_history['Date'].dt.month == pred_date.month)
            same_days = df_history[same_cond]['合計']
            pred1 = same_days.mean() if len(same_days) > 0 else df_history['合計'].mean()

            prev_year = df_history[(df_history['Date'].dt.year == pred_date.year - 1) &
                                  (df_history['Date'].dt.month == pred_date.month)]['合計']
            pred2 = prev_year.mean() if len(prev_year) > 0 else pred1

            recent = df_history[df_history['Date'] >= pred_date - timedelta(days=30)]['合計']
            pred3 = recent.mean() if len(recent) > 0 else pred1

            ensemble = pred1 * 0.4 + pred2 * 0.4 + pred3 * 0.2

            actual = df[df['Date'] == target_date]['合計'].values[0]

            memory_note.detect_bias(recent_n=10)
            corrected_pred, correction = memory_note.apply_correction(ensemble)

            corrected_error = corrected_pred - actual
            corrected_error_pct = (corrected_pred - actual) / actual * 100

            prediction_data = {
                'final_prediction': corrected_pred,
                'pred_lower': corrected_pred * 0.85,
                'pred_upper': corrected_pred * 1.15,
                'confidence_score': 70,
                'ml_prediction': ensemble,
                'stat_prediction': pred1,
                'actual_value': actual,
                'error': corrected_error,
                'error_pct': corrected_error_pct,
                'bias_correction_applied': correction != 0,
                'bias_correction_amount': correction,
            }

            memory_note.save_prediction(target_date, prediction_data)

            total_error += abs(corrected_error_pct)
            success_count += 1

            if (i + 1) % 100 == 0:
                avg_error = total_error / success_count
                print(f"    {i+1}/{len(backtest_dates)} 完了 (平均誤差率: {avg_error:.2f}%)")

        except Exception as e:
            continue

    memory_note.save_to_file()

    print("\n" + "="*70)
    print("【バックテスト完了】")
    print("="*70)

    stats = memory_note.get_accuracy_stats()
    if stats:
        print(f"  ✅ 学習データ: {stats['total']}件")
        print(f"  📊 平均絶対誤差: {stats['mae']:.1f}件")
        print(f"  📊 平均絶対誤差率: {stats['mae_pct']:.2f}%")

    memory_note.detect_bias(recent_n=30)
    if memory_note.bias_info['detected']:
        print(f"\n  ⚠️ バイアス検出: {memory_note.bias_info['direction']}")

    return memory_note


# =============================================================================
# メイン予測関数
# =============================================================================
class PredictionSystem:
    """v66.8 統合予測システム"""

    def __init__(self):
        self.memory_note = AIMemoryNote()
        self.ml_predictor = EnhancedMLPredictor()
        self.is_initialized = False
        self.df_data = None

    def initialize(self, df_orig):
        """システム初期化"""
        print("\n" + "="*70)
        print("【v66.8 システム初期化】15モデル + 2年分学習")
        print("="*70)

        self.df_data = df_orig.copy()

        # バックテスト学習
        self.memory_note = run_backtest_learning(self.df_data, self.memory_note, years=2)

        # MLモデル訓練
        self.ml_predictor.train(self.df_data, '合計')

        self.is_initialized = True
        print("\n✅ 初期化完了！")

    def predict(self, target_date, actuals=None):
        """予測実行"""
        if not self.is_initialized:
            print("⚠️ システムが初期化されていません")
            return None

        if actuals is None:
            actuals = {}

        print("\n" + "="*70)
        print("【v66.8 予測実行】15モデルアンサンブル")
        print("="*70)
        print(f"予測日: {target_date}\n")

        df = self.df_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['合計'] = pd.to_numeric(df['合計'], errors='coerce').fillna(0)
        df = df[df['合計'] > 0]

        # ML予測
        print("【機械学習予測】")
        ml_pred, ml_info = self.ml_predictor.predict(df, target_date, '合計')
        if ml_pred:
            print(f"  ML予測: {ml_pred:,.0f}件 ({ml_info['model_count']}モデル)")

        # 統計予測
        print("\n【統計予測】")
        pred_similar, info_similar = predict_by_similar_days(df, target_date)
        pred_pattern, info_pattern = predict_by_pattern_matching(df, target_date)
        pred_prev, info_prev = predict_by_previous_year(df, target_date)
        pred_trend, info_trend = predict_by_recent_trend(df, target_date)

        print(f"  類似日: {pred_similar:,.0f}件")
        print(f"  パターン: {pred_pattern:,.0f}件")
        print(f"  前年同日: {pred_prev:,.0f}件")
        print(f"  トレンド: {pred_trend:,.0f}件")

        stat_preds = {
            'similar': pred_similar,
            'pattern': pred_pattern,
            'prev_year': pred_prev,
            'trend': pred_trend
        }
        stat_infos = {
            'similar': info_similar,
            'pattern': info_pattern,
            'prev_year': info_prev,
            'trend': info_trend
        }

        # アンサンブル
        ensemble, ensemble_info = ensemble_all_predictions(ml_pred, ml_info, stat_preds, stat_infos)

        # バイアス補正
        print("\n【メタ認知補正】")
        self.memory_note.detect_bias(recent_n=30)
        if self.memory_note.bias_info['detected']:
            print(f"  バイアス検出: {self.memory_note.bias_info['direction']}")

        corrected_pred, correction = self.memory_note.apply_correction(ensemble)

        if correction != 0:
            print(f"  補正前: {ensemble:,.0f}件")
            print(f"  補正量: {correction:+,.0f}件")
            print(f"  補正後: {corrected_pred:,.0f}件")

        # リアルタイム調整
        final_pred = corrected_pred
        if actuals.get(12, 0) > 0:
            # 12時実績がある場合の調整
            ratio = 0.45  # デフォルト比率
            if '件数(～12:00)' in df.columns:
                df_12h = df[(df['件数(～12:00)'] > 0) & (df['合計'] > 0)].copy()
                if len(df_12h) > 0:
                    df_12h['件数(～12:00)'] = pd.to_numeric(df_12h['件数(～12:00)'], errors='coerce')
                    ratio = (df_12h['件数(～12:00)'] / df_12h['合計']).mean()

            if ratio > 0:
                est_12h = actuals[12] / ratio
                final_pred = final_pred * 0.35 + est_12h * 0.65
                print(f"\n  ⏰ 12時実績調整: {actuals[12]}件 → 推定{est_12h:,.0f}件")

        # 信頼区間
        pred_std = final_pred * 0.08
        if ml_info.get('std'):
            pred_std = (pred_std + ml_info['std']) / 2

        pred_lower = max(0, final_pred - 1.96 * pred_std)
        pred_upper = final_pred + 1.96 * pred_std

        # 信頼度
        stats = self.memory_note.get_accuracy_stats()
        if stats and stats['mae_pct'] < 5:
            confidence = 88
        elif stats and stats['mae_pct'] < 10:
            confidence = 78
        else:
            confidence = 68

        print("\n" + "="*70)
        print(f"★★★ 最終予測: {int(final_pred):,}件 ★★★")
        print(f"    信頼区間: {int(pred_lower):,} ～ {int(pred_upper):,}件")
        print(f"    信頼度: {confidence}点")
        if correction != 0:
            print(f"    AI補正: {correction:+,.0f}件")
        print("="*70)

        return {
            'final_prediction': int(final_pred),
            'pred_lower': int(pred_lower),
            'pred_upper': int(pred_upper),
            'confidence_score': confidence,
            'ml_prediction': ml_pred,
            'stat_predictions': stat_preds,
            'bias_correction': correction,
            'model_count': ml_info.get('model_count', 0)
        }


# =============================================================================
# 使用例・テスト関数
# =============================================================================
def demo_usage():
    """使用例のデモ"""
    print("\n" + "="*70)
    print("【v66.8 使用方法】")
    print("="*70)
    print("""
# 1. データの読み込み
import pandas as pd
df = pd.read_excel('実績表.xlsx')

# 2. Date列の確認・変換
if '日付' in df.columns:
    df = df.rename(columns={'日付': 'Date'})

# 3. システム初期化
system = PredictionSystem()
system.initialize(df)

# 4. 予測実行
from datetime import date
result = system.predict(date.today())

# 5. 結果確認
print(f"予測: {result['final_prediction']:,}件")
print(f"信頼区間: {result['pred_lower']:,} ～ {result['pred_upper']:,}件")

# 6. 12時実績がある場合
result = system.predict(date.today(), actuals={12: 450})
""")
    print("="*70)


if __name__ == "__main__":
    demo_usage()

    # モデル一覧表示
    print("\n【搭載モデル一覧】15個")
    print("-"*50)
    models = [
        "1. LightGBM", "2. XGBoost", "3. CatBoost",
        "4. RandomForest", "5. Ridge",
        "6. GradientBoosting", "7. ExtraTrees",
        "8. HistGradientBoosting", "9. SVR",
        "10. Lasso", "11. ElasticNet",
        "12. BayesianRidge", "13. MLP (Neural Network)",
        "14. KNeighbors", "15. AdaBoost"
    ]
    for m in models:
        print(f"  {m}")
    print("-"*50)
