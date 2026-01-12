# =============================================================================
# v67.0 メタ認知AI予測システム【高速・高精度最適化版】
# =============================================================================
# 並列処理 + スマートモデル選択 + 自動チューニング
# 目標: 計算速度最大化 & 精度100点
# =============================================================================

print("="*70)
print("【v67.0 メタ認知AI予測システム】高速・高精度最適化版")
print("="*70)

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import traceback
import time
warnings.filterwarnings("ignore")

# 並列処理
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

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
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
    print("✅ statsmodels")
except ImportError:
    HAS_STATSMODELS = False

try:
    import jpholiday
    HAS_JPHOLIDAY = True
except ImportError:
    HAS_JPHOLIDAY = False

print(f"✅ 並列処理: {'Joblib' if HAS_JOBLIB else 'ThreadPool'}")
print("\n✅ ライブラリ読み込み完了\n")


# =============================================================================
# 高速特徴量エンジニアリング（ベクトル化最適化）
# =============================================================================
def create_features_fast(df, target_col='合計'):
    """高速特徴量生成（ベクトル化処理）"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # 日付特徴量（ベクトル化）
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

    # バイナリ特徴量
    df['is_month_start'] = (df['day'] <= 3).astype(np.int8)
    df['is_month_end'] = (df['day'] >= 28).astype(np.int8)
    df['is_weekend'] = (df['weekday'] >= 5).astype(np.int8)
    df['is_monday'] = (df['weekday'] == 0).astype(np.int8)
    df['is_friday'] = (df['weekday'] == 4).astype(np.int8)

    # 祝日（高速化: 一括処理）
    if HAS_JPHOLIDAY:
        dates = df['Date'].values
        df['is_holiday'] = np.array([1 if jpholiday.is_holiday(pd.Timestamp(d)) else 0
                                     for d in dates], dtype=np.int8)
    else:
        df['is_holiday'] = 0

    # ターゲット関連特徴量
    if target_col in df.columns:
        target = df[target_col].values

        # ラグ特徴量（重要なもののみ）
        for lag in [1, 7, 14, 28, 365]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

        # ローリング統計（重要な窓のみ）
        for window in [7, 14, 30]:
            rolled = df[target_col].shift(1).rolling(window, min_periods=1)
            df[f'rolling_mean_{window}'] = rolled.mean()
            df[f'rolling_std_{window}'] = rolled.std()

        # EMA
        for span in [7, 14]:
            df[f'ema_{span}'] = df[target_col].shift(1).ewm(span=span, adjust=False).mean()

        # グループ統計
        df['weekday_mean'] = df.groupby('weekday')[target_col].transform('mean')
        df['month_mean'] = df.groupby('month')[target_col].transform('mean')

    return df.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# 高速MLモデルクラス
# =============================================================================
class FastMLPredictor:
    """高速MLモデル予測（並列処理対応）"""

    def __init__(self, n_jobs=-1):
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_cols = None
        self.is_trained = False
        self.model_scores = {}
        self.model_weights = {}
        self.n_jobs = n_jobs
        self.best_models = []  # 精度上位モデルのみ使用

    def _get_models(self):
        """最適化されたモデル群"""
        models = {}

        # Tier 1: 高速・高精度（優先）
        if HAS_LGB:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.08, max_depth=6,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, n_jobs=-1
            )

        if HAS_XGB:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.08, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1
            )

        if HAS_CAT:
            models['CatBoost'] = CatBoostRegressor(
                iterations=300, learning_rate=0.08, depth=5,
                random_state=42, verbose=0, thread_count=-1
            )

        # Tier 2: バランス型
        if HAS_HISTGB:
            models['HistGB'] = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_depth=8,
                min_samples_leaf=20, random_state=42
            )

        models['RandomForest'] = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )

        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )

        # Tier 3: 軽量モデル（アンサンブル多様性用）
        models['Ridge'] = Ridge(alpha=1.0)
        models['BayesianRidge'] = BayesianRidge()
        models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5)

        return models

    def _train_single_model(self, name, model, X, y, tscv):
        """単一モデルの訓練（並列処理用）"""
        try:
            start = time.time()
            cv_scores = cross_val_score(
                model, X, y, cv=tscv,
                scoring='neg_mean_absolute_error', n_jobs=1
            )
            cv_mae = -cv_scores.mean()
            model.fit(X, y)
            elapsed = time.time() - start
            return name, model, cv_mae, elapsed
        except Exception as e:
            return name, None, float('inf'), 0

    def train(self, df, target_col='合計'):
        """高速並列訓練"""
        print("\n" + "="*70)
        print("【高速ML訓練】並列処理")
        print("="*70)

        start_total = time.time()

        df_feat = create_features_fast(df, target_col)
        df_train = df_feat.dropna(subset=[target_col])

        exclude = ['Date', target_col, 'year']
        self.feature_cols = [c for c in df_train.columns
                            if c not in exclude
                            and df_train[c].dtype in ['int64', 'float64', 'int32', 'float32']]

        X = df_train[self.feature_cols].fillna(0).values
        y = df_train[target_col].values

        if len(X) < 50:
            return False

        print(f"  訓練データ: {len(X)}件, 特徴量: {len(self.feature_cols)}個")

        X_scaled = self.scaler.fit_transform(X)
        models = self._get_models()
        tscv = TimeSeriesSplit(n_splits=3)  # 高速化のため3分割

        print(f"  🚀 {len(models)}モデルを並列訓練中...")

        # 並列訓練
        results = []
        if HAS_JOBLIB:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._train_single_model)(name, model, X_scaled, y, tscv)
                for name, model in models.items()
            )
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._train_single_model, name, model, X_scaled, y, tscv): name
                    for name, model in models.items()
                }
                for future in as_completed(futures):
                    results.append(future.result())

        # 結果処理
        for name, model, mae, elapsed in results:
            if model is not None:
                self.models[name] = model
                self.model_scores[name] = mae
                print(f"    ✅ {name:15s}: MAE={mae:,.0f} ({elapsed:.1f}s)")

        # 重み計算（精度の逆数）
        if self.model_scores:
            total_inv = sum(1/(s+1) for s in self.model_scores.values())
            self.model_weights = {n: (1/(s+1))/total_inv for n, s in self.model_scores.items()}

            # 上位モデルを選択（精度向上のため）
            sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1])
            self.best_models = [m[0] for m in sorted_models[:5]]

        self.is_trained = True
        elapsed_total = time.time() - start_total
        print(f"\n✅ 訓練完了！ 総時間: {elapsed_total:.1f}秒")
        return True

    def predict(self, df, target_date, target_col='合計'):
        """高速予測"""
        if not self.is_trained:
            return None, {}

        df_feat = create_features_fast(df, target_col)
        target_dt = pd.to_datetime(target_date)
        target_row = df_feat[df_feat['Date'] == target_dt]

        if len(target_row) == 0:
            last_row = df_feat.iloc[-1:].copy()
            last_row['Date'] = target_dt
            for col in ['year', 'month', 'day', 'weekday']:
                last_row[col] = getattr(target_dt, col if col != 'weekday' else 'dayofweek')
            target_row = last_row

        X = target_row[self.feature_cols].fillna(0).values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        predictions = {}
        for name in self.best_models:  # 上位モデルのみ使用
            if name in self.models:
                try:
                    pred = self.models[name].predict(X_scaled)[0]
                    predictions[name] = max(0, pred)
                except:
                    pass

        if not predictions:
            return None, {}

        # 加重平均
        ensemble = sum(predictions[k] * self.model_weights.get(k, 1/len(predictions))
                      for k in predictions)

        return ensemble, {'predictions': predictions, 'count': len(predictions)}


# =============================================================================
# 高速統計モデルクラス
# =============================================================================
class FastStatPredictor:
    """高速統計予測"""

    def __init__(self):
        self.ts_data = None
        self.is_ready = False

    def prepare(self, df, target_col='合計'):
        """データ準備"""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')
        df = df[df[target_col] > 0]
        self.ts_data = df.set_index('Date')[target_col]
        self.is_ready = len(self.ts_data) >= 30
        return self.is_ready

    def predict(self, target_date):
        """高速統計予測"""
        if not self.is_ready:
            return {}

        target_dt = pd.to_datetime(target_date)
        ts = self.ts_data[self.ts_data.index < target_dt]

        if len(ts) < 14:
            return {}

        predictions = {}

        # 1. 移動平均系（高速）
        predictions['SMA_7'] = ts.tail(7).mean()
        predictions['SMA_14'] = ts.tail(14).mean()
        predictions['EMA_7'] = ts.ewm(span=7).mean().iloc[-1]

        # 2. 加重移動平均
        weights = np.arange(1, 8)
        predictions['WMA_7'] = np.average(ts.tail(7), weights=weights)

        # 3. 同曜日平均
        ts_df = ts.reset_index()
        ts_df.columns = ['Date', 'value']
        ts_df['weekday'] = ts_df['Date'].dt.dayofweek
        same_wd = ts_df[ts_df['weekday'] == target_dt.dayofweek]['value']
        if len(same_wd) > 0:
            predictions['Weekday_Mean'] = same_wd.mean()
            predictions['Weekday_Recent'] = same_wd.tail(4).mean()

        # 4. 前年同日
        prev_year = target_dt - timedelta(days=365)
        if prev_year in ts.index:
            predictions['PrevYear'] = ts[prev_year]

        # 5. Holt-Winters（条件付き）
        if HAS_STATSMODELS and len(ts) >= 30:
            try:
                model = ExponentialSmoothing(ts.values, trend='add', seasonal=None)
                fit = model.fit(optimized=True)
                predictions['ExpSmooth'] = fit.forecast(1)[0]
            except:
                pass

        return predictions


# =============================================================================
# 自動精度チューニング
# =============================================================================
class AutoTuner:
    """自動精度チューニング"""

    def __init__(self):
        self.history = []
        self.best_weights = {'ml': 0.6, 'stat': 0.4}
        self.correction_factor = 0

    def record(self, prediction, actual):
        """予測結果を記録"""
        error = prediction - actual
        error_pct = error / actual * 100 if actual > 0 else 0
        self.history.append({
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'error_pct': error_pct,
            'timestamp': datetime.now()
        })

        # 直近の傾向から補正係数を更新
        if len(self.history) >= 5:
            recent_errors = [h['error'] for h in self.history[-10:]]
            self.correction_factor = -np.mean(recent_errors) * 0.5

    def get_correction(self):
        """補正値を取得"""
        return self.correction_factor

    def get_accuracy_score(self):
        """精度スコアを計算（100点満点）"""
        if len(self.history) < 3:
            return 70  # デフォルト

        recent = self.history[-30:] if len(self.history) >= 30 else self.history
        mape = np.mean([abs(h['error_pct']) for h in recent])

        # MAPE → 100点スコア変換
        # 0% → 100点, 5% → 90点, 10% → 80点, 20% → 60点
        score = max(0, min(100, 100 - mape * 2))
        return score


# =============================================================================
# 統合予測システム
# =============================================================================
class OptimizedPredictionSystem:
    """v67.0 高速・高精度予測システム"""

    def __init__(self):
        self.ml_predictor = FastMLPredictor()
        self.stat_predictor = FastStatPredictor()
        self.auto_tuner = AutoTuner()
        self.is_initialized = False
        self.df_data = None

    def initialize(self, df):
        """高速初期化"""
        print("\n" + "="*70)
        print("【v67.0 システム初期化】高速・高精度モード")
        print("="*70)

        start = time.time()
        self.df_data = df.copy()

        # ML訓練
        self.ml_predictor.train(self.df_data, '合計')

        # 統計準備
        self.stat_predictor.prepare(self.df_data, '合計')

        self.is_initialized = True
        elapsed = time.time() - start
        print(f"\n✅ 初期化完了！ 総時間: {elapsed:.1f}秒")

    def predict(self, target_date, actuals=None):
        """高速高精度予測"""
        if not self.is_initialized:
            return None

        actuals = actuals or {}
        start = time.time()

        print("\n" + "="*70)
        print("【v67.0 高速予測】")
        print("="*70)
        print(f"予測日: {target_date}")

        df = self.df_data.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['合計'] = pd.to_numeric(df['合計'], errors='coerce').fillna(0)
        df = df[df['合計'] > 0]

        all_predictions = {}

        # ML予測
        ml_pred, ml_info = self.ml_predictor.predict(df, target_date)
        if ml_pred:
            all_predictions['ML'] = ml_pred
            print(f"\n  ML予測: {ml_pred:,.0f}件 ({ml_info['count']}モデル)")

        # 統計予測
        stat_preds = self.stat_predictor.predict(target_date)
        for name, pred in stat_preds.items():
            if pred > 0:
                all_predictions[f'Stat_{name}'] = pred

        print(f"  統計予測: {len(stat_preds)}モデル")

        if not all_predictions:
            return None

        # アンサンブル（ML重視）
        ml_values = [v for k, v in all_predictions.items() if k == 'ML']
        stat_values = [v for k, v in all_predictions.items() if k.startswith('Stat_')]

        if ml_values and stat_values:
            ensemble = np.mean(ml_values) * 0.65 + np.median(stat_values) * 0.35
        elif ml_values:
            ensemble = np.mean(ml_values)
        else:
            ensemble = np.median(stat_values)

        # 自動チューニング補正
        correction = self.auto_tuner.get_correction()
        ensemble += correction

        # 12時実績調整
        if actuals.get(12, 0) > 0:
            ratio = 0.45
            if '件数(～12:00)' in df.columns:
                df_12h = df[(df['件数(～12:00)'] > 0)]
                if len(df_12h) > 0:
                    ratio = (df_12h['件数(～12:00)'] / df_12h['合計']).mean()
            est = actuals[12] / ratio
            ensemble = ensemble * 0.3 + est * 0.7
            print(f"  12時調整: {actuals[12]} → 推定 {est:,.0f}件")

        # 信頼区間
        values = list(all_predictions.values())
        std = np.std(values)
        lower = max(0, ensemble - 1.96 * std)
        upper = ensemble + 1.96 * std

        # 精度スコア
        score = self.auto_tuner.get_accuracy_score()

        elapsed = time.time() - start

        print("\n" + "="*70)
        print(f"★★★ 最終予測: {int(ensemble):,}件 ★★★")
        print(f"    信頼区間: {int(lower):,} ～ {int(upper):,}件")
        print(f"    精度スコア: {score:.0f}点")
        print(f"    処理時間: {elapsed:.2f}秒")
        print("="*70)

        return {
            'final_prediction': int(ensemble),
            'pred_lower': int(lower),
            'pred_upper': int(upper),
            'confidence_score': score,
            'processing_time': elapsed,
            'model_count': len(all_predictions)
        }

    def update_actual(self, target_date, actual_value):
        """実績値を登録して精度向上"""
        # 直近の予測を取得して記録
        if hasattr(self, '_last_prediction'):
            self.auto_tuner.record(self._last_prediction, actual_value)
            print(f"✅ 実績登録: {actual_value}件 → 精度スコア更新")


# =============================================================================
# Colab UI
# =============================================================================
def create_optimized_ui():
    """最適化版UI"""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
    except:
        return

    system = [None]
    df_data = [None]

    date_picker = widgets.DatePicker(description='予測日:', value=date.today())
    actual_12h = widgets.IntText(value=0, description='12時実績:')
    run_btn = widgets.Button(description='⚡ 高速予測', button_style='success',
                            layout=widgets.Layout(width='180px', height='50px'))
    init_btn = widgets.Button(description='🚀 初期化', button_style='warning',
                             layout=widgets.Layout(width='120px', height='50px'))
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
                system[0] = OptimizedPredictionSystem()
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
                        <div style='background:linear-gradient(135deg,#00b09b,#96c93d);color:white;padding:25px;border-radius:15px;text-align:center;margin:20px 0;'>
                            <h2>⚡ v67.0【高速・高精度版】</h2>
                            <div style='font-size:56px;font-weight:bold;margin:20px 0;'>{res['final_prediction']:,}件</div>
                            <div>信頼区間: {res['pred_lower']:,} ～ {res['pred_upper']:,}件</div>
                            <div>精度スコア: {res['confidence_score']:.0f}点 | 処理: {res['processing_time']:.2f}秒</div>
                        </div>
                        """))
        run_btn.disabled = False

    init_btn.on_click(on_init)
    run_btn.on_click(on_run)

    ui = widgets.VBox([
        widgets.HTML('<h2 style="color:#00b09b;">⚡ v67.0 高速・高精度AI予測</h2>'),
        widgets.HTML('<p>並列処理 + 自動チューニングで精度100点を目指す</p>'),
        date_picker, actual_12h,
        widgets.HBox([run_btn, init_btn]),
        result
    ])
    display(ui, output)


# =============================================================================
# メイン
# =============================================================================
if __name__ == "__main__":
    print("\n【v67.0 搭載機能】")
    print("-"*50)
    print("✅ 並列ML訓練（Joblib/ThreadPool）")
    print("✅ 上位5モデル自動選択")
    print("✅ 高速特徴量エンジニアリング")
    print("✅ 自動精度チューニング")
    print("✅ 実績フィードバック学習")
    print("-"*50)

    try:
        import google.colab
        create_optimized_ui()
    except:
        pass
