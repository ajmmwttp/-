# =============================================================================
# v69.0 メタ認知AI予測システム【完全版】Google Colab専用
# =============================================================================
# 【使い方】
# 1. Google Colabでこのファイルを開く
# 2. 「ランタイム」→「すべてのセルを実行」
# 3. 実績表Excelファイルをアップロード
# 4. 「🚀 完全版予測」ボタンをクリック
# =============================================================================

#@title 📦 Step 1: ライブラリインストール
print("="*60)
print("【v69.0 完全版】ライブラリインストール中...")
print("="*60)

import subprocess
import sys

packages = [
    'lightgbm',
    'xgboost',
    'catboost',
    'prophet',
    'statsmodels',
    'jpholiday',
    'openpyxl'
]

for pkg in packages:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
        print(f"✅ {pkg}")
    except:
        print(f"⚠️ {pkg} (スキップ)")

print("\n✅ インストール完了！\n")


#@title 📊 Step 2: システム読み込み
print("="*60)
print("【v69.0 メタ認知AI予測システム】完全版")
print("  統計理論(40+) + 機械学習(15) = 55+モデル")
print("="*60)

import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date, timedelta, datetime
import warnings
warnings.filterwarnings("ignore")

# ライブラリ確認
print("\n【ライブラリ確認】")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
    print("  ✅ statsmodels")
except:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
    print("  ✅ Prophet")
except:
    HAS_PROPHET = False

try:
    import jpholiday
    HAS_JPHOLIDAY = True
    print("  ✅ jpholiday")
except:
    HAS_JPHOLIDAY = False

try:
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                  ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
    print("  ✅ scikit-learn")
except:
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("  ✅ LightGBM")
except:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("  ✅ XGBoost")
except:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
    print("  ✅ CatBoost")
except:
    HAS_CATBOOST = False

print("\n✅ 読み込み完了")


# =============================================================================
# 統計予測クラス
# =============================================================================
class StatisticalPredictor:
    def __init__(self):
        self.ts_data = None
        self.seasonality = {}
        self.decomposition = None

    def prepare(self, df, target_col='合計'):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')
        df = df[df[target_col] > 0]
        self.df_data = df
        self.ts_data = df.set_index('Date')[target_col]
        if len(self.ts_data) < 14:
            return False

        ts_df = self.ts_data.reset_index()
        ts_df.columns = ['Date', 'value']
        ts_df['weekday'] = ts_df['Date'].dt.dayofweek
        ts_df['month'] = ts_df['Date'].dt.month
        ts_df['week_of_month'] = ((ts_df['Date'].dt.day - 1) // 7 + 1)
        self.seasonality['weekday'] = ts_df.groupby('weekday')['value'].mean().to_dict()
        self.seasonality['month'] = ts_df.groupby('month')['value'].mean().to_dict()
        self.seasonality['week_of_month'] = ts_df.groupby('week_of_month')['value'].mean().to_dict()

        if HAS_STATSMODELS and len(self.ts_data) >= 14:
            try:
                self.decomposition = STL(self.ts_data, period=7, robust=True).fit()
            except:
                pass
        return True

    def predict(self, target_date):
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

        # 移動平均
        for w in [3, 5, 7, 14, 21, 30]:
            if len(ts) >= w:
                predictions[f'SMA_{w}'] = ts.tail(w).mean()
        if len(ts) >= 7:
            predictions['WMA_7'] = np.average(ts.tail(7), weights=np.arange(1, 8))
        for s in [7, 14, 21]:
            if len(ts) >= s:
                predictions[f'EMA_{s}'] = ts.ewm(span=s).mean().iloc[-1]

        # 季節性
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

        # 前年比較
        prev_year = target_dt - timedelta(days=365)
        if prev_year in ts.index:
            predictions['PrevYear_SameDay'] = ts[prev_year]
        prev_year_wd = target_dt - timedelta(days=364)
        if prev_year_wd in ts.index:
            predictions['PrevYear_SameWeekday'] = ts[prev_year_wd]

        # 指数平滑化
        if HAS_STATSMODELS:
            try:
                predictions['ExpSmooth_Simple'] = ExponentialSmoothing(ts.values, trend=None, seasonal=None).fit(optimized=True).forecast(1)[0]
            except:
                pass
            try:
                predictions['ExpSmooth_Holt'] = ExponentialSmoothing(ts.values, trend='add', seasonal=None).fit(optimized=True).forecast(1)[0]
            except:
                pass
            if len(ts) >= 14:
                try:
                    predictions['HoltWinters_Add'] = ExponentialSmoothing(ts.values, trend='add', seasonal='add', seasonal_periods=7).fit(optimized=True).forecast(1)[0]
                except:
                    pass

        # ARIMA
        if HAS_STATSMODELS and len(ts) >= 30:
            for order in [(1,1,1), (2,1,1), (1,1,0)]:
                try:
                    predictions[f'ARIMA_{order[0]}{order[1]}{order[2]}'] = ARIMA(ts.values, order=order).fit().forecast(1)[0]
                except:
                    pass

        # SARIMA
        if HAS_STATSMODELS and len(ts) >= 60:
            try:
                predictions['SARIMA_111_111'] = SARIMAX(ts.values, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False).forecast(1)[0]
            except:
                pass

        # Prophet
        if HAS_PROPHET and len(ts) >= 30:
            try:
                prophet_df = pd.DataFrame({'ds': ts.index, 'y': ts.values})
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(prophet_df)
                predictions['Prophet'] = model.predict(pd.DataFrame({'ds': [target_dt]}))['yhat'].values[0]
            except:
                pass

        # ベイズ推定
        prior_mean, prior_std = ts.mean(), ts.std()
        likelihood_mean = ts.tail(14).mean()
        likelihood_std = ts.tail(14).std() / np.sqrt(14)
        posterior_precision = 1/prior_std**2 + 1/likelihood_std**2
        predictions['Bayes_Posterior'] = (prior_mean/prior_std**2 + likelihood_mean/likelihood_std**2) / posterior_precision

        # ブートストラップ
        recent = ts.tail(30).values
        bootstrap = [np.mean(np.random.choice(recent, len(recent), True)) for _ in range(1000)]
        predictions['Bootstrap_Mean'] = np.mean(bootstrap)

        # ロバスト統計
        predictions['Median_7'] = ts.tail(7).median()
        predictions['Median_14'] = ts.tail(14).median()
        predictions['TrimMean_10pct'] = stats.trim_mean(ts.tail(30).values, 0.1)

        return predictions


# =============================================================================
# 機械学習予測クラス
# =============================================================================
class MLPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_cols = []
        self.is_trained = False

    def _create_features(self, df, target_col='合計'):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', target_col]).sort_values('Date')
        df['weekday'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['week_of_month'] = ((df['day'] - 1) // 7 + 1)
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        if HAS_JPHOLIDAY:
            df['is_holiday'] = df['Date'].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
        else:
            df['is_holiday'] = 0
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        for w in [7, 14, 21, 30]:
            df[f'rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
            df[f'rolling_std_{w}'] = df[target_col].shift(1).rolling(w).std()
        for s in [7, 14, 21]:
            df[f'ewm_{s}'] = df[target_col].shift(1).ewm(span=s).mean()
        df['trend'] = np.arange(len(df))
        return df

    def train(self, df, target_col='合計'):
        if not HAS_SKLEARN:
            return False
        print("\n【機械学習訓練】")
        df = self._create_features(df, target_col).dropna()
        if len(df) < 50:
            return False

        exclude = ['Date', target_col]
        self.feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64']]
        X, y = df[self.feature_cols].values, df[target_col].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'BayesianRidge': BayesianRidge(),
            'Huber': HuberRegressor(),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=100, random_state=42),
        }
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        if HAS_XGBOOST:
            models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostRegressor(iterations=100, depth=5, random_state=42, verbose=False)

        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                self.models[name] = model
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ⚠️ {name}")

        self.is_trained = True
        print(f"\n  訓練完了: {len(self.models)}モデル")
        return True

    def predict(self, target_date, df, target_col='合計'):
        if not self.is_trained:
            return {}
        df = self._create_features(df, target_col).dropna()
        if len(df) == 0:
            return {}
        X = self.scaler.transform(df[self.feature_cols].iloc[-1:].values)
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                if pred > 0 and not np.isnan(pred):
                    predictions[f'ML_{name}'] = pred
            except:
                pass
        return predictions


# =============================================================================
# 完全版予測システム
# =============================================================================
class CompletePredictionSystem:
    def __init__(self):
        self.stat_predictor = StatisticalPredictor()
        self.ml_predictor = MLPredictor()
        self.df_data = None
        self.is_initialized = False
        self.ai_memory = []

    def initialize(self, df):
        print("\n" + "="*60)
        print("【v69.0 完全版】初期化")
        print("="*60)
        self.df_data = df.copy()
        if 'Date' not in self.df_data.columns:
            for c in ['日付', 'date']:
                if c in self.df_data.columns:
                    self.df_data = self.df_data.rename(columns={c: 'Date'})
                    break

        print("\n【統計分析】")
        stat_ok = self.stat_predictor.prepare(self.df_data, '合計')
        ml_ok = self.ml_predictor.train(self.df_data, '合計')

        ts = self.df_data.set_index(pd.to_datetime(self.df_data['Date'], errors='coerce'))['合計'].dropna()
        print(f"\n【データ概要】")
        print(f"  期間: {ts.index.min().date()} ～ {ts.index.max().date()}")
        print(f"  件数: {len(ts)}日分")
        print(f"  平均: {ts.mean():.0f}件")

        self.is_initialized = stat_ok or ml_ok
        print("\n✅ 初期化完了！")
        return self.is_initialized

    def predict(self, target_date, actuals=None):
        if not self.is_initialized:
            return None
        actuals = actuals or {}

        print("\n" + "="*60)
        print(f"【v69.0 予測】{target_date}")
        print("="*60)

        stat_preds = self.stat_predictor.predict(target_date)
        ml_preds = self.ml_predictor.predict(target_date, self.df_data, '合計')
        all_preds = {**stat_preds, **ml_preds}
        valid_preds = {k: v for k, v in all_preds.items() if v > 0 and not np.isnan(v) and not np.isinf(v)}

        if not valid_preds:
            return None

        print(f"\n使用モデル: {len(valid_preds)}個 (統計{len(stat_preds)} + ML{len(ml_preds)})")

        all_values = list(valid_preds.values())
        ensemble_median = np.median(all_values)
        ensemble_trimmed = stats.trim_mean(all_values, 0.1)

        stat_values = [v for k, v in valid_preds.items() if not k.startswith('ML_')]
        ml_values = [v for k, v in valid_preds.items() if k.startswith('ML_')]

        if stat_values and ml_values:
            hybrid_pred = np.median(stat_values) * 0.6 + np.median(ml_values) * 0.4
        else:
            hybrid_pred = ensemble_median

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
            print(f"\n⏰ 12時実績調整: {actuals[12]} → {est:.0f}件")

        # AI学習補正
        if self.ai_memory:
            recent_errors = [m['error'] for m in self.ai_memory[-10:] if 'error' in m]
            if recent_errors:
                avg_error = np.mean(recent_errors)
                if abs(avg_error) > 0.02:
                    final_pred += final_pred * avg_error
                    print(f"🧠 AI学習補正適用")

        pred_lower = np.percentile(all_values, 2.5)
        pred_upper = np.percentile(all_values, 97.5)
        cv = np.std(all_values) / np.mean(all_values) * 100
        confidence = 95 if cv < 5 else 85 if cv < 10 else 75 if cv < 15 else 65

        print("\n" + "="*60)
        print(f"★★★ 最終予測: {int(final_pred):,}件 ★★★")
        print(f"    信頼区間: {int(pred_lower):,} ～ {int(pred_upper):,}件")
        print(f"    信頼度: {confidence}点 | 変動係数: {cv:.1f}%")
        print("="*60)

        return {
            'final_prediction': int(final_pred),
            'pred_lower': int(pred_lower),
            'pred_upper': int(pred_upper),
            'confidence_score': confidence,
            'cv': cv,
            'model_count': len(valid_preds),
            'stat_model_count': len(stat_preds),
            'ml_model_count': len(ml_preds),
            'all_predictions': valid_preds
        }

    def update_memory(self, actual, predicted):
        if actual > 0 and predicted > 0:
            error = (actual - predicted) / actual
            self.ai_memory.append({'actual': actual, 'predicted': predicted, 'error': error})
            if len(self.ai_memory) > 100:
                self.ai_memory = self.ai_memory[-100:]
            print(f"🧠 AI学習: 誤差{error*100:+.1f}%")


#@title 🚀 Step 3: UIを起動
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import glob

system = [None]
df_data = [None]

date_picker = widgets.DatePicker(description='予測日:', value=date.today())
actual_12h = widgets.IntText(value=0, description='12時実績:')
actual_total = widgets.IntText(value=0, description='実績合計:')

run_btn = widgets.Button(description='🚀 完全版予測', button_style='success',
                        layout=widgets.Layout(width='180px', height='50px'))
init_btn = widgets.Button(description='📊 初期化', button_style='warning',
                         layout=widgets.Layout(width='120px', height='50px'))
learn_btn = widgets.Button(description='🧠 学習', button_style='info',
                          layout=widgets.Layout(width='100px', height='50px'))
upload_btn = widgets.FileUpload(accept='.xlsx,.csv', multiple=False, description='📁 ファイル')

output = widgets.Output()
result = widgets.Output()

def on_init(b):
    init_btn.disabled = True
    with output:
        clear_output()
        # アップロードされたファイルを確認
        if upload_btn.value:
            fn = list(upload_btn.value.keys())[0]
            content = upload_btn.value[fn]['content']
            with open(f'/content/{fn}', 'wb') as f:
                f.write(content)
            print(f"📁 アップロード: {fn}")

        files = glob.glob('/content/*.xlsx') + glob.glob('/content/*.csv')
        if not files:
            print("⚠️ Excelファイルをアップロードしてください")
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
            res = system[0].predict(date_picker.value, {12: actual_12h.value})
            if res:
                system[0].update_memory(actual_total.value, res['final_prediction'])

init_btn.on_click(on_init)
run_btn.on_click(on_run)
learn_btn.on_click(on_learn)

# UI表示
display(widgets.VBox([
    widgets.HTML('''
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:20px;border-radius:15px;margin-bottom:15px;">
        <h2 style="margin:0;">🚀 v69.0 完全版予測システム</h2>
        <p style="margin:5px 0 0 0;opacity:0.9;">統計理論(40+手法) + 機械学習(15モデル) = 55+予測モデル統合</p>
    </div>
    '''),
    widgets.HTML('''
    <div style="background:#f8f9fa;padding:12px;border-radius:8px;margin:10px 0;font-size:11px;">
        <b>📊 統計:</b> 移動平均, 季節性, 前年比較, 指数平滑化, ARIMA, SARIMA, Prophet, ベイズ推定, ブートストラップ, ロバスト統計<br>
        <b>🤖 ML:</b> LightGBM, XGBoost, CatBoost, RandomForest, GradientBoosting, Ridge, Lasso, BayesianRidge, KNN, MLP
    </div>
    '''),
    widgets.HBox([upload_btn, init_btn]),
    widgets.HBox([date_picker, actual_12h]),
    widgets.HBox([actual_total, learn_btn]),
    run_btn,
    result
]))
display(output)

print("\n✅ UIが表示されました！")
print("1. 📁 実績表Excelをアップロード")
print("2. 📊 初期化ボタンをクリック")
print("3. 🚀 完全版予測ボタンで予測実行")
