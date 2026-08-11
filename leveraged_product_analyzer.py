#!/usr/bin/env python3
"""
Leveraged product analyzer — SPXL / TQQQ / SOXL / GDXU.

WHY THIS EXISTS
    The 2026-08-11 research report could not be completed because every
    market-data primary source was blocked by the network egress policy
    (62 reachability attempts, 0 successes). The report therefore recorded
    all price-dependent items as "算定不能" rather than fabricating them.

    This module is the other half: it takes the daily OHLCV series once the
    user supplies them and computes every price-dependent metric the research
    protocol specifies, using exactly the conventions the protocol mandates.
    Nothing here invents data. Given no input, it produces no numbers.

CONVENTIONS (fixed by the research protocol, not by preference)
    RSI            Wilder smoothing, period 14
    ATR            True Range with Wilder smoothing, period 14
    MACD           EMA(12) - EMA(26), signal EMA(9), histogram = MACD - signal
    Bollinger      SMA(20) +/- 2 population standard deviations
    Realized vol   stdev of daily LOG returns, annualised by sqrt(252)
    3 months       63 trading days
    52 weeks       252 trading days
    Breakouts      judged on confirmed CLOSES, never intraday highs
    Prices         must be split/reverse-split adjusted by the caller

DEPENDENCIES
    Python 3 standard library only. No pandas, no numpy.

USAGE
    python3 leveraged_product_analyzer.py --selftest
    python3 leveraged_product_analyzer.py --template > spxl.csv
    python3 leveraged_product_analyzer.py --product SPXL.csv --index SP500.csv \
        --name SPXL --index-name "S&P 500"
"""

import argparse
import csv
import math
import sys

TRADING_DAYS_YEAR = 252
QUARTER_DAYS = 63

DATE_KEYS = ("date", "Date", "DATE", "日付", "日付け", "年月日")
FIELD_ALIASES = {
    "open": ("open", "Open", "OPEN", "始値", "始値(円)"),
    "high": ("high", "High", "HIGH", "高値"),
    "low": ("low", "Low", "LOW", "安値"),
    "close": ("close", "Close", "CLOSE", "終値", "adj_close", "Adj Close", "調整後終値"),
    "volume": ("volume", "Volume", "VOLUME", "出来高"),
}


# ----------------------------------------------------------------- loading
def _pick(row, aliases):
    for a in aliases:
        if a in row and str(row[a]).strip() != "":
            return row[a]
    return None


def _num(v):
    if v is None:
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "--", "N/A", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_series(path, require_ohlc=True):
    """Load a daily series from CSV/TSV. Returns list of dicts, oldest first.

    Tolerant of English and Japanese column names. Rows with an unparseable
    close are dropped and counted, never silently interpolated.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(fh, dialect=dialect))

    if not rows:
        raise ValueError(f"{path}: no data rows")

    out, dropped = [], 0
    for r in rows:
        date = _pick(r, DATE_KEYS)
        close = _num(_pick(r, FIELD_ALIASES["close"]))
        if date is None or close is None:
            dropped += 1
            continue
        rec = {"date": str(date).strip(), "close": close}
        for f in ("open", "high", "low", "volume"):
            rec[f] = _num(_pick(r, FIELD_ALIASES[f]))
        if require_ohlc and (rec["high"] is None or rec["low"] is None):
            rec["high"] = rec["high"] if rec["high"] is not None else close
            rec["low"] = rec["low"] if rec["low"] is not None else close
        out.append(rec)

    if not out:
        raise ValueError(f"{path}: no parseable rows (checked columns {list(rows[0].keys())})")

    # Detect ordering: if dates descend, reverse to oldest-first.
    if len(out) > 1 and out[0]["date"] > out[-1]["date"]:
        out.reverse()

    # Duplicate dates must be a hard error, never silently sorted away.
    # Sorting a series with repeated dates interleaves unrelated bars and
    # corrupts every downstream metric while still producing plausible
    # numbers -- the exact failure mode this report exists to avoid.
    seen, dupes = set(), []
    for r in out:
        if r["date"] in seen:
            dupes.append(r["date"])
        seen.add(r["date"])
    if dupes:
        u = sorted(set(dupes))
        raise ValueError(
            f"{path}: 日付が重複しています（{len(dupes)}件）: {u[:5]}"
            f"{' ...' if len(u) > 5 else ''}\n"
            "  重複したまま並べ替えると全指標が静かに壊れるため処理を中止します。"
            "  重複行を除去してから再実行してください。"
        )

    # Warn (do not fail) if the input was not already chronological.
    unsorted = any(out[i]["date"] > out[i + 1]["date"] for i in range(len(out) - 1))
    out.sort(key=lambda x: x["date"])
    if unsorted:
        print(f"(注意: {path} は日付順ではなかったため昇順に並べ替えました)",
              file=sys.stderr)
    return out, dropped


# -------------------------------------------------------------- indicators
def sma(vals, n):
    if len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def ema_series(vals, n):
    """EMA seeded with the SMA of the first n observations."""
    if len(vals) < n:
        return []
    k = 2.0 / (n + 1.0)
    e = sum(vals[:n]) / n
    out = [e]
    for v in vals[n:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi_wilder(closes, n=14):
    """Wilder RSI. Returns the latest value, or None if insufficient data."""
    if len(closes) < n + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))


def true_ranges(bars):
    tr = []
    for i in range(1, len(bars)):
        h, l = bars[i]["high"], bars[i]["low"]
        pc = bars[i - 1]["close"]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return tr


def atr_wilder(bars, n=14):
    tr = true_ranges(bars)
    if len(tr) < n:
        return None
    a = sum(tr[:n]) / n
    for v in tr[n:]:
        a = (a * (n - 1) + v) / n
    return a


def macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None
    ef, es = ema_series(closes, fast), ema_series(closes, slow)
    ef = ef[len(ef) - len(es):]  # align tails
    line = [f - s for f, s in zip(ef, es)]
    sig = ema_series(line, signal)
    if not sig:
        return None
    m, s = line[-1], sig[-1]
    return {"macd": m, "signal": s, "hist": m - s}


def bollinger(closes, n=20, k=2.0):
    if len(closes) < n:
        return None
    w = closes[-n:]
    m = sum(w) / n
    var = sum((x - m) ** 2 for x in w) / n  # population sd, per spec
    sd = math.sqrt(var)
    return {"mid": m, "upper": m + k * sd, "lower": m - k * sd, "sd": sd}


def log_returns(closes):
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0]


def realized_vol(closes, n):
    lr = log_returns(closes)
    if len(lr) < n or n < 2:
        return None
    w = lr[-n:]
    m = sum(w) / len(w)
    var = sum((x - m) ** 2 for x in w) / (len(w) - 1)  # sample sd
    return math.sqrt(var) * math.sqrt(TRADING_DAYS_YEAR)


def max_drawdown(closes):
    """Maximum drawdown, with the peak that actually precedes the trough.

    The running peak index must be captured AT the moment the deepest
    drawdown is recorded. Reading it after the loop returns the last peak in
    the series, which is a different bar whenever the series makes new highs
    after the max-drawdown trough — and that would also make the recovery
    test compare against the wrong reference price.
    """
    peak, peak_idx = closes[0], 0
    mdd, trough_idx, mdd_peak_idx = 0.0, 0, 0
    for i, c in enumerate(closes):
        if c > peak:
            peak, peak_idx = c, i
        dd = c / peak - 1.0
        if dd < mdd:
            mdd, trough_idx, mdd_peak_idx = dd, i, peak_idx
    recovered, ri = False, None
    for j in range(trough_idx, len(closes)):
        if closes[j] >= closes[mdd_peak_idx]:
            recovered, ri = True, j
            break
    return {"mdd": mdd, "peak_idx": mdd_peak_idx, "trough_idx": trough_idx,
            "recovered": recovered, "recover_idx": ri}


def pct_change(closes, n):
    if len(closes) < n + 1:
        return None
    return closes[-1] / closes[-1 - n] - 1.0


# ------------------------------------------------- 4-stage decomposition
def decompose(index_closes, product_closes, k=3.0):
    """The protocol's four-stage return attribution.

      1 index cumulative return
      2 naive k x that cumulative return
      3 theoretical path: daily index return x k, compounded daily (pre-cost)
      4 the product's actual realised return

    (2 - 3) is attributed mainly to path dependency / compounding.
    (3 - 4) is attributed mainly to fees, financing, tracking and price-vs-NAV.
    Residual that cannot be decided is left unattributed, by design.
    """
    n = min(len(index_closes), len(product_closes))
    if n < 2:
        return None
    ic, pc = index_closes[-n:], product_closes[-n:]

    s1 = ic[-1] / ic[0] - 1.0
    s2 = k * s1
    theo = 1.0
    for i in range(1, len(ic)):
        r = ic[i] / ic[i - 1] - 1.0
        theo *= (1.0 + k * r)
        if theo <= 0:
            theo = 0.0
            break
    s3 = theo - 1.0
    s4 = pc[-1] / pc[0] - 1.0
    return {"days": n - 1, "index": s1, "naive_kx": s2, "theoretical": s3,
            "actual": s4, "path_effect": s3 - s2, "cost_effect": s4 - s3}


# ------------------------------------------------------------- reporting
def _f(x, nd=2, pct=False, na="算定不能"):
    if x is None:
        return na
    return f"{x*100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def analyze(bars, name, index_bars=None, index_name=None):
    closes = [b["close"] for b in bars]
    last, prev = closes[-1], (closes[-2] if len(closes) > 1 else None)
    out = []
    A = out.append

    A(f"# {name}  ——  分析結果")
    A(f"データ期間: {bars[0]['date']} 〜 {bars[-1]['date']}  ({len(bars)}営業日)")
    A("")
    A("## 価格")
    A(f"  確定終値           {last:.4f}   ({bars[-1]['date']})")
    if prev:
        A(f"  前日比             {last-prev:+.4f}  ({_f(last/prev-1, 2, True)})")
    for lbl, n in (("5日", 5), ("20日", 20), ("3か月(63日)", QUARTER_DAYS),
                   ("52週(252日)", TRADING_DAYS_YEAR)):
        A(f"  {lbl:<16} {_f(pct_change(closes, n), 2, True)}")

    A("")
    A("## 移動平均と乖離率")
    for n in (20, 50, 200):
        m = sma(closes, n)
        dev = (last / m - 1.0) if m else None
        A(f"  SMA{n:<4}          {_f(m, 4)}   乖離 {_f(dev, 2, True)}")

    A("")
    A("## オシレーター")
    A(f"  RSI14 (Wilder)     {_f(rsi_wilder(closes), 2)}")
    md = macd(closes)
    if md:
        A(f"  MACD(12,26)        {md['macd']:.4f}   Signal(9) {md['signal']:.4f}   "
          f"Hist {md['hist']:+.4f}")
    else:
        A("  MACD               算定不能（データ不足）")
    bb = bollinger(closes)
    if bb:
        A(f"  BB(20,2sd)         下 {bb['lower']:.4f} / 中 {bb['mid']:.4f} / 上 {bb['upper']:.4f}")
    else:
        A("  BB(20,2sd)         算定不能（データ不足）")

    A("")
    A("## ボラティリティ")
    atr = atr_wilder(bars)
    A(f"  ATR14 (Wilder)     {_f(atr, 4)}   終値比 {_f(atr/last if atr else None, 2, True)}")
    for n in (20, 60):
        A(f"  実現ボラ{n}日(年率)  {_f(realized_vol(closes, n), 2, True)}")

    A("")
    A("## レンジ")
    for lbl, n in (("20日", 20), ("3か月(63日)", QUARTER_DAYS), ("52週(252日)", TRADING_DAYS_YEAR)):
        if len(closes) >= n:
            w = closes[-n:]
            hi, lo = max(w), min(w)
            A(f"  {lbl:<14} 高 {hi:.4f} / 安 {lo:.4f}   高値からの下落 {_f(last/hi-1,2,True)}")
        else:
            A(f"  {lbl:<14} 算定不能（{n}営業日分のデータが必要、現在{len(closes)}日）")

    A("")
    A("## ドローダウン")
    dd = max_drawdown(closes)
    A(f"  期間最大DD         {_f(dd['mdd'], 2, True)}  "
      f"(peak {bars[dd['peak_idx']]['date']} → trough {bars[dd['trough_idx']]['date']})")
    A(f"  回復必要率         {_f(-dd['mdd']/(1+dd['mdd']) if dd['mdd']>-1 else None, 2, True)}")
    A(f"  回復済みか         {'はい ('+bars[dd['recover_idx']]['date']+')' if dd['recovered'] else 'いいえ（未回復）'}")

    if index_bars:
        ic = [b["close"] for b in index_bars]
        A("")
        A(f"## 原指数との対比 ({index_name or 'index'})")
        for lbl, n in (("5日", 5), ("20日", 20), ("3か月", QUARTER_DAYS)):
            d = decompose(ic[-(n+1):], closes[-(n+1):]) if len(ic) > n and len(closes) > n else None
            if d:
                A(f"  [{lbl}] 指数 {_f(d['index'],2,True)} / 単純3倍 {_f(d['naive_kx'],2,True)}"
                  f" / 3倍複利理論 {_f(d['theoretical'],2,True)} / 実績 {_f(d['actual'],2,True)}")
                A(f"         経路依存 {_f(d['path_effect'],2,True)}"
                  f"   費用・追跡差 {_f(d['cost_effect'],2,True)}")
            else:
                A(f"  [{lbl}] 算定不能（データ不足）")
        rv_p, rv_i = realized_vol(closes, 20), realized_vol(ic, 20)
        if rv_p and rv_i and rv_i > 0:
            A(f"  実現ボラ倍率(20日)  {rv_p/rv_i:.2f}x  （日次3倍商品の理論値は約3.00x）")

    A("")
    A("## 押し目買い判定に必要な入力のうち、本データで確認できた項目")
    checks = []
    r = rsi_wilder(closes)
    m20, m50, m200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    checks.append(("RSIによる売られ過ぎ(<30)", None if r is None else r < 30,
                   "算定不能" if r is None else f"RSI={r:.1f}"))
    if len(closes) >= TRADING_DAYS_YEAR:
        hi = max(closes[-TRADING_DAYS_YEAR:])
        dfh = last / hi - 1
        checks.append(("直近高値から15〜30%下落", dfh <= -0.15, f"{dfh*100:.1f}%"))
    for lbl, m in (("20日線上", m20), ("50日線上", m50), ("200日線上", m200)):
        checks.append((lbl, None if m is None else last > m,
                       "算定不能" if m is None else f"{(last/m-1)*100:+.1f}%"))
    for lbl, ok, detail in checks:
        mark = "算定不能" if ok is None else ("該当" if ok else "非該当")
        A(f"  [{mark:^6}] {lbl:<24} {detail}")
    A("")
    A("  ※ 利益予想改定・バリュエーション・市場幅・信用不安の有無は価格系列からは")
    A("     判定できない。これらは別途の証拠が必要であり、本ツールは推測しない。")
    return "\n".join(out)


# ------------------------------------------------------------- self-test
def selftest():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))

    # RSI boundary behaviour
    rising = [100 * (1.01 ** i) for i in range(40)]
    falling = [100 * (0.99 ** i) for i in range(40)]
    check("RSI 単調上昇 = 100", abs(rsi_wilder(rising) - 100.0) < 1e-9, f"{rsi_wilder(rising):.6f}")
    check("RSI 単調下降 = 0", abs(rsi_wilder(falling) - 0.0) < 1e-9, f"{rsi_wilder(falling):.6f}")

    # RSI known-answer, derived analytically rather than assumed.
    # Wilder smoothing on alternating +/-1 has a two-phase fixed point:
    #   A = (13B + 1)/14 ,  B = 13A/14   =>   A = 14/27 , B = 13/27
    # Ending on a DOWN tick: avg_gain = 13/27, avg_loss = 14/27
    #   RS = 13/14  ->  RSI = 100 * 13/27 = 48.1481...
    # Ending on an UP tick, by symmetry:  RSI = 100 * 14/27 = 51.8518...
    # Note this is NOT 50: the last tick's phase biases Wilder's average.
    down_end = [100.0]
    for i in range(400):
        down_end.append(down_end[-1] + (1.0 if i % 2 == 0 else -1.0))
    up_end = [100.0]
    for i in range(400):
        up_end.append(up_end[-1] + (-1.0 if i % 2 == 0 else 1.0))
    exp_dn, exp_up = 100.0 * 13.0 / 27.0, 100.0 * 14.0 / 27.0
    check("RSI ジグザグ(下げ終い) = 100*13/27", abs(rsi_wilder(down_end) - exp_dn) < 1e-9,
          f"{rsi_wilder(down_end):.8f} vs {exp_dn:.8f}")
    check("RSI ジグザグ(上げ終い) = 100*14/27", abs(rsi_wilder(up_end) - exp_up) < 1e-9,
          f"{rsi_wilder(up_end):.8f} vs {exp_up:.8f}")

    # SMA
    check("SMA(3) of 1..10 = 9.0", abs(sma([float(i) for i in range(1, 11)], 3) - 9.0) < 1e-12)

    # ATR: constant range, no gaps -> ATR == range
    bars = [{"high": 11.0, "low": 9.0, "close": 10.0} for _ in range(40)]
    check("ATR 一定レンジ = 2.0", abs(atr_wilder(bars) - 2.0) < 1e-9, f"{atr_wilder(bars):.6f}")

    # Bollinger: constant series -> sd 0, bands collapse to mid
    flat = [5.0] * 30
    bb = bollinger(flat)
    check("BB 定数系列で幅0", abs(bb["sd"]) < 1e-12 and abs(bb["upper"] - bb["lower"]) < 1e-12)

    # Realized vol: zero variance -> 0
    check("実現ボラ 定数系列 = 0", abs(realized_vol([7.0] * 40, 20)) < 1e-12)

    # Max drawdown known answer
    dd = max_drawdown([100.0, 120.0, 60.0, 90.0, 130.0])
    check("最大DD = -50%", abs(dd["mdd"] + 0.5) < 1e-12, f"{dd['mdd']:.4f}")
    check("最大DD 回復検出", dd["recovered"] is True)
    check("最大DD peak/trough index", dd["peak_idx"] == 1 and dd["trough_idx"] == 2,
          f"peak={dd['peak_idx']} trough={dd['trough_idx']}")

    # Regression guard: the series makes a NEW high after the max-drawdown
    # trough. The reported peak must stay the pre-trough peak (index 1), not
    # the later high, and recovery must be measured against that peak.
    dd2 = max_drawdown([100.0, 200.0, 100.0, 150.0, 210.0, 205.0])
    check("最大DD 後日の新高値に引きずられない",
          dd2["peak_idx"] == 1 and dd2["trough_idx"] == 2
          and abs(dd2["mdd"] + 0.5) < 1e-12 and dd2["recover_idx"] == 4,
          f"peak={dd2['peak_idx']} trough={dd2['trough_idx']} "
          f"mdd={dd2['mdd']:.4f} recover={dd2['recover_idx']}")

    # End-to-end decomposition: a product built as EXACT 3x daily compounding
    # must make stage 3 == stage 4, while stage 2 (naive 3x) differs.
    idx = [100.0]
    for r in (0.01, -0.02, 0.015, -0.005, 0.02, -0.03, 0.01):
        idx.append(idx[-1] * (1 + r))
    prod = [50.0]
    for i in range(1, len(idx)):
        prod.append(prod[-1] * (1 + 3 * (idx[i] / idx[i - 1] - 1)))
    d = decompose(idx, prod)
    check("4段階分解 理論=実績（厳密3倍合成）", abs(d["theoretical"] - d["actual"]) < 1e-12,
          f"theo {d['theoretical']:.10f} vs actual {d['actual']:.10f}")
    check("4段階分解 単純3倍 ≠ 実績", abs(d["naive_kx"] - d["actual"]) > 1e-6,
          f"naive {d['naive_kx']:.6f} vs actual {d['actual']:.6f}")
    check("経路依存 + 費用差 = 実績 - 単純3倍",
          abs((d["path_effect"] + d["cost_effect"]) - (d["actual"] - d["naive_kx"])) < 1e-12)

    # MACD sanity: on a constant series MACD and signal are both 0
    m = macd([10.0] * 60)
    check("MACD 定数系列 = 0", abs(m["macd"]) < 1e-9 and abs(m["hist"]) < 1e-9)

    print("\n" + ("すべての自己検査に合格" if ok else "検査に失敗あり"))
    return 0 if ok else 1


TEMPLATE = """date,open,high,low,close,volume
2026-08-06,,,,,
2026-08-07,,,,,
2026-08-10,,,,,
"""


def main():
    p = argparse.ArgumentParser(description="Leveraged product analyzer")
    p.add_argument("--product", help="CSV of the product's daily OHLCV")
    p.add_argument("--index", help="CSV of the underlying index (close is enough)")
    p.add_argument("--name", default="PRODUCT")
    p.add_argument("--index-name", default=None)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--template", action="store_true")
    a = p.parse_args()

    if a.selftest:
        return selftest()
    if a.template:
        sys.stdout.write(TEMPLATE)
        return 0
    if not a.product:
        p.print_help()
        return 2

    try:
        bars, dropped = load_series(a.product)
        idx = None
        if a.index:
            idx, _ = load_series(a.index, require_ohlc=False)
    except (ValueError, OSError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    if dropped:
        print(f"(注意: 解析できず除外した行 {dropped}件。補間はしていない)\n")
    print(analyze(bars, a.name, idx, a.index_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
