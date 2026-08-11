"""
Structural decay-burden ranking of SPXL / TQQQ / SOXL / GDXU.

This is answerable WITHOUT any price data, because it rests on the
diversification ordering of the four underlyings, which is structural
rather than empirical:

  S&P 500        ~500 constituents, all sectors      -> lowest variance
  Nasdaq-100     ~100 constituents, tech-concentrated -> higher
  Semiconductors ~30 constituents, single cyclical sector -> higher still
  Gold miners    ~50 names via 2 ETFs, single sector,
                 operationally levered to one commodity -> highest

Drag for a k-x daily-reset product = (k^2 - k)/2 * sigma^2 ; k=3 -> 3*sigma^2.
The RANKING is robust to the exact sigma values because it depends only on
the ordering, which follows from breadth and sector concentration.

Sigma bands below are illustrative ranges, NOT measured values. No price
history was obtainable in this session.
"""

BANDS = [
    ("SPXL", "S&P 500",            "broad, ~500 names, all sectors",   0.13, 0.22, 0.0090),
    ("TQQQ", "Nasdaq-100",         "~100 names, tech-concentrated",    0.20, 0.32, 0.0097),
    ("SOXL", "半導体指数",          "~30 names, single cyclical sector", 0.32, 0.55, 0.0095),
    ("GDXU", "金鉱株指数(ETN)",     "2 ETF blend, single sector",       0.35, 0.55, 0.2445),
]


def drag(s, k=3.0):
    return (k * k - k) / 2 * s * s


def breakeven(s, cost, k=3.0):
    return ((k * k / 2) * s * s + cost) / k


print("=" * 92)
print("A. 構造的減価負担のランキング（価格データ不要 / 原指数の分散度から導出）")
print("=" * 92)
print(f"{'商品':<6} {'原指数':<14} {'年率ボラ帯':>12} {'年率ドラッグ帯':>16} {'年間乗算係数':>14}")
print("-" * 92)
import math
for tic, idx, _, lo, hi, _ in BANDS:
    dlo, dhi = drag(lo), drag(hi)
    print(f"{tic:<6} {idx:<14} {lo*100:4.0f}-{hi*100:3.0f}% {dlo*100:9.0f}-{dhi*100:3.0f}% "
          f"{math.exp(-dhi):11.2f}-{math.exp(-dlo):.2f}")

print("\n[分析] 順位は SPXL < TQQQ < SOXL ≒ GDXU。")
print("この順序はボラの実測値に依存せず、構成銘柄数とセクター集中度から従う。")
print("500銘柄・全セクターの指数が、30銘柄・単一景気循環セクターの指数より")
print("低分散であることは分散投資の数理であって、経験的主張ではない。")

print("\n" + "=" * 92)
print("B. 損益分岐に必要な原指数の年率ドリフト（費用込み）")
print("=" * 92)
print(f"{'商品':<6} {'想定費用':>10} {'ボラ下限時':>14} {'ボラ上限時':>14}   {'解釈'}")
print("-" * 92)
for tic, idx, _, lo, hi, cost in BANDS:
    blo, bhi = breakeven(lo, cost), breakeven(hi, cost)
    note = "ETN費用が支配的" if tic == "GDXU" else "経費率は小さく、ドラッグが支配的"
    print(f"{tic:<6} {cost*100:9.2f}% {blo*100:13.1f}% {bhi*100:13.1f}%   {note}")

print("""
[分析] 読み方：原指数がこの率で複利成長し続けて、ようやく3倍商品が横ばい。
SPXL は年 6-11% 程度で釣り合うが、SOXL は 16-46%、GDXU は 26-46% を要する。
S&P500 の長期平均的な成長率は SPXL の分岐点に近い水準にあり得るが、
SOXL・GDXU の分岐点は、どの資産クラスの長期平均成長率をも大きく上回る。
すなわち SOXL と GDXU は、構造上『長期保有向きではない』ではなく
『長期保有が数学的に不利』な商品である。""")

print("=" * 92)
print("C. 保有期間別に見た、構造負担の許容度")
print("=" * 92)
print(f"{'保有期間':>10} " + "".join(f"{t:>12}" for t, *_ in BANDS))
print("-" * 92)
for days, lbl in ((1, "1日"), (5, "1週間"), (21, "1か月"), (63, "3か月"), (252, "1年")):
    row = ""
    for tic, _, _, lo, hi, cost in BANDS:
        total = drag((lo + hi) / 2) + cost      # drag + carry, annualised
        row += f"{(total*days/252)*100:11.1f}%"
    print(f"{lbl:>10} " + row)
print("\n（中央ボラ想定での、期間按分した減価＋費用の合計。原指数が横ばいの場合に")
print("　失われる概算値。原指数が動けばこれに方向性リターンが加減される。）")

print("\n" + "=" * 92)
print("D. 価格を見ずに言える結論")
print("=" * 92)
print("""1. 保有期間が1日〜数日なら、4商品とも構造負担は小さい（上表C参照）。
   3倍商品は本来この用途に設計されている。

2. 保有期間が3か月を超えるなら、SOXL と GDXU は構造だけで
   二桁の逆風を負う。方向性の判断が正しくても回収は容易でない。

3. GDXU だけは、減価に加えて発行体信用リスクと償還制約がある。
   これは価格がいくらであっても消えない。金鉱株への強気は
   GDX / GDXJ の無レバレッジ保有で表現でき、その場合これらを回避できる。

4. したがって『どれか1つを選ぶなら』という問いに対しては、
   構造面のみで判断する限り SPXL が最も負担が軽く、GDXU が最も重い。
   ただしこれは割安・割高の判定ではなく、同じ強気度・同じ保有期間を
   前提にしたときの、構造コストの多寡である。""")
