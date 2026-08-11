"""
Deterministic leveraged-ETF/ETN mathematics.

Every number produced here is derived analytically or by exact simulation from
stated assumptions. NOTHING here depends on live market data, so all of it is
reproducible and auditable independent of the blocked data sources.
"""

JPY = 1_000_000


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- 1. recovery
rule("1. DRAWDOWN -> REQUIRED RECOVERY   (exact: r = d/(1-d))")
print(f"{'drawdown':>10} | {'required gain':>14} | {'JPY1,000,000 becomes':>21} | {'loss':>12}")
print("-" * 68)
for d in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
    need = d / (1 - d)
    remain = JPY * (1 - d)
    print(f"{d*100:9.0f}% | {need*100:13.2f}% | {remain:20,.0f} | {JPY-remain:11,.0f}")


# ------------------------------------------------------- 2. volatility drag
rule("2. VOLATILITY DRAG OF A DAILY-RESET k-x FUND (continuous-time, pre-fee)")
print("""Underlying:  dS/S = mu*dt + sigma*dW
k-x daily-reset fund: dL/L = k*(dS/S)

  log-growth of underlying      = mu - sigma^2/2
  k * (log-growth of underlying)= k*mu - k*sigma^2/2
  log-growth of the k-x fund    = k*mu - k^2*sigma^2/2

  DRAG (fund vs k-x of underlying's compounded return)
      = (k^2 - k)/2 * sigma^2       ... per year
  For k = 3 :  (9-3)/2 = 3   ->   drag = 3 * sigma^2 per year
""")
print(f"{'index annualised vol':>21} | {'annual drag (k=3)':>18} | {'multiplicative factor/yr':>25}")
print("-" * 70)
import math
for s in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80):
    drag = 3 * s * s
    print(f"{s*100:20.0f}% | {drag*100:17.1f}% | {math.exp(-drag):24.4f}")
print("\nRead: at 40% index vol, a 3x fund loses ~48%/yr of log-return purely to")
print("variance, BEFORE fees and financing, even if the index ends flat.")


# ------------------------------------------- 3. flat-index path dependency
rule("3. PATH DEPENDENCY: INDEX ENDS EXACTLY FLAT, 3x FUND DOES NOT")
def three_x_path(daily_returns, k=3.0):
    v = 1.0
    for r in daily_returns:
        v *= (1 + k * r)
    return v

scenarios = {
    "+10% then -9.0909% (index flat)": [0.10, -1 / 11],
    "-10% then +11.1111% (index flat)": [-0.10, 1 / 9],
    "+5%/-4.7619% x 10 cycles (flat)": [0.05, -1 / 21] * 10,
    "+2%/-1.9608% x 50 cycles (flat)": [0.02, -1 / 51] * 50,
}
print(f"{'scenario':>34} | {'index':>8} | {'3x fund':>9} | {'3x shortfall':>13}")
print("-" * 74)
for name, rets in scenarios.items():
    idx = 1.0
    for r in rets:
        idx *= (1 + r)
    lev = three_x_path(rets)
    print(f"{name:>34} | {(idx-1)*100:7.3f}% | {(lev-1)*100:8.3f}% | {(lev-1)*100:12.3f}%")
print("\nThe index returns to its starting value in every row. The 3x product never does.")


# --------------------------------------- 4. trending market positive effect
rule("4. THE OTHER SIDE: SUSTAINED LOW-VOL TREND COMPOUNDS *ABOVE* 3x")
print(f"{'daily index move':>17} | {'days':>5} | {'index':>9} | {'3x simple':>10} | {'3x actual':>10}")
print("-" * 64)
for dr, n in ((0.004, 63), (0.004, 252), (0.008, 63), (-0.004, 63), (-0.008, 63)):
    idx = (1 + dr) ** n - 1
    lev = (1 + 3 * dr) ** n - 1
    print(f"{dr*100:16.2f}% | {n:5d} | {idx*100:8.2f}% | {idx*3*100:9.2f}% | {lev*100:9.2f}%")
print("\nA smooth uptrend makes the 3x product BEAT 3x the index cumulative return.")
print("A smooth downtrend makes it lose LESS than 3x. Both are compounding, not alpha.")


# ------------------------------------------------ 5. single-day wipeout math
rule("5. SINGLE-DAY WIPEOUT THRESHOLD FOR A 3x DAILY-RESET PRODUCT")
print("""A k-x daily product's value multiplier for one day is (1 + k*r).
It reaches zero when 1 + k*r = 0  ->  r = -1/k.

  k = 3  ->  r = -33.3333%   (index down one-third in a single session)
  k = 2  ->  r = -50%

Fees and financing make the true threshold slightly SHALLOWER than -1/3.
For GDXU this is not merely theoretical: the ETN's indicative note value is
contractually floored at zero, so a sufficiently large one-day index decline
is a permanent, total, non-recoverable loss even if the index rebounds the
next day. An ETF cannot go below zero either, but GDXU carries the added
issuer-credit layer described in the report.
""")
print(f"{'one-day index move':>19} | {'3x product move':>16} | {'JPY1,000,000 becomes':>21}")
print("-" * 62)
for r in (-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -1 / 3):
    lev = 3 * r
    val = max(0.0, JPY * (1 + lev))
    print(f"{r*100:18.2f}% | {lev*100:15.2f}% | {val:20,.0f}")


# ------------------------------------------------------ 6. JPY risk framing
rule("6. JPY 1,000,000 EXPOSURE — PRODUCT MOVE vs FX, SEPARATED")
print("Product-price scenarios (FX held constant):")
print(f"{'3x product move':>17} | {'JPY value':>13} | {'P/L':>13}")
print("-" * 48)
for m in (0.30, 0.20, 0.10, 0.05, -0.05, -0.10, -0.20, -0.30, -0.50):
    v = JPY * (1 + m)
    print(f"{m*100:16.0f}% | {v:12,.0f} | {v-JPY:+12,.0f}")

print("\nFX-only scenarios (product price held constant, JPY strengthens):")
print(f"{'USD/JPY move':>14} | {'JPY value':>13} | {'P/L':>13}")
print("-" * 45)
for fx in (-0.05, -0.10, -0.15):
    v = JPY * (1 + fx)
    print(f"{fx*100:13.0f}% | {v:12,.0f} | {v-JPY:+12,.0f}")

print("\nCombined (illustrative): a 3x product -30% WITH a 10% stronger JPY")
print(f"  JPY value = 1,000,000 * 0.70 * 0.90 = {JPY*0.70*0.90:,.0f}"
      f"  (P/L {JPY*0.70*0.90-JPY:+,.0f})")


# ------------------------------------- 7. ETN running-cost approximation
rule("7. ETN ANNUAL RUNNING-COST FORM (structure only — rates NOT verified)")
print("""Annualised cost run-rate, simplified:

    approx annual cost
        = Daily Investor Fee Rate
        + Daily Financing Factor x (official reference rate + Financing Spread)

The Daily Financing Charge itself accrues, per the prospectus form:

    prior-day Closing Indicative Note Value
        x Daily Financing Factor
        x Daily Financing Rate
        / 365
        x calendar days elapsed

CAVEAT: this run-rate ignores the variation of the indicative value,
intra-period compounding, and holiday day-count. The reference rate is
defined in the pricing supplement and must NOT be proxied with the Federal
Funds Rate or the US 10-year yield.

The actual fee rate, financing factor, spread and reference-rate definition
for GDXU COULD NOT BE RETRIEVED in this session — see the verification list.
Because the multiplier on a 3x note is applied to a leveraged notional, the
financing component is charged on roughly 2x the investor's equity, so the
all-in carry is materially larger than a 1x product's expense ratio.
""")

# ------------------------------------------------ 8. cost-of-carry erosion
rule("8. COST-OF-CARRY EROSION OVER TIME (illustrative rates, NOT GDXU's actual)")
print(f"{'all-in annual cost':>19} | {'21d':>8} | {'63d':>8} | {'126d':>8} | {'252d':>8}")
print("-" * 60)
for c in (0.0095, 0.02, 0.04, 0.06, 0.08, 0.10):
    row = [(1 - c) ** (d / 252) - 1 for d in (21, 63, 126, 252)]
    print(f"{c*100:18.2f}% | {row[0]*100:7.2f}% | {row[1]*100:7.2f}% | "
          f"{row[2]*100:7.2f}% | {row[3]*100:7.2f}%")
print("\nThese are ILLUSTRATIVE cost levels to show the shape of the erosion.")
print("They are NOT the verified fees of any of the four products.")


# --------------------------------------------- 9. combined survival math
rule("9. DRAG + COST COMBINED: BREAK-EVEN INDEX DRIFT FOR A 3x PRODUCT")
print("Required annualised index log-drift (mu) for the 3x product to end flat,")
print("solving  3*mu - 4.5*sigma^2 - cost = 0  ->  mu = (4.5*sigma^2 + cost)/3\n")
print(f"{'index vol':>10} | " + " | ".join(f"cost {c*100:>4.1f}%" for c in (0.01, 0.03, 0.05, 0.08)))
print("-" * 62)
for s in (0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
    cells = []
    for c in (0.01, 0.03, 0.05, 0.08):
        mu = (4.5 * s * s + c) / 3
        cells.append(f"{mu*100:9.2f}%")
    print(f"{s*100:9.0f}% | " + " | ".join(cells))
print("\nRead: at 50% index vol and 5% all-in cost, the underlying index must")
print("compound at ~39%/yr just for the 3x product to break even. This is the")
print("single most important number a buy-and-hold 3x holder can look at.")
