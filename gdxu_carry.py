"""
GDXU carry-and-drag sensitivity.

PURPOSE: show the SHAPE and MAGNITUDE of GDXU's structural cost, and how much
the underlying index must compound merely to break even.

INPUT PROVENANCE — read before using any number below:
  * Daily Investor Fee 0.95%/yr, reference rate = Federal Reserve Bank Prime
    Loan Rate, Financing Spread history 2.25% -> 3.25% (eff. 2025-11-21)
    -> 5.00% (eff. 2026-02-06): SEARCH-SNIPPET DERIVED, NOT VERIFIED against
    the pricing supplement. The issuer's own documents were unreachable.
  * Daily Financing Factor: NOT RETRIEVED. Modelled at 2.0 (the conventional
    borrowed-notional multiple for a 3x note) with a 3.0 sensitivity case.
  * US Prime Rate: NOT RETRIEVED. Modelled from the fed funds target range
    3.50-3.75% using the conventional upper-bound + 300bp convention -> 6.75%,
    with +/- cases. THIS IS AN ASSUMPTION, NOT A VERIFIED RATE.
  * Index volatility: NOT MEASURED. No price history was obtainable.

Nothing here is a verified statement about GDXU's actual cost today.
It is a structural sensitivity analysis under explicitly stated assumptions.
"""

FEE = 0.0095  # Daily Investor Fee, per annum


def annual_cost(prime, spread, factor):
    return FEE + factor * (prime + spread)


def breakeven_drift(sigma, cost, k=3.0):
    """Solve k*mu - (k^2/2)*sigma^2 - cost = 0 for mu."""
    return ((k * k / 2.0) * sigma * sigma + cost) / k


print("=" * 78)
print("A. ANNUAL RUNNING COST vs FINANCING SPREAD  (factor=2.0, prime=6.75%)")
print("=" * 78)
print("   approx annual cost = 0.95% + factor x (prime + spread)\n")
print(f"{'spread':>8} | {'effective from':>16} | {'annual cost':>12} | {'63-day cost':>12}")
print("-" * 58)
hist = [(0.0225, "pre 2025-11-21"), (0.0325, "2025-11-21"), (0.0500, "2026-02-06")]
for sp, eff in hist:
    c = annual_cost(0.0675, sp, 2.0)
    c63 = (1 - c) ** (63 / 252) - 1
    print(f"{sp*100:7.2f}% | {eff:>16} | {c*100:11.2f}% | {c63*100:11.2f}%")

print("\nThe spread was raised TWICE inside three months. Under these modelling")
print("assumptions that lifted the annual running cost by roughly 5.5 points.")

print("\n" + "=" * 78)
print("B. SENSITIVITY OF ANNUAL COST TO THE TWO UNVERIFIED INPUTS")
print("=" * 78)
print(f"{'prime':>7} |" + "".join(f"  factor={f:.1f}" for f in (1.0, 2.0, 3.0)))
print("-" * 44)
for prime in (0.0550, 0.0675, 0.0800):
    row = "".join(f"{annual_cost(prime, 0.05, f)*100:10.2f}%" for f in (1.0, 2.0, 3.0))
    print(f"{prime*100:6.2f}% |" + row)
print("\nEven the mildest cell (factor=1.0, prime=5.50%) exceeds 11%/yr.")
print("The Daily Financing Factor is the single largest unknown: it roughly")
print("doubles or triples the carry. It MUST be read off the pricing supplement.")

print("\n" + "=" * 78)
print("C. BREAK-EVEN: ANNUAL INDEX DRIFT NEEDED FOR GDXU TO END FLAT")
print("=" * 78)
print("   mu = (4.5*sigma^2 + cost) / 3      [k=3, continuous approximation]\n")
costs = [
    ("spread 2.25% (old)", annual_cost(0.0675, 0.0225, 2.0)),
    ("spread 5.00% (now)", annual_cost(0.0675, 0.0500, 2.0)),
]
print(f"{'index vol':>10} | " + " | ".join(f"{n:>19}" for n, _ in costs))
print("-" * 56)
for s in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
    cells = " | ".join(f"{breakeven_drift(s, c)*100:18.1f}%" for _, c in costs)
    print(f"{s*100:9.0f}% | {cells}")

print("\nGold-miner equity indices have historically been high-volatility.")
print("At 40% index vol and the current modelled cost, the underlying index must")
print("compound at roughly 32%/yr for GDXU merely to return zero.")
print("NOTE: the vol input is NOT measured here — no price history was obtainable.")

print("\n" + "=" * 78)
print("D. WHAT THE CARRY ALONE COSTS A FLAT-INDEX HOLDER (factor=2, prime=6.75%)")
print("=" * 78)
print("Index perfectly flat, volatility ignored — pure contractual carry on JPY1,000,000\n")
print(f"{'holding period':>16} | {'spread 2.25%':>14} | {'spread 5.00%':>14}")
print("-" * 50)
c_old = annual_cost(0.0675, 0.0225, 2.0)
c_new = annual_cost(0.0675, 0.0500, 2.0)
for days, lbl in ((1, "1 day"), (5, "5 days"), (21, "21 days"),
                  (63, "63 days"), (126, "126 days"), (252, "252 days")):
    a = (1 - c_old) ** (days / 252) - 1
    b = (1 - c_new) ** (days / 252) - 1
    print(f"{lbl:>16} | {1_000_000*a:13,.0f} | {1_000_000*b:13,.0f}")

print("\n" + "=" * 78)
print("E. THE COST STACK — WHY GDXU IS NOT 'GDX WITH 3x'")
print("=" * 78)
print("""Layer 1  GDX and GDXJ each charge their own management fee, which is already
         embedded in the index level the note references.
Layer 2  The note's Daily Investor Fee (0.95%/yr modelled).
Layer 3  The Daily Financing Charge on the borrowed notional (the dominant
         layer under every assumption tested above).
Layer 4  A 0.125% Redemption Fee, and a 25,000-note minimum redemption that
         puts primary-market redemption out of reach for retail holders,
         leaving the secondary market as the only practical exit.
Layer 5  BMO senior unsecured credit risk. There is no portfolio, no NAV and
         no segregated collateral behind the note — only the issuer's promise.

Layers 1 and 2 are small. Layer 3 dominates and is the one that changed twice
in three months. Layer 4 determines what an exit actually costs in stress.
Layer 5 is not a cost but a tail risk that no amount of correct market timing
on gold can hedge away.""")
