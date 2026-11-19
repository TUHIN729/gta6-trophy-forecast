"""GTA VI trophy rarity forecaster (Python 3.13, standard library only).

Model: earners and players are forecast separately, then combined.

  rate_i(target) = 100 * (E_exist_i + E_new_i) / N_target

* Players N(t): log N is fit against log(1 + hours since release,
  2026-11-19T00:00Z) -- a power law, so growth slows as launch week goes
  on -- and extrapolated to the target.  An exponential fit is tried too
  and the better log-space fit wins.  When the history shows no growth at
  all (launch night: the page sat at 638 players for hours), the fit is
  uninformative, so a launch-phase prior is used instead: N grows like
  (1 + t) from the first read, i.e. the page's next refresh catches up
  several hours of player growth at once.  Exophase refreshes a game's
  counts in batches, hours apart, not continuously.
* New-cohort earn rate: when players_tracked jumps, the newly tracked
  cohort's earn rate for trophy i is implied by the counts,
  e_i = (r_new*N_new - r_old*N_old) / (N_new - N_old).  The per-trophy
  ratio k_i = e_i / r_old is pooled over the refresh events in the
  history (median) and shrunk toward a prior from the "m-model"
  e_i = 100 * (1 - (1 - r_i/100)^m).  With no refresh event in the
  history, m = 0.07: a new cohort earns a typical trophy at ~7% of the
  rate the tracked players show (launch-night aggregate: the refresh
  added 3,230 players who between them held ~103 non-first trophies,
  while the 100% trophy stayed at 100%, which the m-model reproduces
  exactly).  m is re-estimated from the history's own refresh events
  whenever they exist, from the trophies_earned totals.
* Progression: the existing cohort only ever gains trophies, so its
  rate drifts up on the logit scale.  The drift is estimated per trophy
  from stretches of history where players_tracked is flat (so the move
  is pure progression, uncontaminated by dilution), median-pooled and
  shrunk toward zero, and capped.
* Spreads: quantiles are normal on the logit scale around the median,
  with sigma combining a floor, a random-walk term growing with the
  horizon, a dilution term growing with log(N_target / N_last), and a
  bonus for rare trophies.  The 100% trophy gets a tight spread.

Everything is deterministic: no randomness, no network, no time source
other than the case file itself.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Launch-phase prior for the new-cohort earn fraction (see docstring).
M_PRIOR = 0.07


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    x = max(-40.0, min(40.0, x))
    return 100.0 / (1.0 + math.exp(-x))


def round4(x):
    """Round to 4 decimals, half up, as the starter does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Acklam's approximation to the standard normal quantile."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("level out of (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s):
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def median(vals):
    if not vals:
        return None
    v = sorted(vals)
    n = len(v)
    mid = n // 2
    if n % 2:
        return v[mid]
    return 0.5 * (v[mid - 1] + v[mid])


def ols(xs, ys):
    """Least squares y = a + b*x; returns (a, b, r2) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    return (a, b, r2)


def fit_players(times_h, ns, t_target):
    """Forecast players_tracked at the target; returns (n_target, growth)."""
    n_last = ns[-1]
    n_first = ns[0]
    if len(ns) >= 3 and max(ns) / max(min(ns), 1) >= 1.02:
        # Power law in log(1 + t): growth slows over launch week.
        xs_pow = [math.log(max(1.0 + t, 0.2)) for t in times_h]
        ys = [math.log(max(n, 1)) for n in ns]
        fit_pow = ols(xs_pow, ys)
        # Exponential in t, for comparison.
        fit_exp = ols(list(times_h), ys)
        best = None
        if fit_pow is not None:
            best = ("pow", fit_pow)
        if fit_exp is not None and (best is None or fit_exp[2] > best[1][2]):
            best = ("exp", fit_exp)
        if best is not None:
            kind, (a, b, _r2) = best
            if kind == "pow":
                pred = math.exp(a + b * math.log(max(1.0 + t_target, 0.2)))
            else:
                pred = math.exp(a + b * t_target)
            # A fitted slope must not predict a shrinking player base,
            # and a 4.5-day extrapolation is capped at 8x the last read.
            pred = max(n_last, min(pred, n_last * 8.0))
            return pred, pred / max(n_last, 1)
    # Flat history: the page has not refreshed yet.  Launch-phase prior:
    # the page's next refresh catches up growth like (1 + t) from the
    # first read, so the target's page shows that whole catch-up.  The
    # catch-up is capped at 8x the last read, the same cap as the fitted
    # branch, so a long flat stretch cannot imply runaway growth.
    t_first = times_h[0]
    num = max(1.0 + t_target, 0.2)
    den = max(1.0 + t_first, 0.2)
    pred = n_first * (num / den)
    pred = max(n_last, min(pred, n_last * 8.0))
    return pred, pred / max(n_last, 1)


def m_model_e(rate_pct, m):
    """New-cohort earn rate under the m-model (percent)."""
    if rate_pct >= 99.99:
        return 100.0
    if rate_pct <= 0.0:
        return 0.0
    return 100.0 * (1.0 - (1.0 - rate_pct / 100.0) ** m)


def solve_m(rates_prev, sum_e):
    """m whose m-model total new-cohort earn equals the observed sum."""
    if sum_e <= 0:
        return 1e-4
    lo, hi = 1e-4, 10.0

    def total(m):
        return sum(m_model_e(r, m) for r in rates_prev if r is not None)

    if total(hi) <= sum_e:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if total(mid) < sum_e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    n_tr = len(trophies)
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_h = horizon_min / 60.0
    horizon_days = horizon_min / 1440.0

    target_dt = parse_time(case.get("target"))
    if target_dt is None and history:
        last_dt = parse_time(history[-1].get("at"))
        target_dt = last_dt
    t_target = hours_since_release(target_dt) if target_dt else 0.0

    # --- history series -------------------------------------------------
    times_h = []
    ns = []
    earned = []
    rates_hist = []
    for read in history:
        dt = parse_time(read.get("at"))
        if dt is None:
            dt = RELEASE
        times_h.append(hours_since_release(dt))
        n = read.get("players_tracked")
        ns.append(float(n) if n else 0.0)
        e = read.get("trophies_earned")
        earned.append(float(e) if e is not None else None)
        rates_hist.append(list(read.get("rates") or []))
    if not history:
        times_h = [0.0]
        ns = [638.0]
        earned = [None]
        rates_hist = [[None] * n_tr]
    n_last = ns[-1] if ns[-1] > 0 else 638.0
    n_first = ns[0] if ns[0] > 0 else n_last

    # Last non-null rate per trophy.
    last_rate = []
    for i in range(n_tr):
        r = None
        for row in reversed(rates_hist):
            if i < len(row) and row[i] is not None:
                r = float(row[i])
                break
        last_rate.append(r if r is not None else 1.0)

    # --- players at target ----------------------------------------------
    n_target, growth = fit_players(times_h, ns, t_target)
    log_growth = math.log(max(growth, 1.0))

    # --- refresh events: implied new-cohort earn rates ------------------
    # k_obs[i]: observed k = e_i / r_old values for trophy i.
    k_obs = [[] for _ in range(n_tr)]
    m_events = []
    for j in range(1, len(history)):
        n_prev, n_cur = ns[j - 1], ns[j]
        if n_prev <= 0 or n_cur <= n_prev * 1.02:
            continue
        dn = n_cur - n_prev
        prev_rates = rates_hist[j - 1]
        cur_rates = rates_hist[j]
        # Observed total new-cohort earn (percent points), from the
        # trophies_earned totals when available, else from the rates.
        if earned[j] is not None and earned[j - 1] is not None:
            sum_e = 100.0 * (earned[j] - earned[j - 1]) / dn
        else:
            sum_e = 0.0
            for i in range(n_tr):
                if i < len(prev_rates) and i < len(cur_rates) \
                        and prev_rates[i] is not None and cur_rates[i] is not None:
                    sum_e += (cur_rates[i] * n_cur - prev_rates[i] * n_prev) / dn
        m_events.append(solve_m(
            [r for r in prev_rates if r is not None], sum_e))
        for i in range(n_tr):
            if i < len(prev_rates) and i < len(cur_rates) \
                    and prev_rates[i] is not None and cur_rates[i] is not None:
                e_i = (cur_rates[i] * n_cur - prev_rates[i] * n_prev) / dn
                e_i = max(0.0, min(100.0, e_i))
                r_old = float(prev_rates[i])
                if r_old > 0.05:
                    k_obs[i].append(e_i / r_old)
    m_med = median(m_events) if m_events else M_PRIOR
    m_med = max(0.005, min(2.0, m_med))

    # Per-trophy new-cohort fraction k_i, shrunk toward the m-model prior.
    k_val = []
    for i in range(n_tr):
        r0 = last_rate[i]
        e_prior = m_model_e(r0, m_med)
        k_prior = (e_prior / r0) if r0 > 0.05 else m_med
        obs = k_obs[i]
        if obs:
            k_hat = median(obs)
            k = (len(obs) * k_hat + 2.0 * k_prior) / (len(obs) + 2.0)
        else:
            k = k_prior
        k_val.append(max(0.0, min(5.0, k)))

    # --- progression slope (logit/hour) from flat-N stretches -----------
    slope_obs = [[] for _ in range(n_tr)]
    for j in range(1, len(history)):
        n_prev, n_cur = ns[j - 1], ns[j]
        if n_prev <= 0 or abs(n_cur - n_prev) / n_prev > 0.01:
            continue
        dt = times_h[j] - times_h[j - 1]
        if dt <= 0.01:
            continue
        prev_rates = rates_hist[j - 1]
        cur_rates = rates_hist[j]
        for i in range(n_tr):
            if i < len(prev_rates) and i < len(cur_rates) \
                    and prev_rates[i] is not None and cur_rates[i] is not None:
                d = (logit(float(cur_rates[i])) - logit(float(prev_rates[i]))) / dt
                if abs(d) < 1.0:
                    slope_obs[i].append(d)
    tier_slope = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = []
        for i, tr in enumerate(trophies):
            if tr.get("tier") == tier:
                vals.extend(slope_obs[i])
        tier_slope[tier] = median(vals) if vals else 0.0
    slope = []
    for i, tr in enumerate(trophies):
        obs = slope_obs[i]
        if obs:
            s_med = median(obs)
            s = (len(obs) / (len(obs) + 3.0)) * s_med
        else:
            s = 0.5 * tier_slope.get(tr.get("tier"), 0.0)
        slope.append(max(-0.03, min(0.03, s)))

    # --- quantiles --------------------------------------------------------
    levels = case.get("levels") or LEVELS
    zs = []
    for lv in levels:
        try:
            zs.append(norm_ppf(float(lv)))
        except ValueError:
            zs.append(0.0)
    if len(zs) != 5:
        zs = list(Z)

    rows = []
    for i, tr in enumerate(trophies):
        r0 = last_rate[i]
        if r0 >= 99.99:
            med = 99.99
            sigma = 0.30
        else:
            # Existing cohort: progression only (its rate never falls).
            move = slope[i] * horizon_h
            move = max(-0.2, min(2.0, move))
            r_exist = inv_logit(logit(r0) + move)
            e_exist = n_last * r_exist / 100.0
            # New cohort: arrives at k_i times the current rate.
            e_new_rate = k_val[i] * r0
            e_new = (e_new_rate / 100.0) * max(n_target - n_last, 0.0)
            med = 100.0 * (e_exist + e_new) / max(n_target, 1.0)
            med = clamp_pct(med)
            med = min(max(med, r0 * 0.02), r0 * 25.0)
            med = clamp_pct(med)
            # Spread: floor + horizon random walk + dilution uncertainty
            # + a bonus for rare trophies.
            rare = 0.25 if med < 1.0 else (0.15 if med < 5.0 else 0.0)
            sigma = math.sqrt(0.45 ** 2 + 0.25 * horizon_days
                              + 0.30 * log_growth ** 2 + rare ** 2)
            sigma = max(0.6, min(2.0, sigma))
            if med >= 90.0:
                sigma = min(sigma, 0.6)
        lg = logit(med)
        qs = []
        prev = -1.0
        for z in zs:
            q = round4(clamp_pct(inv_logit(lg + z * sigma)))
            q = max(q, prev)
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
