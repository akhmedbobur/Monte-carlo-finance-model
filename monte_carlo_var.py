from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252
BLUE, ORANGE, RED = "#4C72B0", "#DD8452", "#C44E52"

#Data

def load_prices(tickers: list[str], years: int) -> pd.DataFrame:
    """Download adjusted daily close prices with yfinance."""
    import yfinance as yf

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers).dropna()
    if len(prices) < 2 * TRADING_DAYS * 0.9:
        raise ValueError(
            f"Only {len(prices)} rows of clean prices; need roughly 2+ years. "
            "Check the tickers or increase --years."
        )
    return prices


def synthetic_prices(tickers: list[str], years: int, seed: int = 7) -> pd.DataFrame:
    """Correlated GBM prices so the pipeline can be tested offline."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    corr = np.full((n, n), 0.45)
    np.fill_diagonal(corr, 1.0)
    vol = np.linspace(0.20, 0.35, n) / np.sqrt(TRADING_DAYS)
    mu = np.linspace(0.06, 0.12, n) / TRADING_DAYS
    days = int(years * TRADING_DAYS)
    z = rng.standard_normal((days, n)) @ np.linalg.cholesky(corr).T
    log_ret = mu + vol * z
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(log_ret, axis=0)), index=idx, columns=tickers)



def estimate_parameters(prices: pd.DataFrame) -> dict:
    """Daily log-return statistics (simulation inputs) plus annualised views."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu_d = log_ret.mean().to_numpy()
    vol_d = log_ret.std().to_numpy()
    cov_d = log_ret.cov().to_numpy()
    corr = log_ret.corr().to_numpy()
    summary = pd.DataFrame(
        {
            "ann_return": mu_d * TRADING_DAYS,
            "ann_vol": vol_d * np.sqrt(TRADING_DAYS),
        },
        index=prices.columns,
    )
    return dict(
        log_ret=log_ret, mu_d=mu_d, vol_d=vol_d, cov_d=cov_d, corr=corr, summary=summary
    )


def simulate_portfolio_paths(
    mu_d: np.ndarray,
    vol_d: np.ndarray,
    corr: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Simulate buy-and-hold portfolio return paths, shape (n_sims, horizon_days + 1).
    Column 0 is day 0 (return 0); the last column is the horizon return.

    z: independent N(0,1) draws, shape (n_sims, horizon_days, n_assets).
    The Cholesky factor L of the correlation matrix turns independent
    draws into correlated ones:  z_corr = z @ L.T
    """
    L = np.linalg.cholesky(corr)
    z_corr = z @ L.T                                  
    daily_log = mu_d + vol_d * z_corr                 
    cum_log = np.cumsum(daily_log, axis=1)            
    asset_simple = np.expm1(cum_log)                  
    port = asset_simple @ weights                    
    return np.hstack([np.zeros((port.shape[0], 1)), port])


def simulate_portfolio_returns(mu_d, vol_d, corr, weights, z) -> np.ndarray:
    """Horizon portfolio return for each simulation (last point of each path)."""
    return simulate_portfolio_paths(mu_d, vol_d, corr, weights, z)[:, -1]


def var_es(port_returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Both returned as positive loss fractions.
    VaR = loss at the (1 - confidence) quantile of returns.
    ES  = average loss in the tail beyond VaR.
    """
    losses = -port_returns
    var = float(np.quantile(losses, confidence))
    es = float(losses[losses >= var].mean())
    return var, es


def parametric_var(mu_d, cov_d, weights, horizon, confidence=0.95) -> float:
    """Closed-form VaR (normal log returns, approximate) as a sanity check on the simulation."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(confidence)
    mu_p = float(weights @ mu_d) * horizon
    sigma_p = float(np.sqrt(weights @ cov_d @ weights * horizon))
    return -float(np.expm1(mu_p - z * sigma_p))  # log-return quantile -> simple-return loss


def blend_correlation(corr: np.ndarray, lam: float) -> np.ndarray:
    """Move every off-diagonal correlation toward 1:  (1-lam)*C + lam*J."""
    n = corr.shape[0]
    out = (1 - lam) * corr + lam * np.ones((n, n))
    np.fill_diagonal(out, 1.0)
    return out


def avg_pairwise(corr: np.ndarray) -> float:
    n = corr.shape[0]
    return float((corr.sum() - n) / (n * (n - 1)))


def correlation_sensitivity(mu_d, vol_d, corr, weights, z, confidence) -> pd.DataFrame:
    """Same random draws (z) in every scenario, so differences come only from correlation."""
    rows = []
    for lam in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        c = blend_correlation(corr, lam)
        pr = simulate_portfolio_returns(mu_d, vol_d, c, weights, z)
        v, e = var_es(pr, confidence)
        rows.append({"avg_corr": avg_pairwise(c), "VaR": v, "ES": e})
    return pd.DataFrame(rows)



# Plots

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_distribution(port_returns, var, es, conf, horizon, path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(port_returns * 100, bins=80, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(-var * 100, color=ORANGE, ls="--", lw=2, label=f"{conf:.0%} VaR: -{var:.1%}")
    ax.axvline(-es * 100, color=RED, ls="-", lw=2, label=f"{conf:.0%} ES: -{es:.1%}")
    ax.set_title(
        f"In the worst {1 - conf:.0%} of months, losses average {es:.1%}", fontweight="bold"
    )
    ax.set_xlabel(f"Simulated {horizon}-day portfolio return (%)")
    ax.set_ylabel("Number of simulations")
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_paths(paths, var, es, conf, horizon, path, n_show=3000, seed=0):
    """Fan chart of all simulated paths plus the distribution of final outcomes."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    final = paths[:, -1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n_show, len(paths)), replace=False)
    sub = paths[idx] * 100
    days = np.arange(paths.shape[1])

    bg, fg = "#12163a", "#d8dcff"
    cmap = LinearSegmentedColormap.from_list("risk", ["#ff7b8a", "#b9a6ff", "#6fa8ff", "#8fe3ff"])
    lo, hi = np.percentile(final, [1, 99])
    colors = cmap(np.clip((sub[:, -1] / 100 - lo) / (hi - lo), 0, 1))
    colors[:, 3] = 0.22

    fig = plt.figure(figsize=(13, 7), dpi=150, facecolor=bg)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1], wspace=0.0)
    ax = fig.add_subplot(gs[0], facecolor=bg)
    axh = fig.add_subplot(gs[1], facecolor=bg, sharey=ax)

    segs = [np.column_stack([days, row]) for row in sub]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=0.7))
    ax.plot(days, np.median(paths, axis=0) * 100, color="white", lw=1.6, alpha=0.9, label="Median path")
    ax.axhline(-var * 100, color="#ff7b8a", lw=1.5)
    ax.axhline(0, color=fg, lw=0.6, alpha=0.4)
    ax.text(
        0.6, -var * 100 - 1.2, f"VaR {conf:.0%}  -{var:.1%}   |   ES  -{es:.1%}",
        color="#12163a", fontsize=12, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#ff9aa6", ec="none"),
    )
    ax.set_xlim(0, horizon)
    ax.set_xticks(sorted({0, 5, 10, 15, horizon}))
    ymax = max(abs(sub.min()), abs(sub.max()))
    ax.set_ylim(-ymax * 1.05, ymax * 1.05)
    ax.set_xlabel("Trading days", color=fg)
    ax.set_ylabel("Portfolio return (%)", color=fg)
    ax.set_title(
        f"{len(paths):,} simulated paths (showing {len(idx):,}): the worst {1 - conf:.0%} end below -{var:.1%}",
        color=fg, fontweight="bold", loc="left",
    )
    ax.legend(loc="upper left", frameon=False, labelcolor=fg)

    counts, edges = np.histogram(final * 100, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = np.where(centers <= -var * 100, "#ff7b8a", "#6fa8ff")
    axh.barh(centers, counts, height=np.diff(edges), color=bar_colors, edgecolor="none")
    axh.axhline(-var * 100, color="#ff7b8a", lw=1.5)
    axh.set_xticks([])
    plt.setp(axh.get_yticklabels(), visible=False)
    axh.tick_params(left=False)
    axh.set_xlabel("Final outcomes", color=fg)

    for a in (ax, axh):
        a.tick_params(colors=fg)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.grid(color=fg, alpha=0.08)
    fig.savefig(path, bbox_inches="tight", facecolor=bg)
    return fig


def plot_correlation(corr, labels, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Historical correlation of daily returns", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def plot_sensitivity(sens, baseline_corr, conf, path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(sens["avg_corr"], sens["VaR"] * 100, marker="o", color=ORANGE, lw=2, label="VaR")
    ax.plot(sens["avg_corr"], sens["ES"] * 100, marker="s", ls="--", color=RED, lw=2, label="ES")
    ax.axvline(baseline_corr, color="grey", ls=":", label=f"Estimated avg correlation ({baseline_corr:.2f})")
    ax.set_title("Higher correlation, higher tail risk: diversification fades", fontweight="bold")
    ax.set_xlabel("Average pairwise correlation")
    ax.set_ylabel(f"{conf:.0%} 1-month loss (%)")
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig

# Main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=["AAPL", "JPM", "XOM", "JNJ"])
    p.add_argument("--weights", nargs="+", type=float, help="Default: equal weight")
    p.add_argument("--years", type=int, default=3, help="Years of history (min 2)")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--horizon", type=int, default=21, help="Trading days (21 = 1 month)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--value", type=float, default=100_000, help="Portfolio value for $ figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="mc_var_output")
    p.add_argument("--synthetic", action="store_true", help="Offline demo with fake prices")
    p.add_argument("--show", action="store_true", help="Open the plot windows")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.show:
        plt.switch_backend("Agg")

    tickers = [t.upper() for t in args.tickers]
    n = len(tickers)
    weights = np.array(args.weights if args.weights else [1 / n] * n, dtype=float)
    if len(weights) != n:
        raise SystemExit("--weights must have one entry per ticker")
    if not np.isclose(weights.sum(), 1.0):
        print(f"Weights sum to {weights.sum():.3f}; normalising to 1.")
        weights = weights / weights.sum()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # Step 1
    prices = synthetic_prices(tickers, args.years) if args.synthetic else load_prices(tickers, args.years)
    print(f"\nData: {len(prices)} daily prices, {prices.index[0].date()} to {prices.index[-1].date()}")
    if args.synthetic:
        print("(synthetic data: results are for testing the pipeline only)")

    # Step 2
    est = estimate_parameters(prices)
    print("\nAnnualised estimates")
    print(est["summary"].round(3).to_string())
    print("\nCorrelation matrix")
    print(pd.DataFrame(est["corr"], index=tickers, columns=tickers).round(2).to_string())

    # Step 3
    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal((args.sims, args.horizon, n))
    paths = simulate_portfolio_paths(est["mu_d"], est["vol_d"], est["corr"], weights, z)
    port = paths[:, -1]

    # Step 4
    var, es = var_es(port, args.confidence)
    pvar = parametric_var(est["mu_d"], est["cov_d"], weights, args.horizon, args.confidence)
    # What you would get by skipping Cholesky (assets treated as independent)
    ind = simulate_portfolio_returns(est["mu_d"], est["vol_d"], np.eye(n), weights, z)
    var_ind, es_ind = var_es(ind, args.confidence)

    c = args.confidence
    print(f"\nResults: {args.sims:,} simulations, {args.horizon}-day horizon, ${args.value:,.0f} portfolio")
    print(f"  Expected return        : {port.mean():+.2%}")
    print(f"  {c:.0%} VaR                : {var:.2%}   (${var * args.value:,.0f})")
    print(f"  {c:.0%} Expected Shortfall : {es:.2%}   (${es * args.value:,.0f})")
    print(f"  Parametric VaR check   : {pvar:.2%}   (approximate; should land near the simulated VaR)")
    print(f"  VaR if correlation ignored (no Cholesky): {var_ind:.2%}  -> understated by {1 - var_ind / var:.0%}")

    # Step 5
    sens = correlation_sensitivity(est["mu_d"], est["vol_d"], est["corr"], weights, z, c)
    print("\nCorrelation sensitivity (same random draws in every row)")
    print(sens.assign(**{k: sens[k].map("{:.2%}".format) for k in ("VaR", "ES")})
          .assign(avg_corr=sens["avg_corr"].round(2)).to_string(index=False))
    sens.to_csv(out / "sensitivity.csv", index=False)

    # Plots
    plot_paths(paths, var, es, c, args.horizon, out / "paths.png")
    plot_distribution(port, var, es, c, args.horizon, out / "distribution.png")
    plot_correlation(est["corr"], tickers, out / "correlation.png")
    plot_sensitivity(sens, avg_pairwise(est["corr"]), c, out / "sensitivity.png")
    print(f"\nSaved charts (paths.png is the fan chart) and CSV to ./{out}/")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
