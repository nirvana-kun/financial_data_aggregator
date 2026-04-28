"""
Financial Data Aggregator
--------------------------
Pulls, normalises, and analyses financial data from Yahoo Finance
across multiple assets and timeframes. Computes risk/return metrics,
correlations, and outputs clean DataFrames ready for further analysis.

This feeds directly into your portfolio analyzer and fieldwave quant track.

Usage:
    agg = FinancialAggregator()
    data = agg.fetch(["AAPL", "MSFT", "NVDA", "BTC-USD"], period="1y")
    print(data.summary())
    data.plot_correlation()
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Run: pip install yfinance")

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.053    # ~5.3% annualised (US 3-month T-bill, update as needed)
TRADING_DAYS   = 252


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AssetMetrics:
    ticker: str
    total_return_pct: float
    annualised_return_pct: float
    annualised_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    beta: Optional[float] = None     # vs SPY, if available
    var_95_pct: float = 0.0          # Value at Risk (95%, daily)

    def __str__(self) -> str:
        return (
            f"  {self.ticker:<10}  "
            f"Return: {self.annualised_return_pct:+.1f}%  "
            f"Vol: {self.annualised_volatility_pct:.1f}%  "
            f"Sharpe: {self.sharpe_ratio:.2f}  "
            f"MaxDD: {self.max_drawdown_pct:.1f}%  "
            f"VaR95: {self.var_95_pct:.1f}%"
        )


@dataclass
class AggregatedData:
    tickers: list[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    metrics: list[AssetMetrics]
    correlation_matrix: pd.DataFrame
    period: str

    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"  FINANCIAL DATA SUMMARY  |  {len(self.tickers)} assets  |  Period: {self.period}",
            f"  Date range: {self.prices.index[0].date()} → {self.prices.index[-1].date()}",
            f"{'='*70}",
            f"  {'Ticker':<10} {'Ann.Return':>10} {'Volatility':>10} {'Sharpe':>7} {'MaxDD':>8} {'VaR95':>7}",
            f"  {'-'*60}",
        ]
        for m in sorted(self.metrics, key=lambda x: -x.sharpe_ratio):
            lines.append(
                f"  {m.ticker:<10} {m.annualised_return_pct:>+9.1f}% "
                f"{m.annualised_volatility_pct:>9.1f}% "
                f"{m.sharpe_ratio:>7.2f} "
                f"{m.max_drawdown_pct:>7.1f}% "
                f"{m.var_95_pct:>6.1f}%"
            )

        lines.append(f"\n  {'='*70}")
        lines.append("  TOP CORRELATIONS:")
        corr = self.correlation_matrix
        pairs = [
            (corr.iloc[i, j], corr.columns[i], corr.columns[j])
            for i in range(len(corr))
            for j in range(i + 1, len(corr))
        ]
        for r, a, b in sorted(pairs, key=lambda x: -abs(x[0]))[:5]:
            bar = "▓" * int(abs(r) * 10) + "░" * (10 - int(abs(r) * 10))
            sign = "+" if r > 0 else "-"
            lines.append(f"  {a} ↔ {b:<10} [{bar}] {sign}{abs(r):.2f}")

        return "\n".join(lines)


# ── Core aggregator ────────────────────────────────────────────────────────────

class FinancialAggregator:

    def fetch(
        self,
        tickers: list[str],
        period: str = "1y",        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: str = "1d",     # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
        benchmark: str = "SPY",
    ) -> AggregatedData:
        """Fetch price data and compute all metrics."""

        # Fetch prices
        all_tickers = list(set(tickers + [benchmark]))
        print(f"Fetching data for: {', '.join(tickers)}  (+ {benchmark} as benchmark)...")

        raw = yf.download(all_tickers, period=period, interval=interval,
                          auto_adjust=True, progress=False)

        if isinstance(raw.columns, pd.MultiIndex):
            prices_all = raw["Close"]
        else:
            prices_all = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

        prices_all = prices_all.dropna(how="all")

        # Separate benchmark
        benchmark_returns = None
        if benchmark in prices_all.columns:
            bm_prices = prices_all[benchmark].dropna()
            benchmark_returns = bm_prices.pct_change().dropna()

        # Keep only requested tickers with sufficient data
        available = [t for t in tickers if t in prices_all.columns]
        missing = [t for t in tickers if t not in prices_all.columns]
        if missing:
            print(f"  Warning: could not fetch {missing}")

        prices = prices_all[available].dropna(how="all")
        returns = prices.pct_change().dropna()

        # Drop any ticker that came back with empty/insufficient data
        valid = [t for t in available if prices[t].dropna().shape[0] > 5]
        skipped = [t for t in available if t not in valid]
        if skipped:
            print(f"  Warning: skipping {skipped} — insufficient data (download may have failed, try again)")
        available = valid
        prices = prices[available]
        returns = returns[available]

        # Compute metrics per asset
        metrics = []
        for ticker in available:
            m = self._compute_metrics(
                ticker, prices[ticker].dropna(), returns[ticker].dropna(), benchmark_returns
            )
            metrics.append(m)

        # Correlation matrix
        corr = returns.corr()

        print(f"  Done. {len(available)} assets, {len(prices)} trading days.\n")

        return AggregatedData(
            tickers=available,
            prices=prices,
            returns=returns,
            metrics=metrics,
            correlation_matrix=corr,
            period=period,
        )

    def _compute_metrics(
        self,
        ticker: str,
        prices: pd.Series,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
    ) -> AssetMetrics:

        n = len(returns)
        if n == 0 or len(prices) == 0:
            raise ValueError(f"No data for {ticker}")
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        ann_return = (1 + total_return) ** (TRADING_DAYS / n) - 1
        ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # VaR 95%
        var_95 = float(np.percentile(returns, 5)) * 100

        # Beta vs benchmark
        beta = None
        if benchmark_returns is not None:
            aligned = returns.align(benchmark_returns, join="inner")
            if len(aligned[0]) > 10:
                cov = np.cov(aligned[0], aligned[1])
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else None

        return AssetMetrics(
            ticker=ticker,
            total_return_pct=total_return * 100,
            annualised_return_pct=ann_return * 100,
            annualised_volatility_pct=ann_vol * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar,
            skewness=float(returns.skew()),
            kurtosis=float(returns.kurtosis()),
            beta=beta,
            var_95_pct=var_95,
        )

    # ── Visualisation ──────────────────────────────────────────────────────────

    def plot(self, data: AggregatedData, save_path: Optional[str] = None) -> None:
        """4-panel dashboard: price, returns, drawdown, correlation."""

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Financial Dashboard — {data.period}", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Normalised price (rebased to 100)
        ax1 = fig.add_subplot(gs[0, 0])
        rebased = (data.prices / data.prices.iloc[0]) * 100
        rebased.plot(ax=ax1)
        ax1.set_title("Normalised Price (Base=100)")
        ax1.set_ylabel("Index Value")
        ax1.legend(fontsize=7)
        ax1.axhline(100, color="gray", linestyle="--", linewidth=0.5)

        # 2. Rolling 30-day volatility
        ax2 = fig.add_subplot(gs[0, 1])
        rolling_vol = data.returns.rolling(30).std() * np.sqrt(TRADING_DAYS) * 100
        rolling_vol.plot(ax=ax2)
        ax2.set_title("Rolling 30-Day Volatility (Annualised)")
        ax2.set_ylabel("Volatility (%)")
        ax2.legend(fontsize=7)

        # 3. Drawdown chart
        ax3 = fig.add_subplot(gs[1, 0])
        for ticker in data.tickers:
            cum = (1 + data.returns[ticker]).cumprod()
            roll_max = cum.expanding().max()
            dd = (cum - roll_max) / roll_max * 100
            ax3.plot(dd.index, dd.values, label=ticker, alpha=0.8)
        ax3.fill_between(dd.index, dd.values, 0, alpha=0.1)
        ax3.set_title("Drawdown (%)")
        ax3.set_ylabel("Drawdown (%)")
        ax3.legend(fontsize=7)

        # 4. Correlation heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        corr = data.correlation_matrix
        im = ax4.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax4.set_xticks(range(len(corr)))
        ax4.set_yticks(range(len(corr)))
        ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax4.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax4.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color="black")
        plt.colorbar(im, ax=ax4)
        ax4.set_title("Correlation Matrix")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {save_path}")
        plt.show()

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_returns_csv(self, data: AggregatedData, filepath: str) -> None:
        data.returns.to_csv(filepath)
        print(f"Returns exported to {filepath}")

    def export_metrics_csv(self, data: AggregatedData, filepath: str) -> None:
        rows = []
        for m in data.metrics:
            rows.append({
                "ticker": m.ticker,
                "total_return_pct": m.total_return_pct,
                "annualised_return_pct": m.annualised_return_pct,
                "annualised_volatility_pct": m.annualised_volatility_pct,
                "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown_pct": m.max_drawdown_pct,
                "calmar_ratio": m.calmar_ratio,
                "beta": m.beta,
                "var_95_pct": m.var_95_pct,
                "skewness": m.skewness,
                "kurtosis": m.kurtosis,
            })
        pd.DataFrame(rows).to_csv(filepath, index=False)
        print(f"Metrics exported to {filepath}")


# ── Sector presets ─────────────────────────────────────────────────────────────

WATCHLISTS = {
    "mag7":      ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "quant_etf": ["SPY", "QQQ", "IWM", "GLD", "TLT", "VXX"],
    "crypto":    ["BTC-USD", "ETH-USD", "SOL-USD"],
    "financials": ["JPM", "GS", "MS", "BAC", "C"],
    "ai_plays":  ["NVDA", "AMD", "MSFT", "GOOGL", "PLTR", "AI"],
}


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agg = FinancialAggregator()

    tickers = WATCHLISTS["mag7"]
    data = agg.fetch(tickers, period="1y")

    print(data.summary())

    # Export
    agg.export_metrics_csv(data, "mag7_metrics.csv")
    agg.export_returns_csv(data, "mag7_returns.csv")

    # Plot (comment out if no display)
    # agg.plot(data, save_path="mag7_dashboard.png")
