import typer

from .data import PriceField, fetch_price_series
from .model import ModelMode, PenaltySpec, detect_regimes_mean_gaussian
from .plotting import plot_price_with_regimes
from .report import build_report, dump_report

app = typer.Typer(help="Detect statistical regime shifts in financial time series.")

@app.command()
def analyze(
    ticker: str = typer.Argument(
        ..., 
        help = "Ticker symbol, e.g. AAPL, SPY"
    ),
    start: str | None = typer.Option(
        None, 
        help="Start date (YYYY-MM-DD)"
    ),
    end: str | None = typer.Option(
        None, 
        help="End date (YYYY-MM-DD)"
    ),
    field: PriceField = typer.Option(
        "adj_close",
        "--field",
        "-f",
        help="Price field to use: 'adj_close' (default) or 'close'"
    ),
    penalty: str = typer.Option(
        "bic",
        "--penalty",
        "-p",
        help="Penalty for number of regimes: 'bic', 'aic', or a positive float"
    ),
    min_seg_len: int = typer.Option(
        20,
        "--min-seg-len",
        help="Minimum regime length in trading days"
    ),
    model: str = typer.Option(
        "mean-var",
        "--model",
        help="Gaussian model: 'mean' (mean-shift only) or 'mean-var' (mean and variance together)"
    ),
    json_out: str | None = typer.Option(
        None, 
        "--json-out", 
        help="Optional path to write a JSON report (e.g. reports/aapl_regimes.json)"
    ),
    plot_out: str | None = typer.Option(
        None,
        "--plot-out",
        help="Optional path to write a PNG plot (e.g. plots/aapl_regimes.png)"
    )
) -> None:
    """
    Fetch price data and compute log returns and run statistical regime detection
    """
    # Fetch data
    try:
        series = fetch_price_series(ticker=ticker, start=start, end=end, field=field)
    except ValueError as exc:
        typer.echo(f"[regime-radar] Error: {exc}")
        raise typer.Exit(code=1)
    
    df = series.data
    returns = df["log_return"].values

    n = len(df)
    first_date = df.index[0].date()
    last_date = df.index[-1].date()

    mean_ret = df["log_return"].mean()
    vol_ret = df["log_return"].std()

    # Basic stats
    typer.echo(f"[regime-radar] Ticker: {series.ticker}")
    typer.echo(f"[regime-radar] Observations: {n}")
    typer.echo(f"[regime-radar] Date range: {first_date} -> {last_date}")
    typer.echo("")
    typer.echo("[regime-radar] Log-return stats (daily):")
    typer.echo(f"  mean      = {mean_ret:.6f}")
    typer.echo(f"  std dev   = {vol_ret:.6f}")
    typer.echo(f"  ann. mean ≈ {mean_ret * 252:.4f}")
    typer.echo(f"  ann. vol  ≈ {vol_ret * (252 ** 0.5):.4f}")
    typer.echo("")

    # Parse penalty for model
    penalty_spec: PenaltySpec
    try:
        penalty_spec = float(penalty)
        if penalty_spec <= 0:
            raise ValueError
    except ValueError:
        penalty_spec = penalty
    
    # Validate model mode
    model_mode: ModelMode
    m = model.lower()
    if m not in ("mean", "mean-var"):
        typer.echo("[regime-radar] Error: --model must be 'mean' or 'mean-var'")
        raise typer.Exit(code=1)
    model_mode = m
    
    # Run segmentation
    try:
        result = detect_regimes_mean_gaussian(
            returns=returns,
            penalty=penalty_spec,
            min_seg_len=min_seg_len,
            mode=model_mode
        )
    except ValueError as exc:
        typer.echo(f"[regime-radar] Regime detection failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("[regime-radar] Regime detection:")
    typer.echo(f"  penalty λ     = {result.penalty:.4f}")
    typer.echo(f"  # of regimes  = {len(result.regimes)}")
    typer.echo(f"  total cost    = {result.total_cost:.4f}")
    typer.echo("")

    # Print each regime w/ dates and stats within segment
    for idx, regime in enumerate(result.regimes, start=1):
        seg_df = df.iloc[regime.start : regime.end + 1]
        seg_start = seg_df.index[0].date()
        seg_end = seg_df.index[-1].date()
        seg_mean = seg_df["log_return"].mean()
        seg_vol = seg_df["log_return"].std()

        typer.echo(f" Regime {idx}:")
        typer.echo(f"   range       : {seg_start} -> {seg_end} "
                   f"(len = {len(seg_df)} trading days)")
        typer.echo(f"   mean (daily): {seg_mean:.6f}")
        typer.echo(f"   vol  (daily): {seg_vol:.6f}")
        typer.echo(f"   mean (ann.) : {seg_mean * 252:.4f}")
        typer.echo(f"   vol  (ann.) : {seg_vol * (252 ** 0.5):.4f}")
        typer.echo("")
    
    # (OPTIONAL) JSON report
    if json_out is not None:
        try:
            report = build_report(ticker=series.ticker, df=df, segmentation=result)
            dump_report(report, json_out)
            typer.echo(f"[regime-radar] JSON report written to {json_out}")
        except Exception as exc:
            typer.echo(f"[regime-radar] Failed to write JSON report: {exc}")
            raise typer.Exit(code=1)
        
    # (OPTIONAL) plot
    if plot_out is not None:
        try:
            title_suffix = f"{model_mode}, penalty={result.penalty:.3f}"
            plot_price_with_regimes(
                df=df,
                segmentation=result,
                ticker=series.ticker,
                path=plot_out,
                title_suffix=title_suffix
            )
            typer.echo(f"[regime-radar] Plot written to {plot_out}")
        except Exception as exc:
            typer.echo(f"[regime-radar] Failed to write plot: {exc}")
            raise typer.Exit(code=1)

def main() -> None:
    app()

if __name__ == "__main__":
    main()