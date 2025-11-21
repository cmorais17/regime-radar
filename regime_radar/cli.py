import typer

from .data import PriceField, fetch_price_series

app = typer.Typer(help="Detect statistical regime shifts in financial time series.")

@app.command()
def analyze(
    ticker: str = typer.Argument(..., help = "Ticker symbol, e.g. AAPL, SPY"),
    start: str | None = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date (YYYY-MM-DD)"),
    field: PriceField = typer.Option(
        "adj_close",
        "--field",
        "-f",
        help="Price field to use: 'adj_close' (default) or 'close'"
    )
) -> None:
    """
    Fetch price data and compute log returns

    Placeholder command for the rest for now
    """
    try:
        series = fetch_price_series(ticker=ticker, start=start, end=end, field=field)
    except ValueError as exc:
        typer.echo(f"[regime-radar] Error: {exc}")
        raise typer.Exit(code=1)
    
    df = series.data

    n = len(df)
    first_date = df.index[0].date()
    last_date = df.index[-1].date()

    mean_ret = df["log_return"].mean()
    vol_ret = df["log_return"].std()

    typer.echo(f"[regime-radar] Ticker: {series.ticker}")
    typer.echo(f"[regime-radar] Observations: {n}")
    typer.echo(f"[regime-radar] Date range: {first_date} -> {last_date}")
    typer.echo("")
    typer.echo("[regime-radar] Log-return stats (daily):")
    typer.echo(f"  mean      = {mean_ret:.6f}")
    typer.echo(f"  std dev   = {vol_ret:.6f}")
    typer.echo(f"  ann. mean ≈ {mean_ret * 252:.4f}")
    typer.echo(f"  ann. vol  ≈ {vol_ret * (252 ** 0.5):.4f}")

def main() -> None:
    app()

if __name__ == "__main__":
    main()