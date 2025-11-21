import typer

app = typer.Typer(help="Detect statistical regime shifts in financial time series.")

@app.command()
def analyze(
    ticker: str = typer.Argument(..., help = "Ticker symbol, e.g. AAPL, SPY"),
    start: str | None = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date (YYYY-MM-DD)"),
):
    """
    Placeholder command for now
    """

    typer.echo(f"[regime-radar] Analyzing {ticker} from {start or 'start'} to {end or 'latest'}")

def main() -> None:
    app()

if __name__ == "__main__":
    main()