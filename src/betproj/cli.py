import typer

app = typer.Typer(no_args_is_help=True)

@app.command()
def hello():
    typer.echo("OK")

if __name__ == "__main__":
    app()
