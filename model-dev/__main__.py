import click
from check import check
from convert import convert
from quant import quantize


@click.group()
def cli(): ...


cli.add_command(check)
cli.add_command(convert)
cli.add_command(quantize)


if __name__ == "__main__":
    cli()
