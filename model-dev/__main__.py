import click
from check import check
from convert import convert
from quant import quantize
from reauthor import reauthor
from train import train


@click.group()
def cli(): ...


cli.add_command(check)
cli.add_command(convert)
cli.add_command(quantize)
cli.add_command(reauthor)
cli.add_command(train)


if __name__ == "__main__":
    cli()
