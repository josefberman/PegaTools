"""
Command-line interface for Pega Tools.
"""

import click
from .core import PegaClient
from .utils import PegaException


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Pega Tools - A collection of tools for working with Pega systems."""
    pass


@main.command()
@click.option('--url', required=True, help='Pega instance URL')
@click.option('--username', help='Username for authentication')
@click.option('--password', help='Password for authentication', hide_input=True)
def health(url: str, username: str, password: str):
    """Check the health status of a Pega instance."""
    try:
        client = PegaClient(url, username, password)
        status = client.get_health_status()
        click.echo("Health Status:")
        for key, value in status.items():
            click.echo(f"  {key}: {value}")
        client.close()
    except PegaException as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
def version():
    """Show the version of Pega Tools."""
    click.echo("Pega Tools v0.1.0")


if __name__ == '__main__':
    main() 