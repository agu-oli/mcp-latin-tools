"""Entry point for the MCP Latin server.

Usage:
    uv run mcp-latin          Start the HTTP MCP server (default port 8001)

"""
from __future__ import annotations

import logging
import os

import click


@click.command()
@click.option("--port", "-p", type=int, default=None, help="HTTP port (default: $MCP_PORT or 8001)")
@click.option("--host", "-H", type=str, default="0.0.0.0", help="Bind host")
@click.option("-v", "--verbose", count=True, help="Verbosity: -v INFO, -vv DEBUG")
def main(port: int | None, host: str, verbose: int) -> None:
    """Start the Latin MCP server (StreamableHTTP transport)."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv
    load_dotenv()  # env var > .env > code default

    import uvicorn
    from mcp_latin.server import build_app

    effective_port = port or int(os.getenv("MCP_PORT", "8001"))
    app = build_app()

    uvicorn.run(app, host=host, port=effective_port)


if __name__ == "__main__":
    main()
