# Dev Container

Ready to run Python 3.13 development environment for the **MCP Latin Tools Server**.

## What's inside

- Python 3.13
- `uv` for dependency management and virtual environments
- Node.js 20 (required for MCP Inspector and other Node-based MCP tooling)
- Port `8001` automatically forwarded for the MCP server
- Automatic dependency installation with `uv sync`
- VS Code extensions for:
  - Python
  - Pylance
  - Ruff linting
  - Jupyter notebooks
  - Dev Containers support

## MCP Server Features

The container is configured for development and testing of the Latin MCP server, including:

- Latin tokenization with enclitic `-que` splitting
- UDPipe-based morphological analysis
- Reported speech detection with a fine-tuned LaBERTa transformer model
- LiLa Knowledge Base SPARQL querying (more details in the general readme file)
- Local MCP development and testing with Claude or MCP Inspector

## Usage

1. Open the project folder in VS Code

2. Select:

```
Reopen in Container
```

Or run

```
Dev Containers: Reopen in Container
```
3. Start the MCP server:

```
uv run mcp-latin
```

or 

```
uv run python -m mcp_latin -vv
```

5. MCP endpoint at:

http://localhost:8001/mcp


MCP inspector:

To test and inspect tools, execute the following command:

```
npx @modelcontextprotocol/inspector
```

