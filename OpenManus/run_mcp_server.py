# coding: utf-8
# A shortcut to launch OpenManus MCP server, where its introduction also solves other import issues.
from OpenManus.app.mcp.server import MCPServer, parse_args

def run_mcp_server(transport=None) -> None:
    args = parse_args()
    if transport:
        args.transport = transport
    server = MCPServer()
    server.run(transport=args.transport)

if __name__ == "__main__":
    run_mcp_server('sse')
