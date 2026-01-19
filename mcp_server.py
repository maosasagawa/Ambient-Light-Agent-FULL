"""
MCP (Model Context Protocol) Server for Ambient Light Control
支持通过MCP协议调用底层灯光控制功能
"""

import json
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.server.stdio
import asyncio

# Core logic (shared by HTTP + MCP)
import api_core

# Initialize MCP Server
app = Server("ambient-light-mcp")


@app.list_resources()
async def list_resources() -> List[Resource]:
    """No additional resources exposed via MCP."""
    return []


@app.read_resource()
async def read_resource(uri: str) -> str:
    raise ValueError(f"Unknown resource URI: {uri}")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """Tools intended for direct generation (will generate and persist)."""
    return [
        Tool(
            name="generate_lighting_effect",
            description="根据自然语言指令生成灯效（会触发生图+落盘）",
            inputSchema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "用户的自然语言指令",
                    }
                },
                "required": ["instruction"],
            },
        ),
        Tool(
            name="determine_intent",
            description="仅做意图识别（matrix/strip/both）",
            inputSchema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "用户的自然语言指令",
                    }
                },
                "required": ["instruction"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Execute tools (same behavior as HTTP generation)."""
    try:
        args = arguments or {}
        instruction = args.get("instruction", "")

        if name == "generate_lighting_effect":
            result = api_core.generate_lighting_effect(instruction)
        elif name == "determine_intent":
            result = {
                "status": "success",
                "target": api_core.determine_intent(instruction),
                "instruction": instruction,
            }
        else:
            result = {"status": "error", "error": f"Unknown tool: {name}"}

        return [
            TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2),
            )
        ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False),
            )
        ]


async def main():
    """
    启动MCP服务器
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
