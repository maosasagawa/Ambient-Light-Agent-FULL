"""
MCP Server for Ambient Light (voice input)
仅用于语音侧输入，触发生图并落盘
"""

import asyncio
import json
from typing import Any, List

import mcp.server.stdio
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent

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
    """Voice entry: generate and persist effects."""
    return [
        Tool(
            name="voice_generate",
            description="语音侧入口：生成灯效并落盘，返回完整结果",
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
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Generate lighting effects for voice input."""
    try:
        args = arguments or {}
        instruction = (args.get("instruction") or "").strip()

        if name != "voice_generate":
            result = {
                "status": "error",
                "error": {
                    "code": "unknown_tool",
                    "message": f"Unknown tool: {name}",
                },
            }
        elif not instruction:
            result = {
                "status": "error",
                "error": {
                    "code": "invalid_request",
                    "message": "instruction is required",
                },
            }
        else:
            result = api_core.generate_lighting_effect(instruction)

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
                text=json.dumps(
                    {
                        "status": "error",
                        "error": {"code": "internal_error", "message": str(e)},
                    },
                    ensure_ascii=False,
                ),
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
