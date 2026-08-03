import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    async with sse_client("http://localhost:8000/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # List available MCP tools
            tools = await session.list_tools()
            print("Available Tools:", [t.name for t in tools.tools])

            # Call Third-Factor Tool over network
            result = await session.call_tool(
                "evaluate_third_factor",
                arguments={
                    "ideal_vector": [1.0, 0.0, -1.0, 0.5],
                    "branch_ids": ["branch_alpha", "branch_beta_shattered"],
                    "candidate_states": [
                        [0.95, 0.05, -0.90, 0.48],
                        [-0.50, 0.80, 0.20, -0.90]
                    ],
                    "f1_reflex_logits": [
                        [0.8, -0.1, -0.7, 0.3],
                        [-0.2, 0.9, 0.1, -0.5]
                    ]
                }
            )
            print("Result:", result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
