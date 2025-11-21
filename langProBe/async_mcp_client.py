from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from anthropic import Anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class AsyncMCPClient:

    def __init__(self, headers: dict = None):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.headers = headers

    async def connect_to_streamable_http_server(self, server_url: str, headers: dict = None):
        """Connect to an MCP server running with streamable http transport"""
        # Use headers parameter if provided, otherwise use self.headers
        if headers is None:
            headers = self.headers
        
        # streamablehttp_client is decorated with @asynccontextmanager
        # It returns an async context manager that yields a tuple
        self._streams_context = streamablehttp_client(url=server_url, headers=headers)
        
        # Directly call __aenter__ to get the streams tuple
        # This avoids any potential issues with AsyncExitStack
        streams = await self._streams_context.__aenter__()
        
        # Register the exit to be called during cleanup
        # push_async_exit can accept the context manager object itself
        self.exit_stack.push_async_exit(self._streams_context)
        
        # streams should be a tuple of (read_stream, write_stream, get_session_id_callback)
        if not isinstance(streams, tuple):
            raise ValueError(f"Expected tuple, got {type(streams)}: {streams}")
        if len(streams) != 3:
            raise ValueError(f"Expected tuple of 3 elements, got {len(streams)} elements: {streams}")
        
        read_stream, write_stream, get_session_id_callback = streams
        self._get_session_id_callback = get_session_id_callback
        
        # Create and enter the ClientSession context
        self._session_context = ClientSession(read_stream, write_stream)
        self.session: ClientSession = await self.exit_stack.enter_async_context(self._session_context)

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])


    async def cleanup(self):
        """Properly clean up the session and streams"""
        # AsyncExitStack will automatically clean up all entered contexts
        # This includes both the streams context and the session context
        await self.exit_stack.aclose()

    async def call_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Call a tool with the given arguments"""
        result = await self.session.call_tool(tool_name, tool_args)
        return result

    async def list_tools(self):
        """List available tools"""
        response = await self.session.list_tools()
        return response

    async def get_prompt(self, *args, **kwargs):
        response = await self.session.get_prompt(*args, **kwargs)
        return response

    async def list_prompts(self):
        response = await self.session.list_prompts()
        return response

    async def list_resources(self):
        response = await self.session.list_resources()
        return response

    async def read_resource(self, *args, **kwargs):
        response = await self.session.read_resource(*args, **kwargs)
        return response

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                        "role": "assistant",
                        "content": content.text
                    })
                messages.append({
                    "role": "user",
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        # print("\nMCP Client Started!")
        # print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

# async def main():
#     client = AsyncMCPClient()
#     try:
#         await client.connect_to_sse_server(server_url="http://localhost:8080/sse")
#         result = await client.call_tool("get_alerts", {"state": "CA"})
#         print(result)
#     finally:
#         await client.cleanup()


# result = asyncio.run(main())