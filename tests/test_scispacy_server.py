import asyncio
import sys
import os
import json
from pathlib import Path
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

class BiomedicalEntityExtractor:
    """
    Test copy of the MCP Client.
    """
    def __init__(self):
        self.server_script = str(PROJECT_ROOT / "src/servers/scispacy_server.py")
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        env = os.environ.copy()
        python_exe = str(PROJECT_ROOT / ".venv310/bin/python")
        if not os.path.exists(python_exe):
            raise RuntimeError(f"Python executable not found at {python_exe}. Please ensure .venv310 is created.")

        params = StdioServerParameters(
            command=python_exe,
            args=[self.server_script],
            env=env
        )
        self.read, self.write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def extract(self, text: str):
        result = await self.session.call_tool("extract_entities", arguments={"text": text})
        if result.content and hasattr(result.content[0], "text"):
            return result.content[0].text
        return "[]"

async def main():
    print("Starting MCP Client Test...")
    async with BiomedicalEntityExtractor() as extractor:
        text = "Patients with breast cancer often have mutations in BRCA1 and CHEK2."
        print(f"Testing text: '{text}'")
        
        try:
            result_str = await extractor.extract(text)
            print(f"Raw Result: {result_str}")
            entities = json.loads(result_str)
            print(f"Parsed Entities: {entities}")
            
            assert "breast cancer" in entities
            assert "BRCA1" in entities or "brca1" in entities
            print("✅ Verification Successful!")
        except Exception as e:
            print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
