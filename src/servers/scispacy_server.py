# /// script
# requires-python = "~=3.10"
# dependencies = [
#     "fastmcp",
#     "numpy==1.26.4",
#     "scispacy==0.6.2",
#     "spacy==3.7.5",
#     "blis==0.7.11",
#     "thinc==8.2.5",
#     "en_core_sci_sm @ https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz",
#     "pydantic",
# ]
# ///

import asyncio
import spacy
import json
from mcp.server.lowlevel import Server
import mcp.types as types
from mcp.server.stdio import stdio_server

# Initialize Server
app = Server("scispacy-server")

# Load model globally
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    # Fallback if model not found, though we expect it installed
    try:
        import en_core_sci_sm
        nlp = en_core_sci_sm.load()
    except ImportError:
        # Last resort: empty dummy if model missing (should not happen in prod)
        print("Warning: en_core_sci_sm model not found.")
        nlp = spacy.blank("en")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="extract_entities",
            description="Extract biomedical entities from text using scispacy (en_core_sci_sm).",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"}
                },
                "required": ["text"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "extract_entities":
        text = arguments.get("text", "")
        if not text:
            return [types.TextContent(type="text", text="[]")]
            
        doc = nlp(text)
        # Return unique entity texts as JSON list
        entities = list(set([ent.text for ent in doc.ents]))
        return [types.TextContent(type="text", text=json.dumps(entities))]
        
    raise ValueError(f"Tool not found: {name}")

async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
