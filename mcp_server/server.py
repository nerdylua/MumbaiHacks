from typing import Union, List, Dict, Any, Optional
from fastmcp import FastMCP
from mcp_server.config.mcp_settings import MCP_SERVER_PORT
from mcp_server.tools.rag_pipeline_mcp import rag_pipeline_mcp

mcp = FastMCP("rag-server")


@mcp.tool(description="Process a document from URL and answer questions using RAG (traditional or structure-aware). Supports dynamic or static k chunk selection and returns answers with source metadata.")
async def process_document_rag(
    document_url: str,
    questions: Union[str, List[str]],
    rag_mode: str,                                  # 'traditional' | 'structure_aware'
    k_strategy: str,                                # 'dynamic' | 'static'
    k_value: Optional[int] = None,                  # required when k_strategy == 'static'
):
    return await rag_pipeline_mcp(
        document_url=document_url,
        questions=questions,
        rag_mode=rag_mode,
        k_strategy=k_strategy,
        k_value=k_value,
    )

def run_server(port: int = None):
    mcp.run(transport="streamable-http", port=port or MCP_SERVER_PORT)