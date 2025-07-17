import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

import aiohttp
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from open_webui.retrieval.utils import (
    query_collection,
    query_collection_with_hybrid_search,
    get_sources_from_files,
)
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.models.knowledge import Knowledges
from open_webui.models.files import Files
from open_webui.config import (
    RAG_TEMPLATE,
    RAG_EMBEDDING_QUERY_PREFIX,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

router = APIRouter()

# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    messages: List[ChatMessage] = Field(..., description="The messages to complete")
    knowledge_collections: Optional[List[str]] = Field(None, description="Knowledge collection names to search")
    file_ids: Optional[List[str]] = Field(None, description="File IDs to search")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    user: Optional[str] = Field(None, description="User identifier")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None
    sources: Optional[List[Dict[str, Any]]] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

def extract_query_from_messages(messages: List[ChatMessage]) -> str:
    """Extract the user query from the messages"""
    # Get the last user message
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""

def format_sources_for_response(sources_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format sources data for the response"""
    sources = []
    if sources_data and "documents" in sources_data and "metadatas" in sources_data:
        documents = sources_data["documents"][0] if sources_data["documents"] else []
        metadatas = sources_data["metadatas"][0] if sources_data["metadatas"] else []
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            source = {
                "index": i,
                "content": doc,
                "metadata": metadata or {},
            }
            # Add file information if available
            if metadata and "file_id" in metadata:
                file_obj = Files.get_file_by_id(metadata["file_id"])
                if file_obj:
                    source["file"] = {
                        "id": metadata["file_id"],
                        "name": file_obj.filename,
                        "type": file_obj.meta.get("type") if file_obj.meta else None,
                    }
            sources.append(source)
    
    return sources

async def get_rag_context(
    request: Request,
    query: str,
    knowledge_collections: Optional[List[str]] = None,
    file_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Get RAG context from knowledge collections and files"""
    context = {"documents": [], "metadatas": []}
    
    try:
        # Get embedding function from app state
        embedding_function = request.app.state.EMBEDDING_FUNCTION
        
        # Query knowledge collections
        if knowledge_collections:
            collection_names = []
            for collection_name in knowledge_collections:
                # Check if collection exists and is accessible
                if VECTOR_DB_CLIENT.has_collection(collection_name):
                    collection_names.append(collection_name)
            
            if collection_names:
                if request.app.state.config.ENABLE_RAG_HYBRID_SEARCH:
                    collection_results = query_collection_with_hybrid_search(
                        collection_names=collection_names,
                        queries=[query],
                        embedding_function=embedding_function,
                        k=request.app.state.config.TOP_K,
                        reranking_function=getattr(request.app.state, 'rf', None),
                        k_reranker=request.app.state.config.TOP_K_RERANKER,
                        r=request.app.state.config.RELEVANCE_THRESHOLD,
                        hybrid_bm25_weight=request.app.state.config.HYBRID_BM25_WEIGHT,
                    )
                else:
                    collection_results = query_collection(
                        collection_names=collection_names,
                        queries=[query],
                        embedding_function=embedding_function,
                        k=request.app.state.config.TOP_K,
                    )
                
                if collection_results and collection_results.get("documents") and collection_results.get("metadatas"):
                    context["documents"].extend(collection_results["documents"][0] or [])
                    context["metadatas"].extend(collection_results["metadatas"][0] or [])
        
        # Query files
        if file_ids:
            files_data = []
            for file_id in file_ids:
                file_obj = Files.get_file_by_id(file_id)
                if file_obj:
                    files_data.append({
                        "id": file_id,
                        "name": file_obj.filename,
                        "type": "file",
                        "file": {
                            "data": {
                                "content": file_obj.data.get("content", ""),
                                "metadata": file_obj.data.get("metadata", {})
                            }
                        }
                    })
            
            if files_data:
                file_context = get_sources_from_files(
                    request=request,
                    files=files_data,
                    queries=[query],
                    embedding_function=embedding_function,
                    k=request.app.state.config.TOP_K,
                    reranking_function=getattr(request.app.state, 'rf', None),
                    k_reranker=request.app.state.config.TOP_K_RERANKER,
                    r=request.app.state.config.RELEVANCE_THRESHOLD,
                    hybrid_bm25_weight=request.app.state.config.HYBRID_BM25_WEIGHT,
                    hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                )
                
                if file_context and file_context.get("documents") and file_context.get("metadatas"):
                    context["documents"].extend(file_context["documents"][0] or [])
                    context["metadatas"].extend(file_context["metadatas"][0] or [])
    
    except Exception as e:
        log.exception(f"Error getting RAG context: {e}")
    
    return context

def create_system_prompt_with_context(context: Dict[str, Any]) -> str:
    """Create a system prompt with RAG context"""
    if not context.get("documents"):
        return str(RAG_TEMPLATE)
    
    context_text = "\n\n".join(context["documents"])
    return f"{str(RAG_TEMPLATE)}\n\nRelevant context:\n{context_text}"

async def call_llm_api(
    request: Request,
    model: str,
    messages: List[ChatMessage],
    context: Dict[str, Any],
    stream: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Call the LLM API with the enhanced messages"""
    # Create system prompt with context
    system_prompt = create_system_prompt_with_context(context)
    
    # Prepare messages for LLM
    llm_messages = [{"role": "system", "content": system_prompt}]
    llm_messages.extend([msg.model_dump() for msg in messages])
    
    # Prepare payload
    payload = {
        "model": model,
        "messages": llm_messages,
        "stream": stream,
        **kwargs
    }
    
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    # Call OpenAI-compatible API
    try:
        # Get the first available OpenAI API configuration
        if not hasattr(request.app.state, 'config') or not request.app.state.config.OPENAI_API_BASE_URLS:
            raise HTTPException(status_code=500, detail="No OpenAI API configuration available")
        
        url = request.app.state.config.OPENAI_API_BASE_URLS[0]
        key = request.app.state.config.OPENAI_API_KEYS[0]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if stream:
                    return StreamingResponse(
                        response.content,
                        status_code=response.status,
                        headers=dict(response.headers),
                    )
                else:
                    return await response.json()
    
    except Exception as e:
        log.exception(f"Error calling LLM API: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def rag_chat_completions(
    request: Request,
    form_data: ChatCompletionRequest,
):
    """RAG-enhanced chat completions endpoint"""
    try:
        # Extract query from messages
        query = extract_query_from_messages(form_data.messages)
        if not query:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Get RAG context
        context = await get_rag_context(
            request=request,
            query=query,
            knowledge_collections=form_data.knowledge_collections,
            file_ids=form_data.file_ids,
        )
        
        # Prepare LLM parameters
        llm_params = {}
        if form_data.max_tokens is not None:
            llm_params["max_tokens"] = form_data.max_tokens
        if form_data.temperature is not None:
            llm_params["temperature"] = form_data.temperature
        if form_data.top_p is not None:
            llm_params["top_p"] = form_data.top_p
        if form_data.frequency_penalty is not None:
            llm_params["frequency_penalty"] = form_data.frequency_penalty
        if form_data.presence_penalty is not None:
            llm_params["presence_penalty"] = form_data.presence_penalty
        if form_data.stop is not None:
            llm_params["stop"] = form_data.stop
        
        # Call LLM API
        if form_data.stream:
            return await call_llm_api(
                request=request,
                model=form_data.model,
                messages=form_data.messages,
                context=context,
                stream=True,
                **llm_params
            )
        else:
            response = await call_llm_api(
                request=request,
                model=form_data.model,
                messages=form_data.messages,
                context=context,
                stream=False,
                **llm_params
            )
            
            # Format sources for response
            sources = format_sources_for_response(context)
            
            # Add sources to response if available
            if sources:
                response["sources"] = sources
            
            return response
    
    except Exception as e:
        log.exception(f"Error in RAG chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/collections")
async def list_knowledge_collections():
    """List available knowledge collections"""
    try:
        collections = Knowledges.get_knowledge_bases()
        return {
            "collections": [
                {
                    "id": collection.id,
                    "name": collection.name,
                    "description": collection.description,
                    "created_at": collection.created_at,
                    "updated_at": collection.updated_at,
                }
                for collection in collections
            ]
        }
    except Exception as e:
        log.exception(f"Error listing knowledge collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files")
async def list_files():
    """List available files"""
    try:
        files = Files.get_files()
        return {
            "files": [
                {
                    "id": file.id,
                    "name": file.filename,
                    "type": file.meta.get("type") if file.meta else None,
                    "created_at": file.created_at,
                    "updated_at": file.updated_at,
                }
                for file in files
            ]
        }
    except Exception as e:
        log.exception(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models(request: Request):
    """List available models"""
    try:
        # Get models from the main app state
        if hasattr(request.app.state, 'OPENAI_MODELS'):
            models = []
            for model_id, model_info in request.app.state.OPENAI_MODELS.items():
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "open-webui",
                })
            return {"data": models}
        else:
            return {"data": []}
    except Exception as e:
        log.exception(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 