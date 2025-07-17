# RAG API Documentation

This document describes the new RAG (Retrieval-Augmented Generation) API endpoints that have been added to Open WebUI. These endpoints provide OpenAI-compatible chat completions with RAG capabilities, allowing you to query knowledge collections and files without authentication requirements.

## Overview

The RAG API provides the following key features:

- **OpenAI-compatible format**: Responses follow the standard OpenAI API format
- **No authentication required**: Endpoints can be used without user authentication
- **No chat storage**: Conversations are not stored in the database
- **RAG capabilities**: Query knowledge collections and files for relevant context
- **Source citations**: Responses include links to cited documents
- **Streaming support**: Both streaming and non-streaming responses are supported

## Base URL

All endpoints are available under the `/api/v1/rag` prefix:

```
http://localhost:8080/api/v1/rag
```

## Endpoints

### 1. List Models

**GET** `/api/v1/rag/models`

Returns a list of available models in OpenAI-compatible format.

**Response:**
```json
{
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "open-webui"
    }
  ]
}
```

### 2. List Knowledge Collections

**GET** `/api/v1/rag/knowledge/collections`

Returns a list of available knowledge collections.

**Response:**
```json
{
  "collections": [
    {
      "id": "collection-id",
      "name": "Collection Name",
      "description": "Collection description",
      "created_at": 1677610602,
      "updated_at": 1677610602
    }
  ]
}
```

### 3. List Files

**GET** `/api/v1/rag/files`

Returns a list of available files.

**Response:**
```json
{
  "files": [
    {
      "id": "file-id",
      "name": "document.pdf",
      "type": "pdf",
      "created_at": 1677610602,
      "updated_at": 1677610602
    }
  ]
}
```

### 4. Chat Completions

**POST** `/api/v1/rag/chat/completions`

The main endpoint for RAG-enhanced chat completions.

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "What is the main topic of the documents?"
    }
  ],
  "knowledge_collections": ["collection-id-1", "collection-id-2"],
  "file_ids": ["file-id-1", "file-id-2"],
  "stream": false,
  "max_tokens": 500,
  "temperature": 0.7,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["\n", "END"],
  "user": "user-identifier"
}
```

**Parameters:**
- `model` (required): The model to use for completion
- `messages` (required): Array of message objects with role and content
- `knowledge_collections` (optional): Array of knowledge collection IDs to search
- `file_ids` (optional): Array of file IDs to search
- `stream` (optional): Whether to stream the response (default: false)
- `max_tokens` (optional): Maximum number of tokens to generate
- `temperature` (optional): Sampling temperature (default: 1.0)
- `top_p` (optional): Nucleus sampling parameter (default: 1.0)
- `frequency_penalty` (optional): Frequency penalty (default: 0.0)
- `presence_penalty` (optional): Presence penalty (default: 0.0)
- `stop` (optional): Array of stop sequences
- `user` (optional): User identifier

**Response (Non-streaming):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677610602,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Based on the documents, the main topic is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  },
  "sources": [
    {
      "index": 0,
      "content": "Relevant document content...",
      "metadata": {
        "file_id": "file-id-1",
        "name": "document.pdf",
        "source": "document.pdf"
      },
      "file": {
        "id": "file-id-1",
        "name": "document.pdf",
        "type": "pdf"
      }
    }
  ]
}
```

**Response (Streaming):**
The streaming response follows the standard OpenAI Server-Sent Events format with chunks containing partial completions.

## Usage Examples

### Python Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8080/api/v1/rag"

# Chat completion with RAG
payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What are the key points from the uploaded documents?"
        }
    ],
    "knowledge_collections": ["my-collection"],
    "file_ids": ["file-123"],
    "stream": False,
    "max_tokens": 1000,
    "temperature": 0.7
}

response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
result = response.json()

print("Response:", result["choices"][0]["message"]["content"])
print("Sources:", result.get("sources", []))
```

### cURL Example

```bash
curl -X POST "http://localhost:8080/api/v1/rag/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "Summarize the main points from the documents"
      }
    ],
    "knowledge_collections": ["collection-1"],
    "file_ids": ["file-1"],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8080/api/v1/rag/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [
      {
        role: 'user',
        content: 'What information can you find about this topic?'
      }
    ],
    knowledge_collections: ['collection-1'],
    file_ids: ['file-1'],
    max_tokens: 1000,
    temperature: 0.7
  })
});

const result = await response.json();
console.log('Response:', result.choices[0].message.content);
console.log('Sources:', result.sources);
```

## RAG Process

The RAG API follows this process:

1. **Query Extraction**: Extracts the user query from the last user message
2. **Context Retrieval**: Searches knowledge collections and files for relevant content
3. **Context Integration**: Adds retrieved context to the system prompt
4. **LLM Generation**: Calls the underlying LLM with the enhanced prompt
5. **Source Attribution**: Returns the response with source citations

## Configuration

The RAG API uses the same configuration as the main Open WebUI application:

- **RAG Template**: System prompt template for RAG responses
- **Top-K**: Number of documents to retrieve
- **Hybrid Search**: Whether to use hybrid search (BM25 + vector search)
- **Reranking**: Whether to use reranking for better document selection
- **Embedding Model**: Model used for generating embeddings

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Model not found
- `500`: Internal server error

Error responses include a detail message explaining the issue.

## Limitations

- The API requires at least one OpenAI-compatible model to be configured
- Knowledge collections and files must exist in the Open WebUI system
- The API uses the first available OpenAI API configuration
- Streaming responses may not include source citations in the same format

## Testing

Use the provided `test_rag_api.py` script to test the endpoints:

```bash
python test_rag_api.py
```

Make sure Open WebUI is running on `localhost:8080` before running the tests. 