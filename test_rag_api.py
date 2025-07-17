#!/usr/bin/env python3
"""
Test script for the new RAG API endpoints
"""

import requests
import json

# Base URL for the Open WebUI API
BASE_URL = "http://localhost:8080/api/v1/rag"

def test_list_models():
    """Test listing available models"""
    print("Testing /models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_list_knowledge_collections():
    """Test listing knowledge collections"""
    print("Testing /knowledge/collections endpoint...")
    response = requests.get(f"{BASE_URL}/knowledge/collections")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_list_files():
    """Test listing files"""
    print("Testing /files endpoint...")
    response = requests.get(f"{BASE_URL}/files")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_chat_completion():
    """Test RAG chat completion"""
    print("Testing /chat/completions endpoint...")
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with an available model
        "messages": [
            {
                "role": "user",
                "content": "What is the main topic of the documents?"
            }
        ],
        "knowledge_collections": [],  # Add collection names if available
        "file_ids": [],  # Add file IDs if available
        "stream": False,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_chat_completion_with_sources():
    """Test RAG chat completion with knowledge collections and files"""
    print("Testing /chat/completions with sources...")
    
    # First, get available collections and files
    collections_response = requests.get(f"{BASE_URL}/knowledge/collections")
    files_response = requests.get(f"{BASE_URL}/files")
    
    collections = collections_response.json().get("collections", [])
    files = files_response.json().get("files", [])
    
    # Use first available collection and file if they exist
    collection_names = [collections[0]["id"]] if collections else []
    file_ids = [files[0]["id"]] if files else []
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with an available model
        "messages": [
            {
                "role": "user",
                "content": "What information can you find about this topic in the available documents?"
            }
        ],
        "knowledge_collections": collection_names,
        "file_ids": file_ids,
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("Testing RAG API Endpoints")
    print("=" * 50)
    
    try:
        test_list_models()
        test_list_knowledge_collections()
        test_list_files()
        test_chat_completion()
        test_chat_completion_with_sources()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure Open WebUI is running on localhost:8080")
    except Exception as e:
        print(f"Error: {e}") 