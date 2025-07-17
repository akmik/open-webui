#!/usr/bin/env python3
"""
Example usage of the RAG API for a realistic scenario
"""

import requests
import json
import time

# Base URL for the Open WebUI RAG API
BASE_URL = "http://localhost:8080/api/v1/rag"

def setup_demo_data():
    """This function would typically be used to set up demo data in Open WebUI"""
    print("Note: This example assumes you have already:")
    print("1. Uploaded some documents to Open WebUI")
    print("2. Created knowledge collections")
    print("3. Configured at least one OpenAI-compatible model")
    print()

def get_available_resources():
    """Get available models, collections, and files"""
    print("Fetching available resources...")
    
    # Get models
    models_response = requests.get(f"{BASE_URL}/models")
    models = models_response.json().get("data", [])
    
    # Get knowledge collections
    collections_response = requests.get(f"{BASE_URL}/knowledge/collections")
    collections = collections_response.json().get("collections", [])
    
    # Get files
    files_response = requests.get(f"{BASE_URL}/files")
    files = files_response.json().get("files", [])
    
    print(f"Available models: {len(models)}")
    print(f"Available collections: {len(collections)}")
    print(f"Available files: {len(files)}")
    
    return models, collections, files

def research_query_example():
    """Example: Research query using RAG"""
    print("=" * 60)
    print("EXAMPLE 1: Research Query")
    print("=" * 60)
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with available model
        "messages": [
            {
                "role": "user",
                "content": "What are the main findings and conclusions from the research documents?"
            }
        ],
        "knowledge_collections": [],  # Add collection IDs if available
        "file_ids": [],  # Add file IDs if available
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.3  # Lower temperature for more focused responses
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(result["choices"][0]["message"]["content"])
            print()
            
            if result.get("sources"):
                print("Sources:")
                for i, source in enumerate(result["sources"]):
                    print(f"  {i+1}. {source.get('file', {}).get('name', 'Unknown')}")
                    print(f"     Content: {source['content'][:100]}...")
                    print()
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def technical_analysis_example():
    """Example: Technical analysis with specific questions"""
    print("=" * 60)
    print("EXAMPLE 2: Technical Analysis")
    print("=" * 60)
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with available model
        "messages": [
            {
                "role": "user",
                "content": "Analyze the technical specifications and provide recommendations for implementation."
            }
        ],
        "knowledge_collections": [],  # Add collection IDs if available
        "file_ids": [],  # Add file IDs if available
        "stream": False,
        "max_tokens": 1500,
        "temperature": 0.5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("Analysis:")
            print(result["choices"][0]["message"]["content"])
            print()
            
            if result.get("sources"):
                print("Referenced Documents:")
                for i, source in enumerate(result["sources"]):
                    print(f"  {i+1}. {source.get('file', {}).get('name', 'Unknown')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def streaming_example():
    """Example: Streaming response"""
    print("=" * 60)
    print("EXAMPLE 3: Streaming Response")
    print("=" * 60)
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with available model
        "messages": [
            {
                "role": "user",
                "content": "Provide a detailed summary of the key points from the documents."
            }
        ],
        "knowledge_collections": [],  # Add collection IDs if available
        "file_ids": [],  # Add file IDs if available
        "stream": True,
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, stream=True)
        if response.status_code == 200:
            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data != '[DONE]':
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                pass
            print()  # New line after streaming
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def conversation_example():
    """Example: Multi-turn conversation with context"""
    print("=" * 60)
    print("EXAMPLE 4: Multi-turn Conversation")
    print("=" * 60)
    
    conversation_messages = [
        {
            "role": "user",
            "content": "What are the main topics covered in the documents?"
        },
        {
            "role": "assistant",
            "content": "Based on the documents, the main topics include..."  # This would be the actual response
        },
        {
            "role": "user",
            "content": "Can you elaborate on the first topic you mentioned?"
        }
    ]
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with available model
        "messages": conversation_messages,
        "knowledge_collections": [],  # Add collection IDs if available
        "file_ids": [],  # Add file IDs if available
        "stream": False,
        "max_tokens": 600,
        "temperature": 0.6
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("Follow-up response:")
            print(result["choices"][0]["message"]["content"])
            print()
            
            if result.get("sources"):
                print("Sources for this response:")
                for i, source in enumerate(result["sources"]):
                    print(f"  {i+1}. {source.get('file', {}).get('name', 'Unknown')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function to run all examples"""
    print("RAG API Usage Examples")
    print("=" * 60)
    print()
    
    setup_demo_data()
    
    try:
        # Get available resources
        models, collections, files = get_available_resources()
        
        if not models:
            print("No models available. Please configure at least one OpenAI-compatible model.")
            return
        
        print(f"Using model: {models[0]['id'] if models else 'No model available'}")
        print()
        
        # Run examples
        research_query_example()
        time.sleep(1)  # Brief pause between examples
        
        technical_analysis_example()
        time.sleep(1)
        
        streaming_example()
        time.sleep(1)
        
        conversation_example()
        
        print("=" * 60)
        print("All examples completed!")
        print()
        print("To use with your own data:")
        print("1. Upload documents to Open WebUI")
        print("2. Create knowledge collections")
        print("3. Update the 'knowledge_collections' and 'file_ids' arrays in the examples")
        print("4. Replace 'gpt-3.5-turbo' with an available model ID")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure Open WebUI is running on localhost:8080")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 