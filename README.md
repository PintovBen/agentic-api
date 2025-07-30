# 🤖 Agentic API - AI-Powered API Client Generator

Generate high-quality Python API clients from Swagger/OpenAPI documentation using advanced AI agents.

> **⚠️ Important**: This agent accepts **ONLY swagger.json based documents**. Other input types (web pages, raw text, etc.) have been tested and provide useless results. Please ensure your API documentation is in swagger.json format.

## ✨ What It Does

- **🔍 Smart Analysis**: Automatically parses and understands Swagger JSON documentation
- **🐍 Clean Code**: Generates typed, async Python clients with proper error handling
- **🧪 Auto-Testing**: Validates generated code with comprehensive quality scoring
- **🚀 Ready to Use**: Produces API clients you can use immediately

## 📋 Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Swagger.json URL (not YAML or other formats)

## 🚀 Quick Start Guide

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/PintovBen/agentic-api.git
cd agentic-api

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Step 2: Build the Docker Image

```bash
# Build the container
make build
```

### Step 3: Generate API Clients

#### Option A: Interactive Mode (Recommended)
```bash
# Start an interactive shell inside the container
docker-compose run --rm agentic-api

# Inside the container, generate clients:
agentic-api generate "https://petstore.swagger.io/v2/swagger.json" --client-name "<CLIENT_NAME>"
agentic-api generate "https://cve.circl.lu/api/swagger.json" --client-name "<CLIENT_NAME>"

# Exit when done
exit
```

#### Option B: Direct Command
```bash
# Generate a client directly
make generate URL=https://petstore.swagger.io/v2/swagger.json
```

### Step 4: Use Your Generated Client

The generated Python client will appear in your current directory with:
- API coverage
- Async/await support  
- Type hints
- Error handling

## 🛠️ Available Commands

```bash
make help          # Show all available commands
make build         # Build the Docker image  
make run           # Start interactive container
make generate      # Generate API client (requires URL parameter)
make clean         # Remove containers and images
```

## 🔧 Environment Configuration

Set these environment variables or create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key-here
AGENTIC_MODEL=gpt-4o-mini
AGENTIC_TEMPERATURE=0.1
```

## 🐛 Troubleshooting

**Container won't start?**
```bash
# Clean rebuild
make clean
make build
```

**API key issues?**
```bash
# Check your environment
docker-compose run --rm agentic-api env | grep OPENAI
```

**URL not working?**
- Ensure the URL returns valid swagger.json (not YAML)
- Test the URL in your browser first
- Some APIs require authentication headers

## 🎯 Using Your Generated API Client

Once you've generated an API client, you can use it in any Python program of your own:

### Simple Usage

1. **Import the generated client** into your Python code
2. **Create a client instance** with your API credentials  
3. **Call any function** from the client in your code

```python
# Example: Using a generated client
import asyncio
from your_generated_client import APIClient

async def main():
    # Create client instance
    client = APIClient(api_key="your-api-key")
    
    # Call any function from the generated client
    result = await client.list_items()
    print(f"Got {len(result.data)} items")
    
    # Close when done
    await client.close()

asyncio.run(main())
```

### Install Dependencies

Your generated client needs these packages:
```bash
pip install httpx pydantic
```

That's it! The generated client is ready to use in any Python project. All functions are documented and include proper error handling, type hints, and async support. 🚀
