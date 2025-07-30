# 🤖 Agentic API - AI-Powered API Client Generator

Generate high-quality Python API clients from Swagger/OpenAPI documentation using advanced AI agents.

> **⚠️ Important**: This agent accepts **ONLY swagger.json based documents**. Other input types (web pages, raw text, etc.) have been tested and provide useless results. Please ensure your API documentation is in swagger.json format.

## ✨ What It Does

- **🔍 Smart Analysis**: Automatically parses and understands Swagger JSON documentation
- **🐍 Clean Code**: Generates typed, async Python clients with proper error handling
- **🧪 Auto-Testing**: Validates generated code with comprehensive quality scoring (typically 8.5-9.5/10)
- **🚀 Ready to Use**: Produces production-ready API clients you can use immediately

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
agentic-api generate "https://petstore.swagger.io/v2/swagger.json"
agentic-api generate "https://cve.circl.lu/api/swagger.json"

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
- Complete API coverage
- Async/await support  
- Type hints
- Error handling
- Documentation
- Usage examples

## 📋 Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Swagger.json URL (not YAML or other formats)

## 🎯 Example Usage

```bash
# Popular APIs that work great:
make generate URL=https://petstore.swagger.io/v2/swagger.json
make generate URL=https://cve.circl.lu/api/swagger.json  
make generate URL=https://api.apis.guru/v2/specs/github.com/1.1.4/swagger.json

# The generated client will be saved as a Python file in your current directory
```

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

## 📁 Output

Generated API clients include:
- **Main client file**: Complete Python API client
- **Documentation**: Usage examples and API reference  
- **Type definitions**: Full typing support
- **Error handling**: Robust error management
- **Async support**: Modern async/await patterns

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

## 🚀 Advanced Usage

For development or customization:
```bash
# Development mode with live code mounting
docker-compose run --rm agentic-api-dev

# Custom model configuration  
AGENTIC_MODEL=gpt-4 make generate URL=your-swagger-url.json
```

That's it! You now have a powerful AI agent that generates production-ready API clients from any Swagger JSON specification. 🎉
