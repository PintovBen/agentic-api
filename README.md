# agentic-api

🤖 **AI-Powered API Client Generator** 

Generate high-quality Python API clients from Swagger/OpenAPI documentation using LangGraph multi-agent workflows.

## ✨ Features

- **🔍 Smart Swagger Analysis**: Automatically parses and understands API documentation
- **🐍 Clean Python Generation**: Creates typed, async Python clients with proper error handling
- **🧪 Built-in Testing**: Comprehensive validation framework with quality scoring
- **⚡ CLI & Python API**: Use via command line or integrate programmatically
- **🔧 Multi-Agent Architecture**: LangGraph workflow with specialized agents for analysis, generation, and validation

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/PintovBen/agentic-api.git
cd agentic-api
pip install -e .
```

### Setup

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Generate Your First Client

```bash
# CLI: Generate a client
agentic-api generate https://petstore.swagger.io/v2/swagger.json

# Test the generated client
agentic-api test generated_client.py
```

```python
# Python API: Generate programmatically
from agentic_api import APIClientGenerator

generator = APIClientGenerator()
client_code = await generator.generate_from_url(
    "https://petstore.swagger.io/v2/swagger.json"
)

# The generated client is ready to use!
```

## 📊 Quality Assurance

Every generated client includes:
- ✅ **Syntax Validation** - Guaranteed valid Python code
- ✅ **Import Resolution** - All dependencies properly handled  
- ✅ **Structure Analysis** - Classes, methods, and models detected
- ✅ **Functionality Testing** - Client instantiation and method validation
- ✅ **Quality Scoring** - Overall quality score with recommendations

Example test results:
```
📊 Test Results for generated_client.py
Overall Score: 90.0% ✅

✅ Syntax Valid
✅ Imports Valid  
✅ Client Class Found
📈 Methods Found: 4
📈 Models Found: 6
✅ Client Instantiation: Success
```

## 🏗️ Architecture

**LangGraph Multi-Agent Workflow:**

```
📥 Swagger URL → 🔍 Analysis Agent → 🔧 Code Generator → ✅ Validator → 📦 Clean Python Client
```

- **Fetch Agent**: Downloads and validates Swagger documentation
- **Analysis Agent**: Understands API structure, endpoints, and models
- **Generation Agent**: Creates clean, typed Python client code
- **Validation Agent**: Tests and scores the generated client

## 📁 Project Structure

```
agentic-api/
├── src/agentic_api/
│   ├── __init__.py
│   ├── cli.py                # Command line interface
│   ├── core/
│   │   ├── agent.py          # LangGraph workflow
│   │   ├── generator.py      # High-level API
│   │   └── config.py         # Configuration
│   └── testing/
│       └── client_tester.py  # Testing framework
├── examples/                 # Usage examples
├── tests/                   # Unit tests
└── IMPROVEMENTS.md          # Latest improvements
```

## 📚 Examples

See the `examples/` directory:
- **`basic_usage.py`** - Simple generation example
- **`complete_workflow.py`** - Full workflow with testing and reporting

## 🛠️ Development

```bash
# Run tests
pytest

# Development dependencies
pip install -e ".[dev]"
```

## 🎯 What Makes This Special

1. **🧠 AI-Powered**: Uses advanced LLMs to understand complex API docs
2. **🔄 Multi-Agent**: Specialized agents for different aspects of client generation
3. **📊 Quality-First**: Built-in testing ensures reliable generated code
4. **⚡ Production Ready**: Generates clients that work out of the box
5. **🔧 Extensible**: Easy to add support for new patterns and frameworks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

*Built with ❤️ using LangGraph, LangChain, and OpenAI*
