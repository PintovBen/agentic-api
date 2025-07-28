"""Main LangGraph agent for API client generation."""

import json
import re
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agentic_api.core.config import AgentConfig


class AgentState(TypedDict):
    """State for the API client generation agent."""
    
    input_source: str  # URL, file path, or raw text
    input_type: str    # 'swagger', 'webpage', 'text', 'file'
    raw_content: str   # Raw content from source
    swagger_content: Dict[str, Any]  # Parsed swagger if applicable
    analysis: Dict[str, Any]
    generated_code: str
    validation_results: Dict[str, Any]
    messages: List[BaseMessage]
    error: str


class APIGenerationAgent:
    """LangGraph-based agent for generating API clients from Swagger docs."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("detect_input_type", self._detect_input_type_node)
        workflow.add_node("fetch_content", self._fetch_content_node)
        workflow.add_node("parse_content", self._parse_content_node)
        workflow.add_node("analyze_api", self._analyze_api_node)
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("validate_code", self._validate_code_node)
        
        # Define edges
        workflow.set_entry_point("detect_input_type")
        workflow.add_edge("detect_input_type", "fetch_content")
        workflow.add_edge("fetch_content", "parse_content")
        workflow.add_edge("parse_content", "analyze_api")
        workflow.add_edge("analyze_api", "generate_code")
        workflow.add_edge("generate_code", "validate_code")
        workflow.add_edge("validate_code", END)
        
        return workflow.compile()
    
    async def _detect_input_type_node(self, state: AgentState) -> AgentState:
        """Detect the type of input source and prepare for processing."""
        input_source = state["input_source"]
        
        # Determine input type
        if input_source.startswith(('http://', 'https://')):
            if any(keyword in input_source.lower() for keyword in ['swagger', 'openapi', '.json', '.yaml', '.yml']):
                state["input_type"] = "swagger"
            else:
                state["input_type"] = "webpage"
        elif input_source.endswith(('.json', '.yaml', '.yml')):
            state["input_type"] = "swagger_file"
        elif input_source.endswith(('.txt', '.md', '.rst', '.html')):
            state["input_type"] = "doc_file"
        elif '\n' in input_source or len(input_source) > 200:  # Likely raw text
            state["input_type"] = "text"
            state["raw_content"] = input_source
        else:
            # Try to determine if it's a file path
            from pathlib import Path
            if Path(input_source).exists():
                if input_source.endswith(('.json', '.yaml', '.yml')):
                    state["input_type"] = "swagger_file"
                else:
                    state["input_type"] = "doc_file"
            else:
                state["input_type"] = "text"
                state["raw_content"] = input_source
        
        state["messages"].append(
            HumanMessage(content=f"Detected input type: {state['input_type']} for source: {input_source[:100]}...")
        )
        
        return state
    
    async def _fetch_content_node(self, state: AgentState) -> AgentState:
        """Fetch content from various sources."""
        if state.get("error") or state["input_type"] == "text":
            return state  # Content already in raw_content for text type
        
        input_source = state["input_source"]
        input_type = state["input_type"]
        
        try:
            if input_type in ["swagger", "webpage"]:
                # Fetch from URL
                import httpx
                
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(input_source, headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; AgenticAPI/1.0; +https://github.com/agentic-api)'
                    })
                    response.raise_for_status()
                    
                    if input_type == "swagger":
                        # Try to parse as JSON/YAML
                        content_type = response.headers.get('content-type', '').lower()
                        if 'json' in content_type:
                            state["swagger_content"] = response.json()
                            state["raw_content"] = response.text
                        elif 'yaml' in content_type or 'yml' in content_type:
                            try:
                                import yaml
                                state["swagger_content"] = yaml.safe_load(response.text)
                                state["raw_content"] = response.text
                            except ImportError:
                                state["raw_content"] = response.text
                        else:
                            state["raw_content"] = response.text
                    else:  # webpage
                        html_content = response.text
                        
                        # Try to use BeautifulSoup if available, otherwise use basic parsing
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # Remove script and style elements
                            for script in soup(["script", "style"]):
                                script.decompose()
                            
                            # Get text content
                            text_content = soup.get_text()
                            
                            # Clean up whitespace
                            lines = (line.strip() for line in text_content.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            clean_text = ' '.join(chunk for chunk in chunks if chunk)
                            
                            state["raw_content"] = clean_text
                        except ImportError:
                            # Fallback: basic HTML tag removal with regex
                            import re
                            # Remove script and style blocks
                            clean_html = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                            clean_html = re.sub(r'<style[^>]*>.*?</style>', '', clean_html, flags=re.DOTALL | re.IGNORECASE)
                            # Remove HTML tags
                            clean_text = re.sub(r'<[^>]+>', '', clean_html)
                            # Clean up whitespace
                            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                            
                            state["raw_content"] = clean_text
            
            elif input_type in ["swagger_file", "doc_file"]:
                # Read from file
                from pathlib import Path
                file_path = Path(input_source)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                state["raw_content"] = content
                
                if input_type == "swagger_file":
                    if file_path.suffix.lower() == '.json':
                        state["swagger_content"] = json.loads(content)
                    else:  # YAML
                        import yaml
                        state["swagger_content"] = yaml.safe_load(content)
            
            state["messages"].append(
                HumanMessage(content=f"Successfully fetched content from {input_source}")
            )
            
        except Exception as e:
            state["error"] = f"Failed to fetch content: {str(e)}"
            state["messages"].append(
                HumanMessage(content=f"Error fetching content: {str(e)}")
            )
        
        return state
    
    async def _parse_content_node(self, state: AgentState) -> AgentState:
        """Parse and structure the content for analysis."""
        if state.get("error"):
            return state
        
        input_type = state["input_type"]
        raw_content = state["raw_content"]
        
        # If we already have swagger content, skip parsing
        if state.get("swagger_content"):
            return state
        
        # Try to extract structured information from unstructured content
        if input_type in ["webpage", "text", "doc_file"]:
            parse_prompt = f"""
            Analyze the following API documentation and extract structured information:
            
            CONTENT:
            {raw_content[:5000]}...
            
            Extract and format as JSON:
            1. api_name: The name of the API
            2. base_url: The base URL for API calls
            3. authentication: How to authenticate (API key, OAuth, etc.)
            4. endpoints: List of endpoints with methods, paths, parameters, responses
            5. data_models: Any data structures or schemas mentioned
            6. examples: Code examples or request/response samples
            7. rate_limits: Any rate limiting information
            8. additional_info: Any other relevant details
            
            If you can't find specific information, use "not_specified" as the value.
            Focus on extracting concrete API implementation details.
            """
            
            messages = [
                SystemMessage(content="You are an expert at parsing API documentation and extracting structured information from unstructured text."),
                HumanMessage(content=parse_prompt)
            ]
            
            try:
                response = await self.llm.ainvoke(messages)
                
                # Try to parse as JSON
                try:
                    parsed_info = json.loads(response.content)
                    state["analysis"] = {"parsed_documentation": parsed_info}
                except json.JSONDecodeError:
                    # If JSON parsing fails, store as raw analysis
                    state["analysis"] = {"raw_parsing": response.content}
                
                state["messages"].append(response)
                
            except Exception as e:
                state["error"] = f"Failed to parse content: {str(e)}"
        
        return state
    
    async def _analyze_api_node(self, state: AgentState) -> AgentState:
        """Analyze API documentation to understand structure and generate insights."""
        if state.get("error"):
            return state
        
        # Determine what content we have
        swagger_content = state.get("swagger_content", {})
        raw_content = state.get("raw_content", "")
        input_type = state["input_type"]
        existing_analysis = state.get("analysis", {})
        
        if swagger_content:
            # We have structured Swagger/OpenAPI content
            analysis_prompt = f"""
            Analyze the following Swagger/OpenAPI documentation and provide a structured analysis:
            
            {json.dumps(swagger_content, indent=2)}
            
            Please provide:
            1. API overview (title, version, base URL)
            2. Authentication methods
            3. ALL endpoints and their purposes
            4. Data models/schemas
            5. Key patterns and conventions
            
            Format as JSON with keys: overview, auth, endpoints, models, patterns
            """
        else:
            # We have unstructured content that was parsed
            analysis_prompt = f"""
            Based on the following API documentation content, provide a comprehensive analysis:
            
            INPUT TYPE: {input_type}
            
            RAW CONTENT:
            {raw_content[:4000]}...
            
            EXISTING PARSED INFO:
            {json.dumps(existing_analysis, indent=2)}
            
            Provide a structured analysis including:
            1. API overview (name, purpose, base URL if found)
            2. Authentication methods mentioned
            3. All API endpoints, methods, and parameters you can identify
            4. Data models, request/response formats
            5. Usage patterns and conventions
            6. Rate limits or restrictions
            7. Code examples if present
            
            Be thorough and extract as much structured information as possible.
            Format as JSON with keys: overview, auth, endpoints, models, patterns, examples, limitations
            """
        
        messages = [
            SystemMessage(content="You are an expert API analyst. Analyze documentation and provide structured insights for API client generation."),
            HumanMessage(content=analysis_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Try to parse as JSON, fallback to text
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                analysis = {"raw_analysis": response.content}
            
            # Merge with existing analysis
            if existing_analysis:
                analysis.update(existing_analysis)
            
            state["analysis"] = analysis
            state["messages"].append(response)
            
        except Exception as e:
            state["error"] = f"Failed to analyze API documentation: {str(e)}"
        
        return state
    
    async def _generate_code_node(self, state: AgentState) -> AgentState:
        """Generate Python API client code from analyzed documentation."""
        if state.get("error"):
            return state
        
        swagger_content = state.get("swagger_content", {})
        raw_content = state.get("raw_content", "")
        analysis = state["analysis"]
        input_type = state["input_type"]
        
        # Determine API name for the client class
        api_name = "API"
        if swagger_content:
            api_name = swagger_content.get('info', {}).get('title', 'API')
        elif analysis.get('overview', {}).get('name'):
            api_name = analysis['overview']['name']
        elif analysis.get('parsed_documentation', {}).get('api_name'):
            api_name = analysis['parsed_documentation']['api_name']
        
        client_name = self.config.client_name or api_name.replace(' ', '').replace('-', '') + 'Client'
        
        # Create comprehensive code generation prompt
        if swagger_content:
            # We have structured Swagger content
            content_section = f"""
            SWAGGER CONTENT:
            {json.dumps(swagger_content, indent=2)[:4000]}...
            """
        else:
            # We have unstructured content
            content_section = f"""
            ORIGINAL DOCUMENTATION:
            {raw_content[:3000]}...
            
            EXTRACTED ANALYSIS:
            {json.dumps(analysis, indent=2)}
            """
        
        code_prompt = f"""
        You are a Python code generator. Generate ONLY valid Python code without any markdown formatting, explanations, or code blocks.

        Based on this API documentation, generate a complete Python API client:
        
        INPUT TYPE: {input_type}
        {content_section}
        
        REQUIREMENTS:
        1. Generate ONLY Python code - no markdown, no explanations, no code blocks
        2. Start directly with imports
        3. Create Pydantic models for all data structures found in the documentation
        4. Create a main client class: {client_name}
        5. Make it {'async' if self.config.async_client else 'sync'}
        6. Use httpx for HTTP requests
        7. Include proper error handling and validation
        8. Add comprehensive type hints and docstrings
        9. Handle authentication as specified in the documentation
        10. Implement ALL endpoints/methods mentioned in the documentation
        11. Add request/response validation where possible
        12. Include rate limiting handling if mentioned
        13. End with a simple usage example in comments
        
        IMPORTANT NOTES:
        - If the documentation is incomplete, make reasonable assumptions
        - Focus on creating a practical, usable client
        - Include error handling for common HTTP scenarios
        - Add logging capabilities
        - Make the client extensible and maintainable
        
        Generate clean, executable Python code only.
        """
        
        messages = [
            SystemMessage(content="You are an expert Python developer specializing in API client generation. Generate clean, well-documented, production-ready code that works with any type of API documentation."),
            HumanMessage(content=code_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            raw_code = response.content
            
            # Clean up the generated code
            generated_code = self._clean_generated_code(raw_code)
            
            state["generated_code"] = generated_code
            state["messages"].append(response)
            
        except Exception as e:
            state["error"] = f"Failed to generate code: {str(e)}"
        
        return state
    
    def _clean_generated_code(self, raw_code: str) -> str:
        """Clean up generated code by removing markdown and fixing formatting."""
        import re
        
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*\n', '', raw_code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```.*$', '', code, flags=re.MULTILINE)
        
        # Remove explanatory text before code
        lines = code.split('\n')
        python_start = 0
        
        for i, line in enumerate(lines):
            # Look for import statements or class definitions to find where Python code starts
            if (line.strip().startswith('import ') or 
                line.strip().startswith('from ') or 
                line.strip().startswith('class ') or
                line.strip().startswith('def ') or
                line.strip().startswith('"""') and 'python' not in line.lower()):
                python_start = i
                break
        
        # Extract only the Python code part
        clean_lines = lines[python_start:]
        
        # Remove any trailing explanatory text
        python_end = len(clean_lines)
        for i in range(len(clean_lines) - 1, -1, -1):
            line = clean_lines[i].strip()
            if (line and 
                not line.startswith('#') and 
                not line.startswith('"""') and
                not line.startswith("'''") and
                'Key Features' not in line and
                'Usage Example' not in line and
                'Dependencies' not in line):
                python_end = i + 1
                break
        
        clean_code = '\n'.join(clean_lines[:python_end])
        
        # Remove any remaining markdown artifacts
        clean_code = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_code)  # Remove bold
        clean_code = re.sub(r'\*([^*]+)\*', r'\1', clean_code)      # Remove italics
        clean_code = re.sub(r'`([^`]+)`', r'\1', clean_code)        # Remove inline code
        
        return clean_code.strip()
    
    async def _validate_code_node(self, state: AgentState) -> AgentState:
        """Validate the generated code for syntax and best practices."""
        if state.get("error"):
            return state
        
        generated_code = state["generated_code"]
        
        # 1. Basic syntax validation
        try:
            compile(generated_code, '<generated>', 'exec')
            syntax_valid = True
            syntax_error = None
        except SyntaxError as e:
            syntax_valid = False
            syntax_error = str(e)
        
        # 2. Comprehensive testing using our testing framework
        test_report = None
        try:
            from agentic_api.testing.client_tester import GeneratedClientTester
            tester = GeneratedClientTester()
            test_report = await tester.test_generated_client(generated_code)
        except Exception as e:
            print(f"Warning: Could not run comprehensive tests: {e}")
        
        # 3. LLM-based validation
        validation_prompt = f"""
        Review this generated Python API client code for:
        1. Code quality and best practices
        2. Type safety and error handling
        3. Documentation completeness
        4. API design patterns
        5. Potential improvements
        
        CODE:
        {generated_code}
        
        Provide feedback as JSON with keys: quality_score (1-10), issues, suggestions, approval
        """
        
        messages = [
            SystemMessage(content="You are a senior Python code reviewer. Evaluate code quality, patterns, and best practices."),
            HumanMessage(content=validation_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            try:
                validation_results = json.loads(response.content)
            except json.JSONDecodeError:
                validation_results = {"raw_feedback": response.content}
            
            validation_results["syntax_valid"] = syntax_valid
            if syntax_error:
                validation_results["syntax_error"] = syntax_error
            
            # Add test report if available
            if test_report:
                validation_results["test_report"] = {
                    "overall_score": test_report.overall_score,
                    "syntax_valid": test_report.syntax_valid,
                    "imports_valid": test_report.imports_valid,
                    "has_main_class": test_report.has_main_class,
                    "method_count": test_report.method_count,
                    "model_count": test_report.model_count,
                    "recommendations": test_report.recommendations
                }
            
            state["validation_results"] = validation_results
            state["messages"].append(response)
            
        except Exception as e:
            state["error"] = f"Failed to validate code: {str(e)}"
        
        return state
    
    async def generate_from_source(self, input_source: str) -> Dict[str, Any]:
        """Generate API client from any type of documentation source."""
        initial_state = AgentState(
            input_source=input_source,
            input_type="",
            raw_content="",
            swagger_content={},
            analysis={},
            generated_code="",
            validation_results={},
            messages=[],
            error=""
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "success": not bool(result.get("error")),
            "error": result.get("error"),
            "generated_code": result.get("generated_code", ""),
            "analysis": result.get("analysis", {}),
            "validation": result.get("validation_results", {}),
            "swagger_content": result.get("swagger_content", {}),
            "input_type": result.get("input_type", ""),
            "raw_content": result.get("raw_content", "")[:500] + "..." if len(result.get("raw_content", "")) > 500 else result.get("raw_content", "")
        }
    
    # Backward compatibility
    async def generate_from_url(self, swagger_url: str) -> Dict[str, Any]:
        """Generate API client from Swagger URL (backward compatibility)."""
        return await self.generate_from_source(swagger_url)
