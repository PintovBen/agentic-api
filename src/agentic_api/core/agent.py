"""Main LangGraph agent for API client generation."""

import json
import logging
import re
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agentic_api.core.config import AgentConfig

# Configure logging for clean, pretty output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Clean format without timestamps/levels
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
        
        # Setup clean logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        
        # Ensure clean console output
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')  # Clean format
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.propagate = False  # Prevent duplicate messages
    
    def _create_graph(self) -> StateGraph:
        """Create a simplified LangGraph workflow for Swagger/OpenAPI processing."""
        workflow = StateGraph(AgentState)
        
        # Enhanced workflow with fix step after comprehensive validation
        workflow.add_node("fetch_swagger", self._fetch_swagger_node)
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("validate_code", self._validate_code_node)
        workflow.add_node("validate_against_docs", self._validate_against_docs_node)
        workflow.add_node("fix_code", self._fix_code_node)
        
        # Workflow with fix step after all validations
        workflow.set_entry_point("fetch_swagger")
        workflow.add_edge("fetch_swagger", "generate_code")
        workflow.add_edge("generate_code", "validate_code")
        workflow.add_edge("validate_code", "validate_against_docs")
        workflow.add_edge("validate_against_docs", "fix_code")
        workflow.add_edge("fix_code", END)
        
        return workflow.compile()
    
    async def _fetch_swagger_node(self, state: AgentState) -> AgentState:
        """Fetch and parse Swagger/OpenAPI specification from URL."""
        input_source = state["input_source"]
        
        self.logger.info("")
        self.logger.info("🚀 " + "="*70)
        self.logger.info("🚀 STARTING API CLIENT GENERATION")
        self.logger.info("🚀 " + "="*70)
        self.logger.info(f"📡 Fetching API specification from: {input_source}")
        
        try:
            import httpx
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(input_source, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; AgenticAPI/1.0)',
                    'Accept': 'application/json, application/yaml, text/yaml, */*'
                })
                response.raise_for_status()
                
                # Parse based on content type
                content_type = response.headers.get('content-type', '').lower()
                
                if 'json' in content_type or input_source.endswith('.json'):
                    swagger_content = response.json()
                elif 'yaml' in content_type or input_source.endswith(('.yaml', '.yml')):
                    try:
                        import yaml
                        swagger_content = yaml.safe_load(response.text)
                    except ImportError:
                        state["error"] = "YAML support requires 'pyyaml' package. Install with: pip install pyyaml"
                        return state
                else:
                    # Try JSON first, then YAML
                    try:
                        swagger_content = response.json()
                    except json.JSONDecodeError:
                        try:
                            import yaml
                            swagger_content = yaml.safe_load(response.text)
                        except (ImportError, Exception) as e:
                            state["error"] = f"Could not parse content as JSON or YAML from {input_source}: {str(e)}"
                            return state
                
                # Validate it's a valid Swagger/OpenAPI spec
                if not self._is_valid_swagger_content(swagger_content):
                    state["error"] = f"Content at {input_source} is not a valid Swagger/OpenAPI specification"
                    return state
                
                # Log overview with nice formatting
                paths_count = len(swagger_content.get('paths', {}))
                host = swagger_content.get('host', 'N/A')
                base_path = swagger_content.get('basePath', 'N/A')
                title = swagger_content.get('info', {}).get('title', 'Unknown API')
                version = swagger_content.get('info', {}).get('version', 'Unknown')
                
                self.logger.info("✅ Successfully fetched API specification!")
                self.logger.info("")
                self.logger.info("📊 API OVERVIEW:")
                self.logger.info(f"   📝 Title: {title}")
                self.logger.info(f"   🔢 Version: {version}")
                self.logger.info(f"   🌐 Host: {host}")
                self.logger.info(f"   📁 Base Path: {base_path}")
                self.logger.info(f"   🔗 Endpoints: {paths_count}")
                self.logger.info("")
                
                state["swagger_content"] = swagger_content
                state["input_type"] = "swagger"
                state["raw_content"] = response.text
                
                state["messages"].append(
                    HumanMessage(content=f"Successfully fetched Swagger/OpenAPI spec from {input_source}")
                )
                
        except Exception as e:
            error_msg = f"Failed to fetch Swagger specification: {str(e)}"
            self.logger.info(f"❌ {error_msg}")
            state["error"] = error_msg
            state["messages"].append(
                HumanMessage(content=f"Error fetching from {input_source}: {str(e)}")
            )
        
        return state
    
    def _is_valid_swagger_content(self, content: dict) -> bool:
        """Validate that content is a proper Swagger/OpenAPI specification."""
        if not isinstance(content, dict):
            return False
        
        # Check for Swagger 2.0
        if content.get('swagger') == '2.0' and 'paths' in content:
            return True
        
        # Check for OpenAPI 3.x
        if (content.get('openapi') and 
            content.get('openapi').startswith('3.') and 
            'paths' in content):
            return True
        
        return False
    
    async def generate_from_swagger_url(self, swagger_url: str) -> Dict[str, Any]:
        """Generate API client from Swagger/OpenAPI URL."""
        initial_state = AgentState(
            input_source=swagger_url,
            input_type="",
            raw_content="",
            swagger_content={},
            analysis={},
            generated_code="",
            validation_results={},
            messages=[],
            error=""
        )
        
        # Run the simplified graph
        result = await self.graph.ainvoke(initial_state)
        
        # Pretty completion summary
        self._log_completion_summary(result)
        
        return {
            "success": not bool(result.get("error")),
            "error": result.get("error"),
            "generated_code": result.get("generated_code", ""),
            "validation": result.get("validation_results", {}),
            "swagger_content": result.get("swagger_content", {}),
            "input_type": result.get("input_type", ""),
            "messages": [msg.content for msg in result.get("messages", [])]
        }
    
    def _log_completion_summary(self, result: dict):
        """Log a pretty completion summary."""
        self.logger.info("🎉 " + "="*70)
        self.logger.info("🎉 API CLIENT GENERATION COMPLETED")
        self.logger.info("🎉 " + "="*70)
        
        if result.get("error"):
            self.logger.info(f"❌ Generation failed: {result['error']}")
        else:
            self.logger.info("✅ Generation successful!")
            
            # Show validation results if available
            validation = result.get("validation_results", {})
            if validation.get("quality_score"):
                score = validation['quality_score']
                score_emoji = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                self.logger.info(f"{score_emoji} Final Quality Score: {score}/10")
            
            if validation.get("endpoint_count"):
                self.logger.info(f"🔗 Total Endpoints: {validation['endpoint_count']}")
                
            if validation.get("model_count"):
                self.logger.info(f"📦 Total Models: {validation['model_count']}")
            
            # Show if code was fixed
            if result.get("fixed"):
                self.logger.info("🔧 Code was automatically fixed!")
            
            # Final recommendations
            critical_issues = validation.get("critical_blocking_issues", [])
            if not critical_issues:
                self.logger.info("🚀 Client is ready for production use!")
            else:
                self.logger.info("⚠️  Some issues may remain - review the generated code")
        
        self.logger.info("🎉 " + "="*70)
        self.logger.info("")
    
    # Backward compatibility
    async def generate_from_url(self, swagger_url: str) -> Dict[str, Any]:
        """Generate API client from Swagger URL (backward compatibility)."""
        return await self.generate_from_swagger_url(swagger_url)
    
    async def generate_from_source(self, input_source: str) -> Dict[str, Any]:
        """Generate API client from source (backward compatibility)."""
        return await self.generate_from_swagger_url(input_source)
    
    def _extract_swagger_urls_from_content(self, content: str, base_url: str) -> List[str]:
        """Extract potential Swagger/OpenAPI URLs from content."""
        import re
        
        urls = []
        
        # Look for common patterns in the content
        patterns = [
            r'swagger\.json["\']?\s*:\s*["\']([^"\']+)["\']',
            r'openapi\.json["\']?\s*:\s*["\']([^"\']+)["\']',
            r'swagger\.yaml["\']?\s*:\s*["\']([^"\']+)["\']',
            r'openapi\.yaml["\']?\s*:\s*["\']([^"\']+)["\']',
            r'href\s*=\s*["\']([^"\']*swagger\.json[^"\']*)["\']',
            r'href\s*=\s*["\']([^"\']*openapi\.json[^"\']*)["\']',
            r'href\s*=\s*["\']([^"\']*swagger\.yaml[^"\']*)["\']',
            r'href\s*=\s*["\']([^"\']*openapi\.yaml[^"\']*)["\']',
            r'url\s*[=:]\s*["\']([^"\']*swagger[^"\']*)["\']',
            r'url\s*[=:]\s*["\']([^"\']*openapi[^"\']*)["\']',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if not match.startswith('http'):
                    if match.startswith('/'):
                        urls.append(match)
                    else:
                        urls.append('/' + match)
                else:
                    urls.append(match)
        
        return urls
    
    def _is_valid_swagger_content(self, content: Dict[str, Any]) -> bool:
        """Check if the content is a valid Swagger/OpenAPI specification."""
        if not isinstance(content, dict):
            return False
        
        # Check for Swagger 2.0
        if content.get('swagger') == '2.0' and 'paths' in content:
            return True
        
        # Check for OpenAPI 3.x
        if content.get('openapi') and content.get('openapi').startswith('3.') and 'paths' in content:
            return True
        
        # Check for basic structure that looks like an API spec
        if 'paths' in content and isinstance(content['paths'], dict):
            return True
        
        return False
    
    def _extract_api_details_from_text(self, llm_response: str, raw_content: str) -> Dict[str, Any]:
        """Extract API details when JSON parsing fails."""
        import re
        
        # Try to find key patterns in the LLM response and raw content
        details = {
            "base_url": "",
            "auth": {"type": "none"},
            "endpoints": [],
            "models": {},
            "client_name": "APIClient"
        }
        
        # Look for base URL patterns
        url_patterns = [
            r'https?://[^\s"\'<>]+',
            r'"base_url":\s*"([^"]+)"',
            r'Base URL.*?:\s*(https?://[^\s]+)'
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, llm_response + " " + raw_content, re.IGNORECASE)
            if matches:
                details["base_url"] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                break
        
        # Look for endpoint patterns
        endpoint_patterns = [
            r'(GET|POST|PUT|DELETE)\s+(/[^\s<>"\']+)',
            r'"path":\s*"([^"]+)".*?"method":\s*"([^"]+)"',
            r'/(v\d+/[^\s<>"\']+)'
        ]
        
        for pattern in endpoint_patterns:
            matches = re.findall(pattern, llm_response + " " + raw_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    details["endpoints"].append({
                        "method": match[0].upper(),
                        "path": match[1],
                        "params": {},
                        "response_schema": {}
                    })
        
        return details
    
    async def _generate_code_node(self, state: AgentState) -> AgentState:
        """Generate Python API client code directly from Swagger/OpenAPI specification."""
        if state.get("error"):
            return state
        
        swagger_content = state.get("swagger_content", {})
        raw_content = state.get("raw_content", "")
        
        if not swagger_content and not raw_content:
            state["error"] = "No Swagger/OpenAPI content available for code generation"
            return state
        
        # Use the raw Swagger JSON/YAML content directly
        swagger_text = raw_content if raw_content else json.dumps(swagger_content, indent=2)
        
        # Extract key info for better generation
        host = swagger_content.get('host', 'N/A')
        base_path = swagger_content.get('basePath', 'N/A')
        expected_base_url = f"https://{host}{base_path}" if host != 'N/A' and base_path != 'N/A' else "unknown"
        
        self.logger.info("🤖 " + "="*70)
        self.logger.info("🤖 GENERATING PYTHON API CLIENT")
        self.logger.info("🤖 " + "="*70)
        self.logger.info(f"🎯 Target API: {expected_base_url}")
        self.logger.info("⚙️  Generating comprehensive Python client with all endpoints and models...")
        self.logger.info("")
        
        code_prompt = f"""
You are an expert Python developer creating a production-ready API client from a Swagger/OpenAPI specification.

SWAGGER/OPENAPI SPECIFICATION:
{swagger_text}

CRITICAL REQUIREMENTS - Follow these exactly:

1. **BASE URL EXTRACTION AND DEFAULT**: 
   - Extract the correct base URL from the spec: look for "host" + "basePath" OR "servers" array
   - If host is "cve.circl.lu" and basePath is "/api", the base_url should be "https://cve.circl.lu/api"
   - ALWAYS provide this as the DEFAULT value in the constructor
   - Example: `def __init__(self, api_key: str, base_url: str = "https://cve.circl.lu/api"):`
   - NEVER use placeholder URLs like "api.example.com"

2. **FIX KWARGS SYNTAX**:
   - In method signatures, use `**kwargs` NOT `kwargs`
   - In the _request method: `async def _request(self, method: str, endpoint: str, **kwargs) -> Any:`
   - In the request call: `response = await self.client.request(method, endpoint, headers=self.headers, **kwargs)`
   - In method calls: `await self._request("GET", "/endpoint", params=params)`

3. **HANDLE SPECIAL FIELD NAMES**:
   - For fields starting with @ (like @ID, @Name), use Field aliases:
   - Example: `id: str = Field(alias="@ID")` NOT `@ID: str`
   - Import Field: `from pydantic import BaseModel, Field`

4. **AUTHENTICATION**:
   - Check "securityDefinitions" (Swagger 2.0) or "components.securitySchemes" (OpenAPI 3.x)
   - If apiKey type, implement the correct header name
   - Example: `self.headers = {{"X-API-KEY": api_key}}`

5. **ALL ENDPOINTS WITH PROPER PARAMETERS**: 
   - Implement EVERY endpoint found in the "paths" section
   - For methods with optional parameters, use `**kwargs` properly:
   - Example: `async def list_items(self, page: int = 1, per_page: int = 100, **kwargs) -> ItemsList:`
   - Then: `params = {{"page": page, "per_page": per_page, **kwargs}}`

6. **COMPLETE MODELS**:
   - Create Pydantic models for ALL schemas in "definitions" or "components.schemas"
   - Use correct Python types: string->str, integer->int, boolean->bool, array->List, object->Dict
   - Handle special characters in field names with aliases

7. **PROPER RETURN TYPES**:
   - Don't return `None` for GET methods that return data
   - Use the correct Pydantic model as return type
   - Example: `async def get_vulnerability(self, vulnerability_id: str) -> Vulnerability:`

8. **PARAMETER HANDLING**:
   - For pagination: `page: int = 1, per_page: int = 100`
   - For filtering: `**kwargs` to handle optional filters
   - Properly merge parameters: `params = {{"page": page, "per_page": per_page, **kwargs}}`

Generate the complete client following these exact patterns. Include ALL endpoints, ALL models, and fix ALL syntax issues.

Return ONLY clean Python code without markdown formatting.
        """
        
        messages = [
            SystemMessage(content="You are an expert Python developer specializing in creating accurate API clients from Swagger/OpenAPI specifications. You MUST extract the correct base URL, implement ALL endpoints, fix syntax issues, and create complete Pydantic models. Never use placeholder URLs or incomplete implementations."),
            HumanMessage(content=code_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            self.logger.info("📝 " + "="*70)
            self.logger.info("📝 LLM CODE GENERATION RESPONSE")
            self.logger.info("📝 " + "="*70)
            # Show a truncated preview instead of full response
            preview = response.content[:500] + "..." if len(response.content) > 500 else response.content
            self.logger.info(preview)
            self.logger.info("📝 " + "="*70)
            self.logger.info("")
            
            generated_code = self._clean_generated_code(response.content)
            
            # Quick analysis with nice formatting
            class_count = generated_code.count('class ')
            method_count = generated_code.count('async def ')
            
            self.logger.info("✅ Code generation completed!")
            self.logger.info(f"   📋 Classes generated: {class_count}")
            self.logger.info(f"   🔧 Methods generated: {method_count}")
            self.logger.info(f"   📄 Total lines: {len(generated_code.splitlines())}")
            self.logger.info("")
            
            state["generated_code"] = generated_code
            state["messages"].append(response)
            
        except Exception as e:
            error_msg = f"Failed to generate code: {str(e)}"
            self.logger.info(f"❌ {error_msg}")
            state["error"] = error_msg
        
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
        """Validate the generated code for syntax and basic functionality."""
        if state.get("error"):
            return state
        
        generated_code = state["generated_code"]
        
        self.logger.info("🔍 " + "="*70)
        self.logger.info("🔍 VALIDATING GENERATED CODE")
        self.logger.info("🔍 " + "="*70)
        
        # Basic syntax validation
        try:
            compile(generated_code, '<generated>', 'exec')
            syntax_valid = True
            syntax_error = None
            self.logger.info("✅ Syntax validation: PASSED")
        except SyntaxError as e:
            syntax_valid = False
            syntax_error = str(e)
            self.logger.info(f"❌ Syntax validation: FAILED on line {e.lineno}")
            self.logger.info(f"   Error: {syntax_error}")
        
        self.logger.info("🧠 Running comprehensive LLM validation...")
        self.logger.info("")
        
        # LLM-based validation
        validation_prompt = f"""
Review this generated Python API client code and provide a detailed assessment.

GENERATED CODE:
{generated_code}

Perform these CRITICAL checks:

1. **BASE URL DEFAULT VALUE CHECK**:
   - Does the constructor have a default base_url parameter?
   - Look for: `def __init__(self, api_key: str, base_url: str = "https://...")`
   - The default should be the REAL API URL, not a placeholder
   - CRITICAL FAILURE if no default or if it contains "example.com"

2. **KWARGS SYNTAX CHECK**:
   - In _request method: Is it `**kwargs` or just `kwargs`?
   - In method signatures: Is it `**kwargs` or just `kwargs`?
   - In request calls: Is it `**kwargs` or just `kwargs`?
   - CRITICAL FAILURE if `kwargs` is used without `**`

3. **FIELD NAME VALIDATION**:
   - Look for fields starting with @ (like @ID, @Name)
   - Should use `Field(alias="@ID")` pattern
   - CRITICAL FAILURE if raw @ symbols are used as field names

4. **PARAMETER HANDLING**:
   - Methods should properly merge parameters: `params = {{"page": page, **kwargs}}`
   - Not: `params = {{"page": page, kwargs}}`
   - CRITICAL FAILURE if kwargs not properly unpacked in params

5. **IMPORTS CHECK**:
   - Must include `from pydantic import BaseModel, Field` if using aliases
   - Must include proper typing imports
   - CRITICAL FAILURE if Field is used but not imported

6. **RETURN TYPE VALIDATION**:
   - GET methods should return proper model types, not None
   - Methods should have meaningful return type annotations
   - Warning if return types are missing or incorrect

7. **AUTHENTICATION IMPLEMENTATION**:
   - Should have proper header setup in constructor
   - Should use correct header name (X-API-KEY, Authorization, etc.)

8. **ENDPOINT COMPLETENESS**:
   - Count implemented endpoints vs expected (should be 40+ for CVE API)
   - Check if major categories are covered (vulnerability, browse, stats, etc.)

Return JSON with:
{{
    "quality_score": 1-10,
    "syntax_valid": true/false,
    "base_url_has_default": true/false,
    "base_url_is_real": true/false,
    "base_url_found": "actual default URL in constructor",
    "kwargs_syntax_correct": true/false,
    "field_aliases_correct": true/false,
    "has_proper_imports": true/false,
    "parameter_handling_correct": true/false,
    "endpoint_count": number,
    "authentication_correct": true/false,
    "model_count": number,
    "critical_blocking_issues": ["issues that prevent the client from working"],
    "syntax_errors": ["specific syntax problems"],
    "field_name_errors": ["problems with @ symbols in field names"],
    "kwargs_errors": ["specific kwargs usage problems"],
    "missing_features": ["functionality gaps"],
    "suggestions": ["specific actionable fixes"]
}}

Mark issues as CRITICAL if they would prevent the client from working at all.
        """
        
        messages = [
            SystemMessage(content="You are a senior Python code reviewer specializing in API client validation. Focus on identifying critical issues like wrong URLs, syntax errors, missing endpoints, and incomplete implementations. Provide specific, actionable feedback."),
            HumanMessage(content=validation_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            self.logger.info("� " + "="*70)
            self.logger.info("📋 LLM VALIDATION RESULTS")
            self.logger.info("📋 " + "="*70)
            
            try:
                validation_results = json.loads(response.content)
            except json.JSONDecodeError:
                # Extract JSON if wrapped in markdown
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                if json_match:
                    try:
                        validation_results = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        validation_results = {"raw_feedback": response.content, "parse_error": "Could not parse JSON"}
                else:
                    validation_results = {"raw_feedback": response.content, "parse_error": "No JSON found"}
            
            # Add syntax validation results
            validation_results["syntax_valid"] = syntax_valid
            if syntax_error:
                validation_results["syntax_error"] = syntax_error
            
            # Pretty print validation summary
            if "quality_score" in validation_results:
                score = validation_results['quality_score']
                score_emoji = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                self.logger.info(f"{score_emoji} Quality Score: {score}/10")
            
            if "endpoint_count" in validation_results:
                self.logger.info(f"🔗 Endpoints Found: {validation_results['endpoint_count']}")
                
            if "model_count" in validation_results:
                self.logger.info(f"📦 Models Found: {validation_results['model_count']}")
            
            if "critical_blocking_issues" in validation_results:
                issues = validation_results['critical_blocking_issues']
                if issues:
                    self.logger.info(f"⚠️  Critical Issues Found: {len(issues)}")
                    for i, issue in enumerate(issues, 1):
                        self.logger.info(f"   {i}. {issue}")
                else:
                    self.logger.info("✅ No critical blocking issues found!")
            
            self.logger.info("")
            
            state["validation_results"] = validation_results
            state["messages"].append(response)
            
        except Exception as e:
            error_msg = f"Failed to validate code: {str(e)}"
            self.logger.info(f"❌ {error_msg}")
            state["error"] = error_msg
        
        return state
    
    async def _fix_code_node(self, state: AgentState) -> AgentState:
        """Fix the code based on comprehensive validation results."""
        if state.get("error"):
            return state
        
        validation_results = state.get("validation_results", {})
        generated_code = state["generated_code"]
        
        # Check validation from both basic and documentation validation
        critical_issues = validation_results.get("critical_blocking_issues", [])
        syntax_error = validation_results.get("syntax_error")
        kwargs_errors = validation_results.get("kwargs_errors", [])
        field_errors = validation_results.get("field_name_errors", [])
        
        # Also check documentation validation results
        doc_validation = validation_results.get("documentation_validation", {})
        critical_fixes_needed = doc_validation.get("critical_fixes_needed", [])
        missing_endpoints = doc_validation.get("endpoint_validation", {}).get("missing_endpoints", [])
        auth_issues = doc_validation.get("auth_validation", {}).get("issues", [])
        
        # Combine all issues
        all_issues = critical_issues + critical_fixes_needed + auth_issues
        if missing_endpoints:
            all_issues.append(f"Missing {len(missing_endpoints)} endpoints from specification")
        
        # Only proceed if there are actual issues to fix
        if not all_issues and not syntax_error and not kwargs_errors and not field_errors:
            self.logger.info("✅ No critical issues found - code is ready to use!")
            self.logger.info("")
            return state
        
        self.logger.info("🔧 " + "="*70)
        self.logger.info("🔧 FIXING CODE ISSUES")
        self.logger.info("🔧 " + "="*70)
        self.logger.info(f"🎯 Total issues to fix: {len(all_issues)}")
        
        # Show top issues
        for i, issue in enumerate(all_issues[:3], 1):
            self.logger.info(f"   {i}. {issue}")
        if len(all_issues) > 3:
            self.logger.info(f"   ... and {len(all_issues) - 3} more issues")
        
        self.logger.info("")
        self.logger.info("⚙️  Applying comprehensive fixes...")
        
        # Prepare comprehensive fix prompt
        fix_prompt = f"""
You are a Python code expert fixing a broken API client. Your PRIMARY GOAL is to make this code FUNCTIONAL and EXECUTABLE.

CURRENT BROKEN CODE:
{generated_code}

VALIDATION RESULTS:
{json.dumps(validation_results, indent=2)}

ALL ISSUES TO FIX:
{chr(10).join([f"- {issue}" for issue in all_issues])}

🚨 CRITICAL FUNCTIONAL REQUIREMENTS - FIX THESE FIRST:

1. **KWARGS SYNTAX ERRORS** (BLOCKS EXECUTION):
   - FIND: `async def _request(self, method: str, endpoint: str, kwargs)`
   - REPLACE: `async def _request(self, method: str, endpoint: str, **kwargs)`
   - FIND: `self.client.request(method, endpoint, headers=self.headers, kwargs)`
   - REPLACE: `self.client.request(method, f"{{self.base_url}}{{endpoint}}", headers=self.headers, **kwargs)`
   - This is CRITICAL - code will crash without this fix!

2. **BASE URL CONCATENATION** (PREVENTS API CALLS):
   - Ensure ALL requests use: `f"{{self.base_url}}{{endpoint}}"`
   - Example: `await self.client.request("GET", f"{{self.base_url}}/pet/123", ...)`

3. **MODEL INSTANTIATION ERRORS** (RUNTIME CRASHES):
   - FIND: `return SomeModel(response_data)`
   - REPLACE: `return SomeModel(**response_data)` (note the **)
   - FIND: `[Model(item) for item in items]`
   - REPLACE: `[Model(**item) for item in items]`

4. **MISSING REQUIRED IMPORTS**:
   - Ensure `from pydantic import BaseModel, Field` if Field aliases are used
   - Add any missing typing imports

5. **AUTHENTICATION HEADERS**:
   - Check the specification for correct header names
   - For Petstore: use `{{"api_key": api_key}}` not `{{"X-API-KEY": api_key}}`
   - For CVE API: use `{{"X-API-KEY": api_key}}`

6. **PYDANTIC COMPATIBILITY**:
   - Replace `.dict()` with `.model_dump()` for Pydantic v2
   - Or use `.dict()` consistently for Pydantic v1

🎯 FUNCTIONALITY CHECKLIST - ENSURE EACH WORKS:

✅ Constructor creates client instance without errors
✅ _request method can make HTTP calls successfully  
✅ Response parsing doesn't crash with type errors
✅ Model instantiation works with actual API responses
✅ All methods have correct parameter passing (**kwargs not kwargs)
✅ Authentication headers match specification exactly
✅ URL construction includes base_url properly

📋 SECONDARY IMPROVEMENTS (after fixing critical issues):

- Add proper error handling for HTTP status codes
- Implement missing endpoints from specification
- Fix return types from `None` to proper models for GET methods
- Remove unnecessary Field aliases when field name matches JSON key
- Add docstrings for better usability

⚠️ VALIDATION STRATEGY:
After each fix, mentally test:
1. "Can Python import this code?" (syntax check)
2. "Can I create a client instance?" (constructor check)  
3. "Can I call client._request()?" (method signature check)
4. "Will response parsing work?" (model instantiation check)

Return ONLY the corrected Python code that is guaranteed to be functional.
No explanations, no markdown formatting, just clean executable Python code.

FOCUS: Make it work first, optimize later!
        """
        
        messages = [
            SystemMessage(content="You are a Python debugging expert specializing in making broken API clients functional. Your #1 priority is ensuring the code can execute without syntax errors, runtime crashes, or basic functionality failures. Focus on kwargs syntax, URL construction, model instantiation, and authentication headers. Return only working, executable Python code - no explanations or markdown."),
            HumanMessage(content=fix_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # Don't show the full LLM response to reduce clutter
            self.logger.info("✅ Code fixes applied by LLM")
            
            fixed_code = response.content.strip()
            
            # Remove markdown formatting if present
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code[9:]
            if fixed_code.startswith("```"):
                fixed_code = fixed_code[3:]
            if fixed_code.endswith("```"):
                fixed_code = fixed_code[:-3]
            fixed_code = fixed_code.strip()
            
            # Quick syntax validation of fixed code
            try:
                compile(fixed_code, '<fixed>', 'exec')
                
                # Quick analysis of the fixed code
                class_count = fixed_code.count('class ')
                method_count = fixed_code.count('async def ')
                endpoint_count = fixed_code.count('await self._request')
                
                self.logger.info("✅ Fixed code passed syntax validation!")
                self.logger.info(f"   📋 Classes: {class_count}")
                self.logger.info(f"   🔧 Methods: {method_count}")
                self.logger.info(f"   🌐 API Calls: {endpoint_count}")
                self.logger.info("")
                
                state["generated_code"] = fixed_code
                state["fixed"] = True
            except SyntaxError as e:
                self.logger.info(f"❌ Fixed code still has syntax errors: {str(e)}")
                self.logger.info(f"   Error on line {e.lineno}")
                # Keep original code if fix failed
                state["fix_error"] = f"Fixed code has syntax errors: {str(e)}"
            
            state["messages"].append(response)
            
        except Exception as e:
            error_msg = f"Failed to fix code: {str(e)}"
            self.logger.info(f"❌ {error_msg}")
            state["error"] = error_msg
        
        return state
    
    async def _validate_against_docs_node(self, state: AgentState) -> AgentState:
        """Validate the generated client against the original Swagger/OpenAPI specification and suggest improvements."""
        if state.get("error"):
            return state
        
        generated_code = state["generated_code"]
        swagger_content = state.get("swagger_content", {})
        analysis = state.get("analysis", {})
        input_source = state["input_source"]
        
        validation_prompt = f"""
You are validating a generated Python API client against its Swagger/OpenAPI specification.

ORIGINAL SWAGGER/OPENAPI SPECIFICATION:
{json.dumps(swagger_content, indent=2)[:4000]}

EXTRACTED ANALYSIS:
{json.dumps(analysis, indent=2)}

GENERATED API CLIENT CODE:
{generated_code}

INPUT SOURCE: {input_source}

Perform a comprehensive validation and return JSON with specific findings:

1. **URL Validation**: 
   - Check if base URL is correctly extracted and implemented
   - Verify against 'host', 'basePath', or 'servers' in spec

2. **Endpoint Validation**:
   - Compare all paths in spec vs implemented methods
   - Verify HTTP methods match exactly
   - Check parameter mapping (query, path, header, body)
   - Validate required vs optional parameters

3. **Authentication Validation**:
   - Check if auth scheme matches specification
   - Verify implementation of security definitions

4. **Model Validation**:
   - Compare Pydantic models against schema definitions
   - Check field types and requirements
   - Verify model relationships

5. **Code Quality**:
   - Check for proper error handling
   - Validate type hints and documentation
   - Review code structure and patterns

Return JSON with this structure:
{{
    "validation_score": 1-10,
    "url_validation": {{
        "correct": true/false,
        "expected": "correct_url",
        "found": "implemented_url",
        "issue": "description if incorrect"
    }},
    "endpoint_validation": {{
        "total_endpoints_in_spec": number,
        "total_endpoints_implemented": number,
        "missing_endpoints": ["list of missing paths"],
        "incorrect_methods": ["endpoint: expected vs found"],
        "parameter_issues": ["specific parameter problems"]
    }},
    "auth_validation": {{
        "correct": true/false,
        "expected_scheme": "auth type from spec",
        "implemented_scheme": "auth type in code",
        "issues": ["specific auth problems"]
    }},
    "model_validation": {{
        "models_in_spec": number,
        "models_implemented": number,
        "missing_models": ["ModelName1", "ModelName2"],
        "incorrect_models": ["model: field issues"],
        "type_mismatches": ["specific type problems"]
    }},
    "code_quality": {{
        "has_error_handling": true/false,
        "has_type_hints": true/false,
        "has_documentation": true/false,
        "follows_conventions": true/false,
        "issues": ["code quality problems"]
    }},
    "critical_fixes_needed": [
        "Most important fix 1",
        "Most important fix 2"
    ],
    "suggested_improvements": [
        "Improvement suggestion 1",
        "Improvement suggestion 2"
    ],
    "overall_assessment": "Detailed assessment of the generated client"
}}

Be specific about what's wrong and provide actionable feedback for improvements.
        """
        
        messages = [
            SystemMessage(content="You are a senior software engineer specializing in API client validation. Provide detailed, actionable feedback on code quality and spec compliance."),
            HumanMessage(content=validation_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            try:
                doc_validation = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                if json_match:
                    doc_validation = json.loads(json_match.group(1))
                else:
                    doc_validation = {
                        "validation_score": 5,
                        "raw_feedback": response.content,
                        "error": "Could not parse validation JSON"
                    }
            
            # Merge with existing validation results
            existing_validation = state.get("validation_results", {})
            existing_validation["documentation_validation"] = doc_validation
            
            state["validation_results"] = existing_validation
            state["messages"].append(response)
            
        except Exception as e:
            state["error"] = f"Failed to validate against documentation: {str(e)}"
        
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
