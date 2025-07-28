"""Main API client generator interface."""

from typing import Any, Dict, Optional

from agentic_api.core.agent import APIGenerationAgent
from agentic_api.core.config import AgentConfig


class APIClientGenerator:
    """High-level interface for generating API clients from any type of documentation."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the generator with optional configuration."""
        self.config = config or AgentConfig.from_env()
        self.agent = APIGenerationAgent(self.config)
    
    async def generate_from_source(self, source: str) -> str:
        """
        Generate Python API client code from any documentation source.
        
        Args:
            source: Can be:
                - URL to Swagger/OpenAPI JSON/YAML
                - URL to documentation webpage  
                - File path to local documentation
                - Raw text containing API documentation
            
        Returns:
            Generated Python client code as a string
            
        Raises:
            Exception: If generation fails
        """
        result = await self.agent.generate_from_source(source)
        
        if not result["success"]:
            raise Exception(f"Failed to generate API client: {result['error']}")
        
        return result["generated_code"]
    
    async def generate_with_details(self, source: str) -> Dict[str, Any]:
        """
        Generate API client with detailed results including analysis and validation.
        
        Args:
            source: Any type of documentation source (URL, file, text)
            
        Returns:
            Dictionary containing generated code, analysis, validation results, etc.
        """
        return await self.agent.generate_from_source(source)
    
    # Backward compatibility methods
    async def generate_from_url(self, swagger_url: str) -> str:
        """
        Generate Python API client code from a Swagger URL (backward compatibility).
        
        Args:
            swagger_url: URL to the Swagger/OpenAPI documentation
            
        Returns:
            Generated Python client code as a string
            
        Raises:
            Exception: If generation fails
        """
        return await self.generate_from_source(swagger_url)
    
    def set_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Recreate agent with new config
        self.agent = APIGenerationAgent(self.config)
