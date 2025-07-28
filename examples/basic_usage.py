"""Basic example of using the agentic API generator."""

import asyncio
import os
from agentic_api import APIClientGenerator
from agentic_api.core.config import AgentConfig


async def main():
    """Generate API client from Petstore Swagger."""
    
    # Configure the generator
    config = AgentConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        client_name="PetStoreClient",
        async_client=True,
        include_examples=True,
    )
    
    # Initialize generator
    generator = APIClientGenerator(config)
    
    # Generate client from Petstore API
    swagger_url = "https://petstore.swagger.io/v2/swagger.json"
    
    print(f"Generating API client from: {swagger_url}")
    
    try:
        # Get detailed results
        result = await generator.generate_with_details(swagger_url)
        
        if result["success"]:
            print("✅ Generation successful!")
            
            # Show analysis
            analysis = result["analysis"]
            print(f"\nAPI Analysis:")
            print(f"- Title: {analysis.get('overview', {}).get('title', 'N/A')}")
            print(f"- Version: {analysis.get('overview', {}).get('version', 'N/A')}")
            print(f"- Endpoints: {len(analysis.get('endpoints', []))}")
            
            # Show validation
            validation = result["validation"]
            print(f"\nValidation:")
            print(f"- Quality Score: {validation.get('quality_score', 'N/A')}/10")
            print(f"- Syntax Valid: {validation.get('syntax_valid', False)}")
            
            # Save the generated code
            output_file = "generated_petstore_client.py"
            with open(output_file, "w") as f:
                f.write(result["generated_code"])
            
            print(f"\n💾 Client saved to: {output_file}")
            print(f"📄 Lines of code: {len(result['generated_code'].splitlines())}")
            
        else:
            print(f"❌ Generation failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Make sure you have OPENAI_API_KEY set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    asyncio.run(main())
