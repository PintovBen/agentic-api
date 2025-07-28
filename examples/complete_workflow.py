"""
Complete workflow example: Generate and test an API client.
"""
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from agentic_api import APIClientGenerator
from agentic_api.core.config import AgentConfig


async def complete_workflow_example():
    """
    Demonstrate the complete workflow:
    1. Generate API client from Swagger
    2. Test the generated client
    3. Show results
    """
    
    print("🚀 Complete API Client Generation Workflow")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # 1. Configure the generator
        config = AgentConfig(
            model_name="gpt-4o-mini",
            temperature=0.1,
            client_name="PetStoreClient",
            async_client=True,
            include_examples=True,
        )
        
        print("✅ Generator configured")
        
        # 2. Generate client
        print("🔄 Generating API client...")
        generator = APIClientGenerator(config)
        swagger_url = "https://petstore.swagger.io/v2/swagger.json"
        
        result = await generator.generate_with_details(swagger_url)
        
        if not result["success"]:
            print(f"❌ Generation failed: {result['error']}")
            return
        
        print("✅ Client generated successfully!")
        
        # 3. Save the client
        output_file = "workflow_test_client.py"
        with open(output_file, "w") as f:
            f.write(result["generated_code"])
        
        print(f"💾 Client saved to: {output_file}")
        
        # 4. Test the client using our testing framework
        print("🧪 Testing generated client...")
        
        try:
            from agentic_api.testing.client_tester import test_client_from_file, print_test_report
            
            test_report = await test_client_from_file(output_file)
            print_test_report(test_report)
            
        except Exception as e:
            print(f"⚠️ Could not run comprehensive tests: {e}")
            print("   Falling back to basic syntax check...")
            
            # Basic syntax test
            try:
                compile(result["generated_code"], output_file, 'exec')
                print("✅ Syntax validation passed")
            except SyntaxError as e:
                print(f"❌ Syntax error: {e}")
        
        # 5. Show generation details
        print(f"\n📊 Generation Analysis:")
        analysis = result.get("analysis", {})
        validation = result.get("validation", {})
        
        print(f"   Code Lines: {len(result['generated_code'].splitlines())}")
        print(f"   Syntax Valid: {validation.get('syntax_valid', 'Unknown')}")
        print(f"   LLM Quality Score: {validation.get('quality_score', 'N/A')}/10")
        
        if "test_report" in validation:
            tr = validation["test_report"]
            print(f"   Overall Test Score: {tr.get('overall_score', 0):.1f}%")
            print(f"   Methods Found: {tr.get('method_count', 0)}")
            print(f"   Models Found: {tr.get('model_count', 0)}")
        
        print(f"\n🎉 Workflow completed successfully!")
        print(f"   Your API client is ready to use in: {output_file}")
        
        # 6. Show usage example
        print(f"\n💡 Usage Example:")
        print(f"```python")
        print(f"from {output_file[:-3]} import PetStoreClient, Pet")
        print(f"")
        print(f"async def main():")
        print(f"    client = PetStoreClient(api_key='your-api-key')")
        print(f"    # Use the client methods...")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")


if __name__ == "__main__":
    asyncio.run(complete_workflow_example())
