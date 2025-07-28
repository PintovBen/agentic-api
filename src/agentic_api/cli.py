"""Command line interface for the agentic API generator."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agentic_api.core.config import AgentConfig
from agentic_api.core.generator import APIClientGenerator

app = typer.Typer(help="AI Agent for generating Python API clients from any documentation source")
console = Console()


@app.command()
def generate(
    source: str = typer.Argument(..., help="Documentation source: URL (Swagger/webpage), file path, or raw text"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory or file path"),
    client_name: Optional[str] = typer.Option(None, "--client-name", help="Name for the generated client class"),
    async_client: bool = typer.Option(False, "--async-client", help="Generate async client"),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    temperature: float = typer.Option(0.1, "--temperature", help="LLM temperature (0.0-2.0)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate a Python API client from any documentation source.
    
    SUPPORTED SOURCES:
    - Swagger/OpenAPI JSON/YAML URLs (https://api.example.com/swagger.json)
    - Documentation webpages (https://docs.example.com/api)  
    - Local files (.json, .yaml, .md, .txt, .html)
    - Raw text documentation
    
    EXAMPLES:
    - agentic-api generate "https://petstore.swagger.io/v2/swagger.json"
    - agentic-api generate "https://docs.openweathermap.org/api"
    - agentic-api generate "./api-docs.md"
    - agentic-api generate "API has endpoints: GET /users, POST /users"
    """
    
    # Create configuration
    config = AgentConfig(
        model_name=model,
        temperature=temperature,
        client_name=client_name,
        async_client=async_client,
    )
    
    # Run the generation
    asyncio.run(_generate_client(source, output, config, verbose))


async def _generate_client(
    source: str, 
    output: Optional[str], 
    config: AgentConfig, 
    verbose: bool
):
    """Internal function to generate the client."""
    
    try:
        generator = APIClientGenerator(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Determine source type for progress message
            if source.startswith(('http://', 'https://')):
                task_desc = f"Generating from URL: {source[:50]}..."
            elif '\n' in source or len(source) > 200:
                task_desc = "Generating from text documentation..."
            else:
                task_desc = f"Generating from: {source}"
            
            task = progress.add_task(task_desc, total=None)
            
            if verbose:
                result = await generator.generate_with_details(source)
                
                console.print("\n[bold green]✓[/bold green] Generation completed!")
                console.print(f"[dim]Input Type:[/dim] {result.get('input_type', 'unknown')}")
                console.print(f"[dim]Analysis:[/dim] {len(result['analysis'])} insights")
                console.print(f"[dim]Validation:[/dim] {result['validation'].get('quality_score', 'N/A')}/10")
                
                generated_code = result["generated_code"]
            else:
                generated_code = await generator.generate_from_source(source)
                console.print("\n[bold green]✓[/bold green] Generation completed!")
        
        # Determine output path
        if output:
            output_path = Path(output)
            if output_path.is_dir():
                # If directory, create filename from client name or detected API name
                client_name = config.client_name or "api_client"
                filename = f"{client_name.lower().replace(' ', '_')}.py"
                output_path = output_path / filename
        else:
            # Default output
            client_name = config.client_name or "api_client"
            output_path = Path(f"{client_name.lower().replace(' ', '_')}.py")
        
        # Write the generated code
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(generated_code)
        
        console.print(f"[bold blue]📄[/bold blue] Client saved to: {output_path}")
        console.print(f"[dim]Lines of code:[/dim] {len(generated_code.splitlines())}")
        
        # Show usage hint
        console.print(f"\n[dim]Usage:[/dim] python {output_path.name}")
        
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def test(
    client_file: str = typer.Argument(..., help="Path to generated client file to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Test a generated API client for correctness and functionality."""
    
    from pathlib import Path
    
    client_path = Path(client_file)
    if not client_path.exists():
        console.print(f"[red]✗[/red] File not found: {client_file}")
        raise typer.Exit(1)
    
    # Run the test
    asyncio.run(_test_client(client_path, verbose))


async def _test_client(client_path: Path, verbose: bool):
    """Internal function to test a client."""
    
    try:
        from agentic_api.testing.client_tester import test_client_from_file, print_test_report
        
        console.print(f"[blue]🧪[/blue] Testing client: {client_path.name}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Running tests...", total=None)
            report = await test_client_from_file(str(client_path))
        
        # Display results
        if verbose or report.overall_score < 80:
            print_test_report(report)
        else:
            console.print(f"\n[green]✓[/green] Test completed!")
            console.print(f"[dim]Overall Score:[/dim] {report.overall_score:.1f}%")
            console.print(f"[dim]Syntax Valid:[/dim] {'✅' if report.syntax_valid else '❌'}")
            console.print(f"[dim]Methods Found:[/dim] {report.method_count}")
            console.print(f"[dim]Models Found:[/dim] {report.model_count}")
        
        # Exit with error if tests failed
        if report.overall_score < 50:
            console.print(f"[red]⚠️[/red] Client failed tests (score: {report.overall_score:.1f}%)")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error testing client: {str(e)}")
        raise typer.Exit(1)


@app.command()
def config():
    """Show current configuration and environment setup."""
    
    console.print("[bold]Agentic API Configuration[/bold]\n")
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        console.print(f"[green]✓[/green] OPENAI_API_KEY: {'*' * (len(openai_key) - 4) + openai_key[-4:]}")
    else:
        console.print("[red]✗[/red] OPENAI_API_KEY: Not set")
    
    # Show default config
    config = AgentConfig.from_env()
    console.print(f"\n[dim]Default Model:[/dim] {config.model_name}")
    console.print(f"[dim]Temperature:[/dim] {config.temperature}")
    console.print(f"[dim]Max Tokens:[/dim] {config.max_tokens}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
