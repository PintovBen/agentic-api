"""
Comprehensive testing framework for generated API clients.
"""

import ast
import asyncio
import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel


class TestResult(BaseModel):
    """Result of a single test."""
    test_name: str
    passed: bool
    error: Optional[str] = None
    details: Optional[str] = None


class ClientTestReport(BaseModel):
    """Complete test report for a generated client."""
    client_name: str
    syntax_valid: bool
    imports_valid: bool
    has_main_class: bool
    method_count: int
    model_count: int
    test_results: List[TestResult]
    overall_score: float
    recommendations: List[str]


class GeneratedClientTester:
    """Tests generated API clients for correctness and functionality."""
    
    def __init__(self):
        self.temp_dir = None
        self.client_module = None
    
    async def test_generated_client(self, generated_code: str, client_name: str = "GeneratedClient") -> ClientTestReport:
        """
        Comprehensive testing of a generated API client.
        
        Args:
            generated_code: The generated Python code
            client_name: Name of the client being tested
            
        Returns:
            Complete test report
        """
        test_results = []
        recommendations = []
        
        # 1. Syntax validation
        syntax_valid, syntax_error = self._test_syntax(generated_code)
        if not syntax_valid:
            test_results.append(TestResult(
                test_name="syntax_validation",
                passed=False,
                error=syntax_error
            ))
            return ClientTestReport(
                client_name=client_name,
                syntax_valid=False,
                imports_valid=False,
                has_main_class=False,
                method_count=0,
                model_count=0,
                test_results=test_results,
                overall_score=0.0,
                recommendations=["Fix syntax errors before proceeding"]
            )
        
        test_results.append(TestResult(
            test_name="syntax_validation",
            passed=True,
            details="Code has valid Python syntax"
        ))
        
        # 2. Import validation
        imports_valid, import_error, client_module = await self._test_imports(generated_code)
        test_results.append(TestResult(
            test_name="import_validation",
            passed=imports_valid,
            error=import_error,
            details="All imports are resolvable" if imports_valid else None
        ))
        
        if not imports_valid:
            recommendations.append("Check that all required dependencies are installed")
        
        # 3. Structure analysis
        structure_info = self._analyze_structure(generated_code)
        
        test_results.append(TestResult(
            test_name="class_detection",
            passed=structure_info["has_main_class"],
            details=f"Found main class: {structure_info['main_class_name']}" if structure_info["has_main_class"] else "No main client class found"
        ))
        
        test_results.append(TestResult(
            test_name="method_analysis",
            passed=structure_info["method_count"] > 0,
            details=f"Found {structure_info['method_count']} methods"
        ))
        
        test_results.append(TestResult(
            test_name="model_analysis",
            passed=structure_info["model_count"] > 0,
            details=f"Found {structure_info['model_count']} Pydantic models"
        ))
        
        # 4. Functional tests (if imports work)
        if imports_valid and client_module:
            func_results = await self._test_functionality(client_module, structure_info)
            test_results.extend(func_results)
        
        # 5. Quality checks
        quality_results = self._test_code_quality(generated_code)
        test_results.extend(quality_results)
        
        # Calculate overall score
        passed_tests = sum(1 for result in test_results if result.passed)
        overall_score = (passed_tests / len(test_results)) * 100 if test_results else 0
        
        # Generate recommendations
        if overall_score < 70:
            recommendations.append("Consider regenerating the client with improved prompts")
        if structure_info["method_count"] < 3:
            recommendations.append("Client has few methods - verify Swagger parsing")
        if not structure_info["has_error_handling"]:
            recommendations.append("Add proper error handling to the client")
        
        return ClientTestReport(
            client_name=client_name,
            syntax_valid=syntax_valid,
            imports_valid=imports_valid,
            has_main_class=structure_info["has_main_class"],
            method_count=structure_info["method_count"],
            model_count=structure_info["model_count"],
            test_results=test_results,
            overall_score=overall_score,
            recommendations=recommendations
        )
    
    def _test_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Test if the code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error on line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    async def _test_imports(self, code: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Test if all imports can be resolved."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Try to import the module
            spec = importlib.util.spec_from_file_location("test_client", temp_file)
            module = importlib.util.module_from_spec(spec)
            
            # Execute the module to check imports
            spec.loader.exec_module(module)
            
            # Clean up
            Path(temp_file).unlink()
            
            return True, None, module
            
        except ImportError as e:
            return False, f"Import error: {str(e)}", None
        except Exception as e:
            return False, f"Module loading error: {str(e)}", None
    
    def _analyze_structure(self, code: str) -> Dict[str, Any]:
        """Analyze the structure of the generated code."""
        try:
            tree = ast.parse(code)
            
            classes = []
            methods = []
            models = []
            main_class = None
            has_error_handling = False
            has_async = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
                    # Check if it's a Pydantic model
                    for base in node.bases:
                        if (isinstance(base, ast.Name) and base.id == "BaseModel") or \
                           (isinstance(base, ast.Attribute) and base.attr == "BaseModel"):
                            models.append(node.name)
                            break
                    else:
                        # Likely the main client class
                        if not main_class or "client" in node.name.lower():
                            main_class = node.name
                
                elif isinstance(node, ast.FunctionDef):
                    methods.append(node.name)
                    if node.name.startswith('__'):
                        continue  # Skip magic methods
                    
                elif isinstance(node, ast.AsyncFunctionDef):
                    methods.append(node.name)
                    has_async = True
                
                # Check for error handling patterns
                elif isinstance(node, (ast.Try, ast.Raise)):
                    has_error_handling = True
                elif isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and 
                        node.func.attr == "raise_for_status"):
                        has_error_handling = True
            
            return {
                "classes": classes,
                "methods": methods,
                "models": models,
                "main_class_name": main_class,
                "has_main_class": main_class is not None,
                "method_count": len([m for m in methods if not m.startswith('_')]),
                "model_count": len(models),
                "has_error_handling": has_error_handling,
                "has_async": has_async
            }
            
        except Exception as e:
            return {
                "classes": [],
                "methods": [],
                "models": [],
                "main_class_name": None,
                "has_main_class": False,
                "method_count": 0,
                "model_count": 0,
                "has_error_handling": False,
                "has_async": False
            }
    
    async def _test_functionality(self, module: Any, structure_info: Dict) -> List[TestResult]:
        """Test basic functionality of the client."""
        results = []
        
        if not structure_info["main_class_name"]:
            return [TestResult(
                test_name="functionality_test",
                passed=False,
                error="No main client class found"
            )]
        
        try:
            # Get the main client class
            client_class = getattr(module, structure_info["main_class_name"])
            
            # Test instantiation
            try:
                client_instance = client_class()
                results.append(TestResult(
                    test_name="client_instantiation",
                    passed=True,
                    details="Client can be instantiated"
                ))
                
                # Test method signatures
                method_count = 0
                for attr_name in dir(client_instance):
                    if not attr_name.startswith('_'):
                        attr = getattr(client_instance, attr_name)
                        if callable(attr):
                            method_count += 1
                            # Check if method has proper signature
                            sig = inspect.signature(attr)
                            has_params = len(sig.parameters) > 0
                            
                results.append(TestResult(
                    test_name="method_signatures",
                    passed=method_count > 0,
                    details=f"Found {method_count} callable methods"
                ))
                
            except Exception as e:
                results.append(TestResult(
                    test_name="client_instantiation",
                    passed=False,
                    error=f"Failed to instantiate client: {str(e)}"
                ))
        
        except Exception as e:
            results.append(TestResult(
                test_name="functionality_test",
                passed=False,
                error=f"Failed to test functionality: {str(e)}"
            ))
        
        return results
    
    def _test_code_quality(self, code: str) -> List[TestResult]:
        """Test code quality aspects."""
        results = []
        
        # Check for docstrings
        has_docstrings = '"""' in code or "'''" in code
        results.append(TestResult(
            test_name="documentation",
            passed=has_docstrings,
            details="Code includes docstrings" if has_docstrings else "No docstrings found"
        ))
        
        # Check for type hints
        has_type_hints = ":" in code and ("str" in code or "int" in code or "Optional" in code)
        results.append(TestResult(
            test_name="type_hints",
            passed=has_type_hints,
            details="Code includes type hints" if has_type_hints else "No type hints found"
        ))
        
        # Check for proper imports
        required_imports = ["httpx", "pydantic"]
        has_required_imports = all(imp in code for imp in required_imports)
        results.append(TestResult(
            test_name="required_imports",
            passed=has_required_imports,
            details="All required imports present" if has_required_imports else "Missing required imports"
        ))
        
        return results


async def test_client_from_file(file_path: str) -> ClientTestReport:
    """Test a generated client from a file."""
    tester = GeneratedClientTester()
    
    with open(file_path, 'r') as f:
        code = f.read()
    
    client_name = Path(file_path).stem
    return await tester.test_generated_client(code, client_name)


def print_test_report(report: ClientTestReport):
    """Print a formatted test report."""
    print(f"\n{'='*60}")
    print(f"TEST REPORT: {report.client_name}")
    print(f"{'='*60}")
    print(f"Overall Score: {report.overall_score:.1f}%")
    print(f"Syntax Valid: {'✅' if report.syntax_valid else '❌'}")
    print(f"Imports Valid: {'✅' if report.imports_valid else '❌'}")
    print(f"Has Main Class: {'✅' if report.has_main_class else '❌'}")
    print(f"Methods Found: {report.method_count}")
    print(f"Models Found: {report.model_count}")
    
    print(f"\nDetailed Test Results:")
    print(f"{'Test Name':<25} {'Status':<8} {'Details'}")
    print(f"{'-'*60}")
    
    for result in report.test_results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        details = result.details or result.error or ""
        print(f"{result.test_name:<25} {status:<8} {details}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        report = asyncio.run(test_client_from_file(file_path))
        print_test_report(report)
    else:
        print("Usage: python client_tester.py <path_to_generated_client.py>")
