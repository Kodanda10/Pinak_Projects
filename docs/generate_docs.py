#!/usr/bin/env python3
"""
Automated Documentation Generation for Pinak.

This script automatically generates comprehensive documentation by:
1. Analyzing codebase structure and dependencies
2. Extracting docstrings and type hints
3. Generating API documentation
4. Creating architecture diagrams
5. Updating test coverage reports
6. Building comprehensive README and guides

Usage:
    python docs/generate_docs.py [--output-dir OUTPUT_DIR] [--include-tests] [--include-coverage]
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class DocumentationGenerator:
    """Main documentation generator class."""

    def __init__(self, project_root: Path, output_dir: Optional[Path] = None):
        self.project_root = project_root
        self.output_dir = output_dir or project_root / "docs" / "generated"
        self.src_dir = project_root / "src"
        self.test_dir = project_root / "tests"

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data structures
        self.modules = {}
        self.classes = {}
        self.functions = {}
        self.test_coverage = {}
        self.dependencies = {}

    def generate_all_docs(self) -> Dict[str, Any]:
        """Generate all documentation components."""
        print("🔄 Starting comprehensive documentation generation...")

        results = {"timestamp": datetime.now().isoformat(), "components": {}}

        # Analyze codebase
        print("📊 Analyzing codebase...")
        results["components"]["codebase_analysis"] = self.analyze_codebase()

        # Generate API documentation
        print("📚 Generating API documentation...")
        results["components"]["api_docs"] = self.generate_api_docs()

        # Generate architecture documentation
        print("🏗️  Generating architecture documentation...")
        results["components"]["architecture"] = self.generate_architecture_docs()

        # Generate test documentation
        print("🧪 Generating test documentation...")
        results["components"]["test_docs"] = self.generate_test_docs()

        # Generate coverage reports
        print("📈 Generating coverage reports...")
        results["components"]["coverage"] = self.generate_coverage_docs()

        # Generate dependency analysis
        print("🔗 Analyzing dependencies...")
        results["components"]["dependencies"] = self.analyze_dependencies()

        # Generate CI/CD documentation
        print("🚀 Generating CI/CD documentation...")
        results["components"]["cicd"] = self.generate_cicd_docs()

        # Update main README
        print("📝 Updating main README...")
        self.update_main_readme(results)

        print("✅ Documentation generation completed!")
        return results

    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the codebase structure."""
        analysis = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "modules": {},
            "complexity": {},
        }

        # Analyze Python files
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            analysis["total_files"] += 1

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    analysis["total_lines"] += lines

                # Parse AST for module analysis
                tree = ast.parse(content)
                self._analyze_module(py_file, tree, analysis)

            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")

        # Language breakdown
        for file in self.project_root.rglob("*"):
            if file.is_file():
                ext = file.suffix.lower()
                analysis["languages"][ext] = analysis["languages"].get(ext, 0) + 1

        return analysis

    def _analyze_module(self, file_path: Path, tree: ast.AST, analysis: Dict[str, Any]):
        """Analyze a single Python module."""
        module_name = (
            file_path.relative_to(self.src_dir).with_suffix("").as_posix().replace("/", ".")
        )

        module_info = {
            "path": str(file_path),
            "classes": [],
            "functions": [],
            "imports": [],
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                module_info["classes"].append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    }
                )
            elif isinstance(node, ast.FunctionDef):
                module_info["functions"].append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args),
                    }
                )
            elif isinstance(node, ast.Import):
                module_info["imports"].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_info["imports"].append(node.module)

        analysis["modules"][module_name] = module_info

    def generate_api_docs(self) -> Dict[str, Any]:
        """Generate API documentation from docstrings."""
        api_docs = {"modules": {}, "classes": {}, "functions": {}}

        # Import and analyze main modules
        main_modules = [
            "pinak.cli",
            "pinak.memory.manager",
            "pinak.context.broker.broker",
            "pinak.security.auditor",
            "pinak.file_quarantine",
        ]

        for module_name in main_modules:
            try:
                module = __import__(module_name, fromlist=[""])
                api_docs["modules"][module_name] = self._extract_module_docs(module)
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")

        return api_docs

    def _extract_module_docs(self, module) -> Dict[str, Any]:
        """Extract documentation from a module."""
        docs = {"docstring": module.__doc__ or "", "classes": {}, "functions": {}}

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                docs["classes"][name] = self._extract_class_docs(obj)
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                docs["functions"][name] = self._extract_function_docs(obj)

        return docs

    def _extract_class_docs(self, cls) -> Dict[str, Any]:
        """Extract documentation from a class."""
        docs = {"docstring": cls.__doc__ or "", "methods": {}}

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                docs["methods"][name] = self._extract_function_docs(method)

        return docs

    def _extract_function_docs(self, func) -> Dict[str, Any]:
        """Extract documentation from a function."""
        try:
            sig = inspect.signature(func)
            return {
                "docstring": func.__doc__ or "",
                "signature": str(sig),
                "parameters": list(sig.parameters.keys()),
                "return_type": (
                    str(sig.return_annotation)
                    if sig.return_annotation != inspect.Signature.empty
                    else None
                ),
            }
        except Exception:
            return {
                "docstring": func.__doc__ or "",
                "signature": "Unknown",
                "parameters": [],
                "return_type": None,
            }

    def generate_architecture_docs(self) -> Dict[str, Any]:
        """Generate architecture documentation."""
        architecture = {
            "components": {},
            "data_flow": {},
            "dependencies": {},
            "design_patterns": [],
        }

        # Analyze main components
        components = {
            "CLI": "pinak.cli",
            "Memory Service": "pinak.memory.manager",
            "Context Broker": "pinak.context.broker.broker",
            "Security Auditor": "pinak.security.auditor",
            "File Quarantine": "pinak.file_quarantine",
        }

        for name, module in components.items():
            try:
                mod = __import__(module, fromlist=[""])
                architecture["components"][name] = {
                    "module": module,
                    "description": (mod.__doc__.split(".")[0] if mod.__doc__ else "No description"),
                    "classes": len(
                        [
                            c
                            for c in dir(mod)
                            if not c.startswith("_") and inspect.isclass(getattr(mod, c))
                        ]
                    ),
                    "functions": len(
                        [
                            f
                            for f in dir(mod)
                            if not f.startswith("_") and inspect.isfunction(getattr(mod, f))
                        ]
                    ),
                }
            except ImportError:
                architecture["components"][name] = {"error": f"Could not import {module}"}

        return architecture

    def generate_test_docs(self) -> Dict[str, Any]:
        """Generate test documentation."""
        test_docs = {
            "test_files": [],
            "test_coverage": {},
            "test_types": {
                "unit": 0,
                "integration": 0,
                "tdd": 0,
                "world_beating": 0,
                "governance": 0,
            },
        }

        # Analyze test files
        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract test functions
                tree = ast.parse(content)
                tests = []
                markers = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        tests.append(node.name)

                        # Check for markers in decorators
                        for decorator in node.decorator_list:
                            if hasattr(decorator, "attr"):
                                markers.append(decorator.attr)

                test_docs["test_files"].append(
                    {
                        "file": str(test_file.relative_to(self.project_root)),
                        "tests": tests,
                        "markers": list(set(markers)),
                        "test_count": len(tests),
                    }
                )

                # Count test types by markers
                for marker in markers:
                    if marker in test_docs["test_types"]:
                        test_docs["test_types"][marker] += 1

            except Exception as e:
                print(f"Warning: Could not analyze test file {test_file}: {e}")

        return test_docs

    def generate_coverage_docs(self) -> Dict[str, Any]:
        """Generate test coverage documentation."""
        coverage_docs = {
            "overall_coverage": 0,
            "file_coverage": {},
            "missing_lines": {},
            "recommendations": [],
        }

        # Try to read coverage data if available
        coverage_file = self.project_root / "coverage.xml"
        if coverage_file.exists():
            try:
                # Parse coverage XML (simplified)
                with open(coverage_file, "r") as f:
                    content = f.read()

                # Extract basic coverage info (simplified parsing)
                coverage_match = re.search(r'line-rate="([0-9.]+)"', content)
                if coverage_match:
                    coverage_docs["overall_coverage"] = float(coverage_match.group(1)) * 100

            except Exception as e:
                print(f"Warning: Could not parse coverage data: {e}")

        return coverage_docs

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        deps = {
            "python_packages": {},
            "system_dependencies": [],
            "optional_dependencies": {},
        }

        # Read requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if ">=" in line or "==" in line:
                                pkg = line.split()[0]
                                deps["python_packages"][pkg] = line
                            else:
                                deps["python_packages"][line] = line
            except Exception as e:
                print(f"Warning: Could not read requirements.txt: {e}")

        return deps

    def generate_cicd_docs(self) -> Dict[str, Any]:
        """Generate CI/CD documentation."""
        cicd_docs = {"workflows": {}, "jobs": {}, "triggers": [], "status": "unknown"}

        # Analyze GitHub Actions workflows
        workflows_dir = self.project_root / ".github" / "workflows"
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob("*.yml"):
                try:
                    with open(workflow_file, "r") as f:
                        content = f.read()

                    workflow_name = workflow_file.stem
                    cicd_docs["workflows"][workflow_name] = {
                        "file": str(workflow_file.relative_to(self.project_root)),
                        "triggers": self._extract_workflow_triggers(content),
                        "jobs": self._extract_workflow_jobs(content),
                    }

                except Exception as e:
                    print(f"Warning: Could not analyze workflow {workflow_file}: {e}")

        return cicd_docs

    def _extract_workflow_triggers(self, content: str) -> List[str]:
        """Extract workflow triggers from YAML content."""
        triggers = []
        lines = content.split("\n")
        in_on_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("on:"):
                in_on_section = True
                continue
            elif in_on_section and line.startswith("  - ") or line.startswith("    - "):
                triggers.append(line.strip(" -"))
            elif in_on_section and not line.startswith(" ") and line.endswith(":"):
                break

        return triggers

    def _extract_workflow_jobs(self, content: str) -> List[str]:
        """Extract workflow jobs from YAML content."""
        jobs = []
        if "jobs:" in content:
            # Simple extraction - could be improved with proper YAML parsing
            jobs_section = content.split("jobs:")[1]
            job_matches = re.findall(r"^  ([a-zA-Z_-]+):", jobs_section, re.MULTILINE)
            jobs.extend(job_matches)

        return jobs

    def update_main_readme(self, results: Dict[str, Any]):
        """Update the main README with generated documentation."""
        readme_file = self.project_root / "README.md"

        if not readme_file.exists():
            return

        try:
            with open(readme_file, "r") as f:
                content = f.read()

            # Update timestamp
            timestamp = results["timestamp"]
            content = re.sub(r"Last updated: .*", f"Last updated: {timestamp}", content)

            # Update component counts
            if "codebase_analysis" in results["components"]:
                analysis = results["components"]["codebase_analysis"]
                content = re.sub(
                    r"(\d+) Python files",
                    f'{analysis["total_files"]} Python files',
                    content,
                )

            # Write back
            with open(readme_file, "w") as f:
                f.write(content)

        except Exception as e:
            print(f"Warning: Could not update README: {e}")

    def save_docs(self, results: Dict[str, Any]):
        """Save generated documentation to files."""
        # Save JSON summary
        summary_file = self.output_dir / "documentation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Generate HTML documentation
        self._generate_html_docs(results)

        # Generate Markdown documentation
        self._generate_markdown_docs(results)

    def _generate_html_docs(self, results: Dict[str, Any]):
        """Generate HTML documentation."""
        html_file = self.output_dir / "api_documentation.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pinak API Documentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .module {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .class {{ background: #e8f4f8; padding: 10px; margin: 5px 0; border-left: 3px solid #4a90e2; }}
                .function {{ background: #f0f8e8; padding: 8px; margin: 3px 0; border-left: 3px solid #5cb85c; }}
                h1, h2, h3 {{ color: #333; }}
                pre {{ background: #f8f8f8; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>🔍 Pinak API Documentation</h1>
            <p>Generated on: {results['timestamp']}</p>

            <h2>📊 Codebase Overview</h2>
            <div class="module">
                <h3>Statistics</h3>
                <p>Total Files: {results['components'].get('codebase_analysis', {}).get('total_files', 'N/A')}</p>
                <p>Total Lines: {results['components'].get('codebase_analysis', {}).get('total_lines', 'N/A')}</p>
            </div>

            <h2>🏗️ Architecture</h2>
        """

        # Add architecture components
        if "architecture" in results["components"]:
            for name, info in results["components"]["architecture"]["components"].items():
                html_content += f"""
                <div class="module">
                    <h3>{name}</h3>
                    <p>{info.get('description', 'No description')}</p>
                    <p>Classes: {info.get('classes', 0)}, Functions: {info.get('functions', 0)}</p>
                </div>
                """

        html_content += """
            <h2>🧪 Testing</h2>
        """

        # Add test information
        if "test_docs" in results["components"]:
            test_info = results["components"]["test_docs"]
            html_content += f"""
            <div class="module">
                <h3>Test Overview</h3>
                <p>Test Files: {len(test_info.get('test_files', []))}</p>
                <p>Test Types: {test_info.get('test_types', {})}</p>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(html_file, "w") as f:
            f.write(html_content)

    def _generate_markdown_docs(self, results: Dict[str, Any]):
        """Generate Markdown documentation."""
        md_file = self.output_dir / "API_DOCUMENTATION.md"

        md_content = f"""# 🔍 Pinak API Documentation

Generated on: {results['timestamp']}

## 📊 Codebase Overview

- **Total Files:** {results['components'].get('codebase_analysis', {}).get('total_files', 'N/A')}
- **Total Lines:** {results['components'].get('codebase_analysis', {}).get('total_lines', 'N/A')}

## 🏗️ Architecture Components

"""

        # Add architecture components
        if "architecture" in results["components"]:
            for name, info in results["components"]["architecture"]["components"].items():
                md_content += f"""### {name}
- **Module:** {info.get('module', 'N/A')}
- **Description:** {info.get('description', 'No description')}
- **Classes:** {info.get('classes', 0)}
- **Functions:** {info.get('functions', 0)}

"""

        md_content += """
## 🧪 Testing Overview

"""

        # Add test information
        if "test_docs" in results["components"]:
            test_info = results["components"]["test_docs"]
            md_content += f"""- **Test Files:** {len(test_info.get('test_files', []))}
- **Test Types:** {test_info.get('test_types', {})}

### Test Files
"""

            for test_file in test_info.get("test_files", []):
                md_content += f"""#### {test_file['file']}
- **Tests:** {test_file['test_count']}
- **Markers:** {', '.join(test_file['markers'])}

"""

        with open(md_file, "w") as f:
            f.write(md_content)


def main():
    """Main entry point for documentation generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Pinak documentation")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for generated documentation"
    )
    parser.add_argument("--include-tests", action="store_true", help="Include test documentation")
    parser.add_argument("--include-coverage", action="store_true", help="Include coverage reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Find project root
    project_root = Path.cwd()
    while not (project_root / "setup.py").exists() and project_root != project_root.parent:
        project_root = project_root.parent

    if not (project_root / "setup.py").exists():
        print("Error: Could not find project root (setup.py not found)")
        sys.exit(1)

    # Generate documentation
    generator = DocumentationGenerator(project_root, args.output_dir)

    try:
        results = generator.generate_all_docs()
        generator.save_docs(results)

        print("✅ Documentation generation completed successfully!")
        print(f"📁 Generated files saved to: {generator.output_dir}")

        # Print summary
        print("\n📊 Summary:")
        for component, status in results["components"].items():
            if isinstance(status, dict):
                print(f"  ✅ {component}: {len(status)} items")
            else:
                print(f"  ✅ {component}: generated")

    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
