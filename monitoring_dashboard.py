#!/usr/bin/env python3
"""
Pinak CI/CD Monitoring Dashboard

Real-time monitoring and status tracking for the Pinak build setup.
Provides comprehensive visibility into CI/CD pipeline health and progress.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import urllib.request
import urllib.error


class PinakMonitoringDashboard:
    """Comprehensive monitoring dashboard for Pinak CI/CD pipeline."""

    def __init__(self, repo_owner: str = "Pinak-Setu", repo_name: str = "Pinak_Projects"):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.token = os.getenv("GITHUB_TOKEN")

        # Headers for API requests
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Pinak-Monitoring-Dashboard/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def get_workflow_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow runs from GitHub API."""
        try:
            url = f"{self.base_url}/actions/runs?per_page={limit}"
            req = urllib.request.Request(url, headers=self.headers)

            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                return data.get("workflow_runs", [])

        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error getting workflow runs: {e.code}")
            return []
        except Exception as e:
            print(f"❌ Error getting workflow runs: {e}")
            return []

    def get_workflow_status(self, workflow_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze workflow run status and provide comprehensive metrics."""
        status = {
            "total_runs": len(workflow_runs),
            "by_status": {},
            "by_conclusion": {},
            "by_workflow": {},
            "recent_activity": [],
            "health_score": 0,
            "issues": [],
        }

        if not workflow_runs:
            status["issues"].append("No workflow runs found")
            return status

        # Analyze runs
        for run in workflow_runs:
            # Status analysis
            run_status = run.get("status", "unknown")
            run_conclusion = run.get("conclusion", "unknown")
            workflow_name = run.get("name", "Unknown Workflow")

            # Count by status
            status["by_status"][run_status] = status["by_status"].get(run_status, 0) + 1

            # Count by conclusion
            if run_conclusion:
                status["by_conclusion"][run_conclusion] = (
                    status["by_conclusion"].get(run_conclusion, 0) + 1
                )

            # Count by workflow
            status["by_workflow"][workflow_name] = status["by_workflow"].get(workflow_name, 0) + 1

            # Recent activity (last 5 runs)
            if len(status["recent_activity"]) < 5:
                status["recent_activity"].append(
                    {
                        "name": workflow_name,
                        "status": run_status,
                        "conclusion": run_conclusion,
                        "created_at": run.get("created_at", ""),
                        "html_url": run.get("html_url", ""),
                    }
                )

        # Calculate health score
        if status["by_conclusion"]:
            success_count = status["by_conclusion"].get("success", 0)
            failure_count = status["by_conclusion"].get("failure", 0)
            total_completed = success_count + failure_count

            if total_completed > 0:
                success_rate = success_count / total_completed
                status["health_score"] = int(success_rate * 100)

                # Health assessment
                if success_rate >= 0.9:
                    status["health_status"] = "🟢 EXCELLENT"
                elif success_rate >= 0.8:
                    status["health_status"] = "🟡 GOOD"
                elif success_rate >= 0.7:
                    status["health_status"] = "🟠 FAIR"
                else:
                    status["health_status"] = "🔴 POOR"
                    status["issues"].append(f"Low success rate: {success_rate:.1%}")
            else:
                status["health_status"] = "⚪ UNKNOWN"
                status["issues"].append("No completed runs to analyze")
        else:
            status["health_status"] = "⚪ UNKNOWN"
            status["issues"].append("No workflow data available")

        return status

    def get_test_coverage_status(self) -> Dict[str, Any]:
        """Get test coverage information."""
        coverage_status = {
            "coverage_file_exists": False,
            "coverage_percentage": 0,
            "lines_covered": 0,
            "lines_total": 0,
            "status": "unknown",
        }

        # Check for coverage files
        coverage_files = [Path("coverage.xml"), Path("htmlcov/index.html"), Path("coverage.json")]

        for coverage_file in coverage_files:
            if coverage_file.exists():
                coverage_status["coverage_file_exists"] = True

                try:
                    if coverage_file.suffix == ".xml":
                        # Parse XML coverage (simplified)
                        with open(coverage_file, "r") as f:
                            content = f.read()
                            # Extract basic coverage info
                            import re

                            match = re.search(r'line-rate="([0-9.]+)"', content)
                            if match:
                                coverage_status["coverage_percentage"] = float(match.group(1)) * 100

                    elif coverage_file.suffix == ".json":
                        with open(coverage_file, "r") as f:
                            data = json.load(f)
                            coverage_status["coverage_percentage"] = data.get("totals", {}).get(
                                "percent_covered", 0
                            )

                except Exception as e:
                    print(f"Warning: Could not parse coverage file {coverage_file}: {e}")

        # Determine status
        if coverage_status["coverage_percentage"] >= 80:
            coverage_status["status"] = "🟢 EXCELLENT"
        elif coverage_status["coverage_percentage"] >= 70:
            coverage_status["status"] = "🟡 GOOD"
        elif coverage_status["coverage_percentage"] >= 60:
            coverage_status["status"] = "🟠 FAIR"
        else:
            coverage_status["status"] = "🔴 POOR"

        return coverage_status

    def get_documentation_status(self) -> Dict[str, Any]:
        """Check documentation generation status."""
        docs_status = {
            "generated_docs_exist": False,
            "api_docs_exist": False,
            "html_docs_exist": False,
            "markdown_docs_exist": False,
            "last_update": None,
            "status": "unknown",
        }

        # Check for documentation files
        docs_dir = Path("docs")
        if docs_dir.exists():
            generated_dir = docs_dir / "generated"
            if generated_dir.exists():
                docs_status["generated_docs_exist"] = True

                # Check specific files
                api_html = generated_dir / "api_documentation.html"
                api_md = generated_dir / "API_DOCUMENTATION.md"
                summary_json = generated_dir / "documentation_summary.json"

                docs_status["api_docs_exist"] = api_html.exists()
                docs_status["html_docs_exist"] = api_html.exists()
                docs_status["markdown_docs_exist"] = api_md.exists()

                # Check last update
                if summary_json.exists():
                    try:
                        with open(summary_json, "r") as f:
                            data = json.load(f)
                            docs_status["last_update"] = data.get("timestamp")
                    except Exception:
                        pass

        # Determine status
        doc_files_exist = sum(
            [
                docs_status["api_docs_exist"],
                docs_status["html_docs_exist"],
                docs_status["markdown_docs_exist"],
            ]
        )

        if doc_files_exist >= 2:
            docs_status["status"] = "🟢 COMPLETE"
        elif doc_files_exist >= 1:
            docs_status["status"] = "🟡 PARTIAL"
        else:
            docs_status["status"] = "🔴 MISSING"

        return docs_status

    def display_dashboard(self):
        """Display the comprehensive monitoring dashboard."""
        print("\n" + "=" * 80)
        print("🎯 PINAK CI/CD MONITORING DASHBOARD")
        print("=" * 80)
        print(f"📊 Repository: {self.repo_owner}/{self.repo_name}")
        print(f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

        # Get workflow status
        workflow_runs = self.get_workflow_runs()
        workflow_status = self.get_workflow_status(workflow_runs)

        # CI/CD Status Section
        print("🚀 CI/CD PIPELINE STATUS")
        print("-" * 40)
        print(f"Total Runs: {workflow_status['total_runs']}")
        print(
            f"Health Score: {workflow_status['health_score']}% - {workflow_status['health_status']}"
        )

        if workflow_status["by_status"]:
            print("\n📈 Status Breakdown:")
            for status, count in workflow_status["by_status"].items():
                print(f"  {status.title()}: {count}")

        if workflow_status["by_conclusion"]:
            print("\n✅ Conclusion Breakdown:")
            for conclusion, count in workflow_status["by_conclusion"].items():
                icon = (
                    "✅" if conclusion == "success" else "❌" if conclusion == "failure" else "⏳"
                )
                print(f"  {icon} {conclusion.title()}: {count}")

        if workflow_status["issues"]:
            print("\n⚠️  Issues Detected:")
            for issue in workflow_status["issues"]:
                print(f"  • {issue}")

        # Recent Activity
        if workflow_status["recent_activity"]:
            print("\n🔥 Recent Activity:")
            for activity in workflow_status["recent_activity"]:
                status_icon = (
                    "✅"
                    if activity["conclusion"] == "success"
                    else "❌" if activity["conclusion"] == "failure" else "⏳"
                )
                print(f"  {status_icon} {activity['name']} - {activity['status']}")

        # Test Coverage Section
        print("\n" + "-" * 40)
        print("🧪 TEST COVERAGE STATUS")
        print("-" * 40)

        coverage_status = self.get_test_coverage_status()
        print(
            f"Coverage File: {'✅ Found' if coverage_status['coverage_file_exists'] else '❌ Missing'}"
        )
        print(
            f"Coverage: {coverage_status['coverage_percentage']:.1f}% - {coverage_status['status']}"
        )

        # Documentation Section
        print("\n" + "-" * 40)
        print("📚 DOCUMENTATION STATUS")
        print("-" * 40)

        docs_status = self.get_documentation_status()
        print(
            f"Generated Docs: {'✅ Exist' if docs_status['generated_docs_exist'] else '❌ Missing'}"
        )
        print(f"API Docs: {'✅ Available' if docs_status['api_docs_exist'] else '❌ Missing'}")
        print(f"HTML Docs: {'✅ Available' if docs_status['html_docs_exist'] else '❌ Missing'}")
        print(
            f"Markdown Docs: {'✅ Available' if docs_status['markdown_docs_exist'] else '❌ Missing'}"
        )
        print(f"Status: {docs_status['status']}")

        if docs_status["last_update"]:
            print(f"Last Update: {docs_status['last_update']}")

        # System Health Summary
        print("\n" + "=" * 80)
        print("🏥 SYSTEM HEALTH SUMMARY")
        print("=" * 80)

        health_components = {
            "CI/CD Pipeline": workflow_status["health_status"],
            "Test Coverage": coverage_status["status"],
            "Documentation": docs_status["status"],
        }

        all_healthy = all(
            "🟢" in status or "EXCELLENT" in status for status in health_components.values()
        )

        for component, status in health_components.items():
            print(f"{component}: {status}")

        print()
        if all_healthy:
            print("🎉 ALL SYSTEMS OPERATIONAL - READY FOR PRODUCTION! 🚀")
        else:
            print("⚠️  SOME SYSTEMS NEED ATTENTION")

        print("\n" + "=" * 80)

        # Next Steps
        print("🎯 NEXT STEPS:")
        if not all_healthy:
            print("• Fix any failing CI/CD jobs")
            print("• Address code quality issues")
            print("• Improve test coverage if below 80%")
            print("• Generate missing documentation")

        print("• Monitor GitHub Actions for real-time status")
        print("• Review detailed logs for any failures")
        print("• Check PR status for merge readiness")
        print("\n🔗 Useful Links:")
        print(f"• GitHub Actions: https://github.com/{self.repo_owner}/{self.repo_name}/actions")
        print(f"• Repository: https://github.com/{self.repo_owner}/{self.repo_name}")
        if docs_status["html_docs_exist"]:
            print(f"• Documentation: https://github.com/{self.repo_owner}/{self.repo_name}/docs")
        print()

    def monitor_realtime(self, interval_seconds: int = 60):
        """Monitor in real-time with periodic updates."""
        print("🔄 Starting real-time monitoring... (Ctrl+C to stop)")
        print(f"Update interval: {interval_seconds} seconds")
        print()

        try:
            while True:
                # Clear screen (Unix systems)
                print("\033[H\033[J", end="")

                self.display_dashboard()

                print(f"\n⏰ Next update in {interval_seconds} seconds...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

    def export_status(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Export comprehensive status to JSON file."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "repository": f"{self.repo_owner}/{self.repo_name}",
            "ci_cd": self.get_workflow_status(self.get_workflow_runs()),
            "test_coverage": self.get_test_coverage_status(),
            "documentation": self.get_documentation_status(),
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
            print(f"✅ Status exported to {output_file}")

        return status


def main():
    """Main entry point for the monitoring dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Pinak CI/CD Monitoring Dashboard")
    parser.add_argument("--repo-owner", default="Pinak-Setu", help="Repository owner")
    parser.add_argument("--repo-name", default="Pinak_Projects", help="Repository name")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time monitoring")
    parser.add_argument(
        "--interval", type=int, default=60, help="Real-time update interval (seconds)"
    )
    parser.add_argument("--export", type=str, help="Export status to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less verbose)")

    args = parser.parse_args()

    dashboard = PinakMonitoringDashboard(args.repo_owner, args.repo_name)

    if args.export:
        dashboard.export_status(args.export)
    elif args.realtime:
        dashboard.monitor_realtime(args.interval)
    else:
        dashboard.display_dashboard()


if __name__ == "__main__":
    main()
