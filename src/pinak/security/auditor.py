import json
import os
import re
import datetime
import subprocess

class SecurityAuditor:
    """A local security auditing tool, using pip-audit for dependency scanning."""

    def __init__(self, config_path='../config/security_config.json'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # If absolute path provided, use it; else resolve relative to package
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, '..', '..', '..', 'config', 'security_config.json')
        self.config = self._load_config(config_path)

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def scan_file_for_secrets(self, file_path):
        """Scans a single file for hardcoded secrets based on regex patterns."""
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    for rule_name, pattern in self.config['scan_rules'].items():
                        if re.search(pattern, line):
                            findings.append({
                                'type': 'secret',
                                'file': file_path,
                                'line': i + 1,
                                'rule': rule_name,
                                'finding': line.strip()
                            })
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        return findings

    def check_dependencies(self, requirements_path=None):
        """Checks dependencies using pip-audit. If a requirements path is provided and exists, scan that file; otherwise scan current environment."""
        findings = []
        try:
            if requirements_path and os.path.exists(requirements_path):
                command = ["pip-audit", "-r", requirements_path, "--format", "json"]
            else:
                command = ["pip-audit", "--format", "json"]

            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.stdout:
                data = json.loads(result.stdout)
                # pip-audit JSON is a list of packages with potential vulns
                if isinstance(data, list):
                    for pkg in data:
                        name = pkg.get('name')
                        version = pkg.get('version')
                        for vuln in pkg.get('vulns', []) or []:
                            findings.append({
                                'type': 'dependency',
                                'dependency': name,
                                'version': version,
                                'vuln_id': vuln.get('id'),
                                'aliases': vuln.get('aliases', []),
                                'description': vuln.get('description'),
                                'fix_versions': vuln.get('fix_versions', []),
                            })
            return findings
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error running pip-audit: {e}")
            return []

    def generate_report(self, findings, report_type='full_scan'):
        """Generates a timestamped JSON report of findings."""
        reports_dir = self.config['reports_dir']
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_dir, f"report_{report_type}_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(findings, f, indent=2)
        
        print(f"Report generated at {report_path}")
        return report_path