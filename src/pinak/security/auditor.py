
import json
import os
import re
import datetime
import requests

class SecurityAuditor:
    """A local security auditing tool."""

    def __init__(self, config_path='config/security_config.json'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Use provided path if absolute; otherwise resolve relative to package config dir
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, '..', '..', '..', 'config', 'security_config.json')
        self.config = self._load_config(config_path)
        self.vulnerabilities = self._load_vulnerability_db()

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _load_vulnerability_db(self):
        try:
            response = requests.get(self.config['vulnerability_db_url'], timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Could not fetch vulnerability DB. Dependency check will be unavailable. Error: {e}")
            return {}

    def scan_file_for_secrets(self, file_path):
        """Scans a single file for hardcoded secrets based on regex patterns."""
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    for rule_name, pattern in self.config['scan_rules'].items():
                        if re.search(pattern, line):
                            findings.append({
                                'file': file_path,
                                'line': i + 1,
                                'rule': rule_name,
                                'finding': line.strip()
                            })
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        return findings

    def check_dependencies(self, requirements_path):
        """Checks a requirements.txt file against the vulnerability database."""
        findings = []
        if not self.vulnerabilities:
            return findings # DB not loaded

        try:
            with open(requirements_path, 'r') as f:
                for line in f:
                    if '==' in line:
                        name, version = line.strip().split('==')
                        if name in self.vulnerabilities:
                            for vuln in self.vulnerabilities[name]:
                                # This is a simplified check; a real one would use version specifiers
                                findings.append({
                                    'dependency': name,
                                    'version': version,
                                    'vulnerability': vuln['id'],
                                    'description': vuln['v']
                                })
        except FileNotFoundError:
            print(f"Error: requirements.txt file not found at {requirements_path}")
        return findings

    def generate_report(self, findings):
        """Generates a timestamped JSON report of findings."""
        reports_dir = self.config['reports_dir']
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(reports_dir, f"report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(findings, f, indent=2)
        
        print(f"Report generated at {report_path}")
        return report_path
