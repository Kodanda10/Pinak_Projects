#!/usr/bin/env python3
import json, sys, urllib.request

def load_kev_set():
    # CISA KEV catalog JSON
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    cves = {item.get("cveID") for item in data.get("vulnerabilities", []) if item.get("cveID")}
    return cves

def main():
    if len(sys.argv) < 2:
        print("usage: check_kev.py <pip_audit.json>")
        return 2
    kev = load_kev_set()
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        report = json.load(fh)
    bad = []
    for dep in report.get("dependencies", []):
        for vul in dep.get("vulns", []):
            cve = vul.get("id")
            if cve and cve in kev:
                bad.append({"package": dep.get("name"), "version": dep.get("version"), "cve": cve})
    if bad:
        print("Found KEV vulnerabilities in Python dependencies:")
        for b in bad:
            print(f"- {b['package']}@{b['version']} -> {b['cve']}")
        return 1
    print("No KEV-listed vulnerabilities detected in pip dependencies.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

