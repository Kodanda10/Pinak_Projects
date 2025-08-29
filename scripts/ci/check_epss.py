#!/usr/bin/env python3
import json, sys, urllib.request, os

def load_epss_scores():
    # EPSS scores JSON feed
    url = "https://epss.cyentia.com/epss_scores-current.csv.gz"
    try:
        import gzip
        with urllib.request.urlopen(url, timeout=30) as r:
            data = gzip.decompress(r.read()).decode('utf-8')
    except Exception as e:
        print(f"EPSS feed unavailable: {e}. Skipping EPSS gate.")
        return None
    
    scores = {}
    lines = data.strip().split('\n')
    for line in lines[1:]:  # Skip header
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                cve = parts[0].strip('"')
                score = float(parts[1].strip('"'))
                scores[cve] = score
    return scores

def main():
    if len(sys.argv) < 2:
        print("usage: check_epss.py <pip_audit.json>")
        return 2
    
    threshold = float(os.environ.get('EPSS_THRESHOLD', '0.70'))
    epss = load_epss_scores()
    if epss is None:
        # Network failure or feed unavailable → do not fail the job for infra reasons
        print("EPSS gate skipped (feed unavailable)")
        return 0
    
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        report = json.load(fh)
    
    bad = []
    for dep in report.get("dependencies", []):
        for vul in dep.get("vulns", []):
            cve = vul.get("id")
            if cve and cve in epss:
                score = epss[cve]
                if score >= threshold:
                    bad.append({
                        "package": dep.get("name"), 
                        "version": dep.get("version"), 
                        "cve": cve,
                        "epss_score": score
                    })
    
    if bad:
        print(f"Found vulnerabilities with EPSS score >= {threshold}:")
        for b in bad:
            print(f"- {b['package']}@{b['version']} -> {b['cve']} (EPSS: {b['epss_score']:.4f})")
        return 1
    
    print(f"No high-risk vulnerabilities detected (EPSS < {threshold}).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
