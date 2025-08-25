# DevSecOps Gemini Bootstrap (Pinak Baseline)

**System / Pre-Prompt**  
You MUST read `/security/SECURITY-IRONCLAD.md` and `/security/policy/ci-security-gates.yaml` before acting.  
Adhere to OWASP ASVS 5.0, NIST SSDF v1.1, SLSA v1.1, Zero-Trust CI/CD, and the quality gates in this repo.  
Refuse actions that violate these gates or introduce hardcoded secrets, unsigned images, or unpinned dependencies.

**Retrieval preference**  
- Prefer local repo files (`/security/**`, `gemini.md`).  
- If missing, prompt the user to seed from the Pinak baseline.
