# Jules Workflow

This document outlines the standard workflow for Jules when contributing to the project.

## 1. Issue-Driven Work
All development efforts are initiated by a GitHub issue.
- **Scope Definition:** The issue serves as the single source of truth for the task's requirements.
- **Focus:** Work is strictly limited to solving the problem described in the issue.
- **Traceability:** Every Pull Request (PR) must reference the issue it addresses (e.g., `fixes #38`).

## 2. Planless Execution
Jules adopts an agile, execution-first approach rather than relying on heavy upfront planning artifacts, unless complexity demands it.
- **Dynamic Adaptation:** The agent explores the codebase to understand context and adapts its actions accordingly.
- **Iterative Improvement:** Changes are made incrementally, with immediate verification at each step.
- **Speed:** This approach prioritizes rapid prototyping and implementation over extensive documentation of the "how" before the "do".

## 3. PR Creation
When the task is complete, a Pull Request is created.
- **Branching:** Use a dedicated branch for the issue.
- **Descriptive Title:** The PR title should concisely summarize the change.
- **Detailed Description:** Provide a clear explanation of what was changed and why.
- **Issue Linking:** Explicitly link the PR to the tracking issue.

## 4. Review + Merge
Quality assurance is a collaborative process.
- **Plan Review:** Initial plans (when used) are reviewed to ensure alignment with goals.
- **Code Review:** The implementation is reviewed for correctness, style, and security.
- **Feedback Loop:** Feedback is addressed promptly before the code is merged into the main branch.

## 5. Testing Expectations
Rigorous testing is non-negotiable.
- **Regression Testing:** Existing tests must pass to ensure no functionality is broken.
- **New Coverage:** New features or bug fixes must be accompanied by relevant tests (unit, integration, or manual verification steps).
- **Verification:** Every change is verified in the environment (e.g., via `read_file`, running scripts) before being marked as complete.
