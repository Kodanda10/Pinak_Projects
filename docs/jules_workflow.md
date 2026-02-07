# Jules Workflow

This document outlines the workflow for Jules, focusing on issue-driven development, planless execution, and quality assurance.

## 1. Issue-Driven Work

All work must originate from a GitHub issue. This ensures that every task is tracked, prioritized, and has a clear objective. Before starting any work, ensure there is an open issue describing the problem or feature request.

## 2. Planless Execution

Jules prioritizes "planless execution," which emphasizes:

*   **Iterative Coding**: Start coding and exploring the codebase immediately rather than spending excessive time on upfront planning.
*   **Exploration**: Use the tools to explore the codebase, understand existing patterns, and identify necessary changes dynamically.
*   **Adaptability**: Be ready to adjust the approach as new information is discovered during the coding process.

This approach encourages learning by doing and reduces the paralysis of analysis.

## 3. PR Creation

Pull Requests (PRs) should be created early in the process.

*   **Reference the Issue**: The PR description and commit messages must reference the relevant GitHub issue (e.g., `Fixes #38`).
*   **Descriptive Title**: Use a clear and concise title that summarizes the changes.
*   **Context**: Provide enough context in the PR description to help reviewers understand the changes and the reasoning behind them.

## 4. Review & Merge

*   **Code Review**: All changes must undergo a code review process. This involves checking for correctness, style, and potential issues.
*   **Addressing Feedback**: Respond to review comments constructively and make necessary adjustments.
*   **Merge Criteria**: A PR can be merged once it has been approved and all checks (tests, linters) have passed.

## 5. Testing Expectations

Testing is a critical part of the workflow.

*   **Unit Tests**: Write unit tests for new functions and classes to verify their behavior in isolation.
*   **Integration Tests**: Ensure that different parts of the system work together as expected.
*   **Regression Tests**: When fixing bugs, add regression tests to prevent the issue from reoccurring.
*   **Verification**: Always verify changes by running the relevant tests and ensuring they pass before submitting a PR.
