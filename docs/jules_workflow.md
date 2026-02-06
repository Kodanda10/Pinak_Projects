# Jules Workflow

This document outlines the standard workflow for Jules, emphasizing issue-driven development, rapid execution, and rigorous testing.

## 1. Issue-Driven Work
All work begins with a GitHub issue.
- **Scope**: Clearly defined in the issue description.
- **Reference**: Every Pull Request (PR) must reference the specific issue number it addresses (e.g., "Fixes #38").
- **Focus**: Do not deviate from the issue's requirements without explicit user confirmation.

## 2. Planless Execution
We prioritize **action and exploration** over extensive, rigid planning.
- **Explore First**: Use tools to explore the codebase (`list_files`, `read_file`) immediately.
- **Iterate**: Write code, run it, observe failures, and fix them. Do not spend excessive turns on "planning" before touching the code.
- **Adapt**: Plans are tentative. If the code reveals a better path, take it (after verifying).
- **Tool Usage**: Make frequent, small tool calls to verify assumptions rather than assuming how the code works.

## 3. PR Creation
When the task is complete and verified:
- **Branch**: Create a new branch with a descriptive name.
- **Title**: Use a clear, imperative title (e.g., "Add Jules workflow documentation").
- **Description**: Briefly describe the changes and link the issue.
- **Submit**: Use the `submit` tool to finalize the work.

## 4. Review + Merge
- **Code Review**: Always review your own diffs before submitting.
- **User Review**: The user (or maintainer) will review the PR.
- **Merge**: Once approved, the PR is merged into the main branch.

## 5. Testing Expectations
- **Proactive Testing**: Run existing tests relevant to your changes.
- **New Tests**: Add new tests for new features or bug fixes whenever possible.
- **Verification**: Never assume a change works. Verify it with a test run, a script, or a read-back of the file.
- **No Regressions**: Ensure that changes do not break existing functionality.
