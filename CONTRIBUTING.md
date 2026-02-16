# Contributing to Sparse-Identification-of-Nonlinear-Dynamic-Systems

Repository:
[https://github.com/PNG5042/Sparse-Identification-of-Nonlinear-Dynamic-Systems](https://github.com/PNG5042/Sparse-Identification-of-Nonlinear-Dynamic-Systems)

Thank you for contributing to this project. This guide explains how to set up the project locally, how to follow our contribution workflow, and how we enforce quality standards.

Please read this document before submitting a Pull Request (PR).

---

# 1. Prerequisites & Local Setup

## 1.1 Prerequisites

* Python 3.10 or newer
* Git
* pip
* Virtual environment tool (venv recommended)

---

## 1.2 Clone the Repository

```bash
git clone https://github.com/PNG5042/Sparse-Identification-of-Nonlinear-Dynamic-Systems.git
cd Sparse-Identification-of-Nonlinear-Dynamic-Systems
```

---

## 1.3 Create and Activate Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not available:

```bash
pip install pytest flake8 black pysindy
```

---

## 1.5 Run the Project

Example:

```bash
python main.py
```

---

# 2. Running Tests, Linters, and Formatters Locally

All contributors must run checks locally before submitting a PR.

---

## 2.1 Run Unit Tests

```bash
pytest Unit_Testing/
```

All tests must pass before opening a PR.

---

## 2.2 Check Test Coverage (if enabled)

```bash
pytest --cov=.
```

Minimum required coverage: **80%**

---

## 2.3 Run Linter (Flake8)

```bash
flake8 .
```

There must be **zero linting errors**.

---

## 2.4 Run Formatter (Black)

Auto-format:

```bash
black .
```

Check formatting only:

```bash
black --check .
```

All files must pass formatting checks before merge.

---

## 2.5 CI Enforcement

GitHub Actions runs automatically on every Pull Request.

Merge is blocked if:

* Tests fail
* Coverage drops below 80%
* Flake8 fails
* Black fails
* Security vulnerabilities are detected

Branch protection rules prevent direct pushes to `main`.

---

# 3. Contribution Workflow

## 3.1 Create a Branch

Always branch from `main`:

```bash
git checkout main
git pull
git checkout -b feature/short-description
```

Branch naming conventions:

* `feature/<description>`
* `bugfix/<description>`
* `refactor/<description>`
* `docs/<description>`

Example:

```bash
feature/add-316-rupture-test
```

---

## 3.2 Making Changes

When contributing:

* Write clear, modular code
* Add or update unit tests
* Update documentation if behavior changes
* Remove TODO comments before submission
* Keep PRs small and focused

---

## 3.3 Definition of Done (DoD)

Before opening a PR, confirm:

* Feature satisfies issue acceptance criteria
* Unit tests added for new logic
* All tests pass
* ≥ 80% coverage maintained
* Flake8 passes
* Black passes
* Documentation updated if applicable
* No critical security vulnerabilities
* PR reviewed by at least 1 team member

If any condition is unmet, the PR will not be approved.

---

## 3.4 Opening a Pull Request

When opening a PR:

1. Link the related issue (e.g., `Closes #12`)
2. Provide:

   * Summary of changes
   * Rationale
   * Testing performed
   * Screenshots (if graphs/output changed)
3. Confirm all local checks passed

---

## 3.5 Code Review Expectations

* Minimum 1 approval required
* Reviewers verify:

  * Code clarity
  * Correctness
  * Test coverage
  * Documentation updates
  * CI passing
* Reviews must focus on behavior and quality (not personal critique)
* PRs should be reviewed within 48 hours

Large PRs may be requested to split into smaller changes.

---

# 4. Reporting Bugs & Requesting Changes

All bugs and feature requests must be submitted via GitHub Issues:

[https://github.com/PNG5042/Sparse-Identification-of-Nonlinear-Dynamic-Systems/issues](https://github.com/PNG5042/Sparse-Identification-of-Nonlinear-Dynamic-Systems/issues)

---

## 4.1 Bug Report Must Include

* Clear descriptive title
* Steps to reproduce
* Expected behavior
* Actual behavior
* Environment details (OS, Python version)
* Logs or screenshots if applicable

---

## 4.2 Feature Request Must Include

* Problem description
* Proposed solution
* Impact on the system
* Supporting research/data if relevant

Incomplete issues may be returned for clarification.

---

# 5. Where to Ask for Help

For assistance:

* Primary: Email
* Secondary: Comment on the relevant GitHub Issue
* Contact:

  * Team Lead for coordination issues
  * Tech Lead for architecture questions
  * QA Owner for testing/CI questions

Response target: within 24 hours.

---

# 6. Professional Conduct

All contributors must:

* Maintain respectful and professional communication
* Use evidence-based discussion during reviews
* Follow inclusive participation norms
* Adhere to the Team Charter conflict resolution process

---

# 7. PR Submission Checklist

Before submitting:

* Tests added/updated
* Tests pass
* Coverage ≥ 80%
* Flake8 passes
* Black passes
* Documentation updated
* Issue linked
* PR is focused and scoped

---
