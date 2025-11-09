# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.
-py get-pip.py
-Git clone https://github.com/dynamicslab/pysindy 
-Then cd into pysindy and do pip install.

## Code of Conduct

Don't be reckless when committing code to the main branch.
Be respectful, communicate openly, and resolve conflicts constructively.  
To report behavior concerns, contact the team lead via private Discord or email.

## Getting Started

- Python 3.10+
- pip install
- Git and GitHub account
- Pysindy install

## Branching & Workflow

-We will be using the GitFlow method of branching, with each developer having their own branch. Then they will push their change onto the main branch once it has been checked and verified by themselves and one other teammate.
-We will use a rebase system, where after we create a feature, we will update it to match the main branch.

## Issues & Planning

-Each new bug, version, or document change will be posted on the GitHub project with a clear label and description
-Labels could be: bugs, enhancement, documentation, high priority

## Commit Messages

We use the Conventional Commits convention for all commit messages to ensure clarity and consistency.
Format: <type>(optional scope): short summary

Examples:
docs: update API usage section in README  
fix(api): handle missing authentication token  
feat: add processing data feature

For each commit that fixes a specific issue should be label the issue’s number so we know which exact bug was fixed.

## Code Style, Linting & Formatting

Name the formatter/linter, config file locations, and the exact commands to check/fix locally.

-We use **Black** for formatting and **flake8** for linting.

-Check style locally:
"black . && flake8"


## Testing

-All code must include unit tests using pytest.
-Run all tests locally:
"Pytest"

## Pull Requests & Reviews

- Each PR must:
  - Pass all CI checks (lint, tests, build)
  - Be reviewed and approved by at least one teammate
  - Include a descriptive title and link to the related issue

## CI/CD

CI pipeline defined in `.github/workflows/main.yml`
Jobs:
- `lint` (flake8)
- `test` (pytest)
- `build` (verify code runs)

All jobs must pass before merging to `main`.

## Security & Secrets

- Do not commit secrets or credentials.
- Use environment variables or `.env` (excluded via .gitignore).
- Report vulnerabilities privately to the team lead.

## Documentation Expectations

- Update README if setup or dependencies change.
- Add/maintain docstrings for all functions/classes.
- Use markdown in `/docs` for project-level documentation.


## Release Process

- Semantic versioning (vMAJOR.MINOR.PATCH)
- Tag releases on GitHub (e.g., `v1.0.0`)
- Update CHANGELOG.md with summary of fixes/features.

## Support & Contact

Questions or issues?  
Open a GitHub issue with label `question` or message using Email
Expected response: within 24 hours.
