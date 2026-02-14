# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.

## Code of Conduct

Don't be reckless when committing code to the main branch.
Be respectful, communicate openly, and resolve conflicts constructively.  
To report behavior concerns, contact the team lead via private Discord or email.

## Getting Started

- Python 3.10+
- pip install -r requirements.txt
- Git and GitHub account

## Branching & Workflow

Enforce clean commits, consistent naming, and updated documentation

- Clean Commits
  - If adding brand new feature create new branch base one main
  - If adding onto another branch base it one that branch instead
  - When committing do -Updating / feat to make it clear if you’re adding a new feat or simplie updating something

- Consistent naming
  - As stated before, use the same words like 'updating' if you are simplifying or changing something to keep things clear. 
  - For example, instead of everyone using -Adding, Feat, Function to show adding a new feature, just use -Feat. Simple and easy

- Update Documentation
  - As we continue to work on the program, make sure to update how to run your program as you go on another new branch to update the README, CONTRIB, etc. 
  - Better to do little by little over time than all of it at once.


## Issues & Planning

-Each new bug, version, or document change will be posted on the GitHub project with a clear label and description
-Labels could be: bugs, enhancement, documentation, high priority

## Commit Messages

We use the Conventional Commits convention for all commit messages to ensure clarity and consistency.
- Use bullet points of what change/added/deleted, etc

Examples:
- updated 1% test
  - divided data to short/long test
  - use 5 different test methods
  - pick best one
  - Improve accuracy

For each commit that fixes a specific issue should be label the issue’s number so we know which exact bug was fixed.

## Testing

- Currently, you will have to run it manumal by running each python program in the ternimal
(plan to change later in the future)

## Pull Requests & Reviews

- Each PR must:
  - Pass all CI checks (lint, tests, build)
  - Be reviewed and approved by at least one teammate
  - Include a descriptive title and link to the related issue

## How to Run CI checks
  - python -m pytest Unit_testing/
  - python -m flake8 "Folder/File_path"
  - python -m black "Folder/File_path"
    - Add --check after black if you don't want auto correct formatting

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
