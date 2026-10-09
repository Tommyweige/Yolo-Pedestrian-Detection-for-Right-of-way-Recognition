# Issue tracker

Use GitHub Issues for this repository. Infer the repository from `git remote -v`.
Use `gh issue view <number> --comments` to read a ticket and `gh issue create --body-file <file>` to publish one.

Tickets link to their parent and declare genuine blockers. Use native GitHub sub-issues and dependencies when available.
Do not invent dependencies between independent tickets. Do not modify or close a specification parent while splitting it.

PRs as a request surface: no.

Apply the roles in [triage-labels.md](triage-labels.md). New, approved specifications and tickets use `ready-for-agent`.
