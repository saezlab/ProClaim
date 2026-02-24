---
name: documentChanges
description: Document implementation changes as a timestamped markdown note.
argument-hint: A short description of the feature or branch name to document
---
Create a markdown file in `.github/notes/` that documents recent implementation changes.

Use today's date as a timestamp in the filename with format `YYYY-MM-DD_<slug>.md`, where `<slug>` is a lowercase-hyphenated summary of the feature or branch.

The document should include:

1. **Title** — feature name and date
2. **Branch** — the current git branch name
3. **Summary** — a concise paragraph describing what was done
4. **New Files** — a table of newly created files with their purpose
5. **Modified Files** — a table of changed files with a summary of changes
6. **Architecture** — an ASCII diagram or description of the system design, if applicable
7. **Key Design Decisions** — bullet list of important choices and their rationale
8. **Bug Fixes** — any issues discovered and resolved during implementation

Omit empty sections. Keep descriptions concise. Derive all content from the actual codebase and conversation context rather than placeholders.
