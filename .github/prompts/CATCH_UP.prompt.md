---
name: CATCH_UP
description: "Catch up on recent code changes, open issues, and git status to plan next steps."
argument-hint: Optional focus area or number of recent days to review (e.g., "verification system", "last 3 days")
agent: agent
---
Help me catch up on the current state of this project so I can plan my next steps.

## 1. Read Recent Notes

Read files given by the user or if unspecified read [.github/notes/](.github/notes/) starting from the most recent. If the user specified a time window, limit to that range; otherwise read the last 3 notes.

For each note, extract:
- **Date & topic**
- **Branch**
- **What was done** (one-sentence summary)
- **Unresolved issues or open questions** mentioned in the note
- **Files changed** (collapsed list)

## 2. Check Git Status (if specified or if notes indicate recent code changes)

Run `git status`, `git log --oneline -15`, and `git diff --stat HEAD~5` to capture:
- Current branch
- Uncommitted or staged changes
- Recent commit history

## 3. Identify Loose Ends

Cross-reference the notes and git state to surface:
- Bug fixes that may need follow-up testing
- Design decisions flagged as temporary or deferred
- Files modified recently but not mentioned in any note
- TODOs or FIXMEs in the most recent commit (`git diff HEAD~1 HEAD` grepped for TODO/FIXME)

## 4. Produce a Status Brief

Output a concise markdown brief with these sections:

```
## Status Brief — <today's date>

### Current State
<1-2 sentences: branch, what's working, what's not>

### Recent Changes (newest first)
| Date | Topic | Status |
|------|-------|--------|
| ...  | ...   | done / partial / blocked |

### Open Issues & Loose Ends
- <bullet list of unresolved items from notes + git inspection>

### Suggested Next Steps
1. <prioritised action items based on the above>
2. ...
```

Keep the brief under ~40 lines. Prioritise actionable items over history recaps. If the user provided a focus area, weight that topic more heavily in the next-steps suggestions.
