---
agent: code_base_analysis
---

# Agent Codebase Analysis & Documentation Prompt

## Objective
Analyze the existing codebase comprehensively and create structured documentation in `.github/` that helps AI agents understand the project's architecture, dependencies, coding patterns, and workflows.

## Instructions

### Phase 1: Initial Codebase Discovery

1. **Repository Structure Analysis**
   - Map the complete directory structure
   - Identify main modules, submodules, and their purposes
   - Document entry points (main scripts, CLI tools, APIs)
   - Note configuration files and their roles

2. **Dependency Mapping**
   - Analyze `requirements.txt`, `environment.yml`, `setup.py`
   - Identify direct vs. transitive dependencies
   - Document optional dependencies (GPU, development, testing)
   - Note any git submodules and their integration points

3. **Core Components Identification**
   - Identify key classes, functions, and modules
   - Map data flow between components
   - Document public APIs and internal utilities
   - Note any design patterns in use (Factory, Observer, etc.)

### Phase 2: Create Documentation Structure

Create the following folder structure in `.github/`:

```
.github/
├── copilot-instructions.md          # Main instructions for AI assistants
├── architecture/
│   ├── overview.md                  # High-level architecture diagram & description
│   ├── module-hierarchy.md          # Module dependencies and relationships
│   ├── data-flow.md                 # Data flow diagrams and explanations
│   └── integration-points.md        # External system integrations
├── dependencies/
│   ├── core-dependencies.md         # Essential runtime dependencies
│   ├── optional-dependencies.md     # GPU, dev, test dependencies
│   ├── version-constraints.md       # Known compatibility issues
│   └── submodules.md                # Git submodule documentation
├── patterns/
│   ├── coding-conventions.md        # Style guide and naming conventions
│   ├── common-patterns.md           # Frequently used code patterns
│   ├── anti-patterns.md             # What to avoid
│   └── testing-patterns.md          # Test structure and conventions
├── workflows/
│   ├── development-workflow.md      # Local dev setup and iteration
│   ├── testing-workflow.md          # How to run and write tests
│   ├── deployment-workflow.md       # Build, package, deploy steps
│   └── common-tasks.md              # Step-by-step guides for frequent tasks
└── changelog/
    ├── tracking-guide.md            # How to track changes
    ├── YYYY-MM-DD-description.md    # Individual change logs
    └── breaking-changes.md          # Major breaking changes history
```

### Phase 3: Document Each Category

#### Architecture Documentation (`architecture/`)

**overview.md** should include:
- High-level system architecture
- Major subsystems and their responsibilities
- Technology stack overview
- Key design decisions and rationale

**module-hierarchy.md** should include:
- Module dependency graph
- Import patterns and conventions
- Circular dependency notes (if any)
- Layer separation (presentation, business logic, data)

**data-flow.md** should include:
- Data transformation pipelines
- Input/output formats
- State management approach
- Critical data structures

**integration-points.md** should include:
- External libraries and how they're used
- API boundaries
- File I/O patterns
- Database/storage interactions

#### Dependencies Documentation (`dependencies/`)

**core-dependencies.md** should include:
- Each core dependency with purpose
- Version requirements and why
- Alternative options considered
- Known issues or limitations

**optional-dependencies.md** should include:
- GPU acceleration requirements
- Development tools (linters, formatters)
- Testing frameworks
- Documentation generators

**version-constraints.md** should include:
- Python version requirements
- OS-specific dependencies
- Known incompatibilities
- Migration notes between versions

**submodules.md** should include:
- Each submodule's purpose
- How it integrates with main project
- Update procedures
- Alternative implementations

#### Patterns Documentation (`patterns/`)

**coding-conventions.md** should include:
- Naming conventions (classes, functions, variables)
- File organization standards
- Import ordering
- Docstring format
- Type hints usage

**common-patterns.md** should include:
- Configuration management patterns
- Error handling patterns
- Logging patterns
- Resource initialization patterns
- Common class structures

**anti-patterns.md** should include:
- Known problematic patterns in the codebase
- Why they should be avoided
- Better alternatives
- Migration strategies

**testing-patterns.md** should include:
- Test file organization
- Fixture patterns
- Mock/stub patterns
- Test data management
- Assertion styles

#### Workflows Documentation (`workflows/`)

**development-workflow.md** should include:
- Environment setup steps
- Running the project locally
- Hot reloading / development mode
- Debugging approaches

**testing-workflow.md** should include:
- Running all tests
- Running specific test suites
- Writing new tests
- Coverage requirements
- CI/CD integration

**deployment-workflow.md** should include:
- Build process
- Package creation
- Release procedures
- Environment configuration
- Rollback procedures

**common-tasks.md** should include:
- Adding a new feature (step-by-step)
- Fixing a bug (step-by-step)
- Adding a new dependency
- Updating documentation
- Performance optimization workflow

#### Changelog Documentation (`changelog/`)

**tracking-guide.md** should include:
- What changes to track
- How to document changes
- Naming convention for change logs
- When to create breaking-change notes
- How to link changes to issues/PRs

**Individual change logs** should include:
- Date and author
- Type: [Feature, Bugfix, Refactor, Performance, Breaking]
- Affected modules/files
- Description of changes
- Rationale for changes
- Testing performed
- Migration notes (if breaking)

**breaking-changes.md** should include:
- Chronological list of breaking changes
- Version where change occurred
- What broke and why
- Migration guide
- Deprecation timeline (if applicable)

### Phase 4: Analysis Guidelines

When analyzing code, focus on:

1. **Critical Paths**: Identify the most important code paths for core functionality
2. **Configuration Points**: Note all places where behavior can be configured
3. **State Management**: How state is created, modified, and shared
4. **Error Handling**: Common error scenarios and how they're handled
5. **Performance Considerations**: CPU/GPU usage, memory management, I/O patterns
6. **Extension Points**: Where new functionality is typically added
7. **Gotchas**: Non-obvious behavior, edge cases, known issues

### Phase 5: Tracking Code Changes

#### For Each Change Session:

1. **Before Making Changes**:
   - Read relevant sections in `.github/`
   - Understand affected modules
   - Check for related patterns

2. **While Making Changes**:
   - Note deviations from existing patterns
   - Document new patterns introduced
   - Identify affected integration points

3. **After Making Changes**:
   - Create a changelog entry in `.github/changelog/`
   - Update affected documentation sections
   - Note any new gotchas or edge cases

#### Changelog Entry Template:

```markdown
# [YYYY-MM-DD] [Brief Description]

**Type**: [Feature | Bugfix | Refactor | Performance | Breaking | Documentation]

**Author**: [Name/System]

## Changes

### Modified Files
- `path/to/file1.py`: [description]
- `path/to/file2.py`: [description]

### Added Files
- `path/to/new_file.py`: [purpose]

### Deleted Files
- `path/to/old_file.py`: [reason]

## Rationale
[Why these changes were made]

## Impact
- **Affected Modules**: [list]
- **Breaking Changes**: [Yes/No - explain if yes]
- **Performance Impact**: [None/Positive/Negative - explain]
- **New Dependencies**: [list if any]

## Testing
- [How changes were tested]
- [New tests added]
- [Regression tests passed]

## Migration Guide
[If breaking change, provide step-by-step migration instructions]

## Related
- **Issue**: #[issue-number]
- **PR**: #[pr-number]
- **Related Changes**: [links to related changelogs]
```

### Phase 6: Maintenance Protocol

#### Weekly Review:
- Ensure all changes have changelog entries
- Update patterns documentation with new conventions
- Consolidate related changes into higher-level documentation

#### Monthly Review:
- Review architecture documentation for accuracy
- Update dependency documentation
- Identify outdated gotchas
- Archive old changelogs (keep index)

#### Major Version Updates:
- Create comprehensive migration guide
- Update all architecture diagrams
- Document all breaking changes
- Review and update all workflow documentation

## Deliverables Checklist

- [ ] Complete directory structure created in `.github/`
- [ ] All template files populated with initial content
- [ ] Architecture documented with diagrams
- [ ] All dependencies cataloged and explained
- [ ] Common patterns extracted and documented
- [ ] Workflows documented with step-by-step guides
- [ ] Changelog tracking system established
- [ ] Initial `.github/copilot-instructions.md` created/updated
- [ ] Cross-references between documents established
- [ ] Code examples included where helpful
- [ ] Known issues and gotchas documented

## Best Practices

1. **Be Specific**: Use concrete examples, file paths, and code snippets
2. **Be Actionable**: Provide step-by-step instructions, not just descriptions
3. **Be Accurate**: Verify all information by examining actual code
4. **Be Complete**: Cover both happy paths and edge cases
5. **Be Concise**: Use clear, direct language without unnecessary verbosity
6. **Be Current**: Update documentation as you discover new information
7. **Cross-Reference**: Link related documents together
8. **Include Context**: Explain the "why" behind decisions, not just the "what"

## Example Analysis Process

```bash
# 1. Start with high-level structure
find . -type f -name "*.py" | head -20

# 2. Identify entry points
grep -r "if __name__ == '__main__'" --include="*.py"

# 3. Analyze imports to understand dependencies
grep -r "^import \|^from " --include="*.py" | sort | uniq -c | sort -rn

# 4. Find configuration files
find . -name "*.yml" -o -name "*.yaml" -o -name "*.json" -o -name "*.toml" -o -name "*.ini"

# 5. Identify test patterns
find . -path "*/test*" -name "*.py"

# 6. Look for documentation
find . -name "README*" -o -name "*.md"
```

## Starting Point Template

When beginning analysis, create this initial file:

**.github/analysis-progress.md**:
```markdown
# Codebase Analysis Progress

## Status: [In Progress | Complete]
**Started**: [Date]
**Last Updated**: [Date]

## Completed Sections
- [ ] Repository structure mapped
- [ ] Dependencies documented
- [ ] Architecture overview created
- [ ] Module hierarchy documented
- [ ] Data flow documented
- [ ] Integration points identified
- [ ] Coding conventions extracted
- [ ] Common patterns documented
- [ ] Testing patterns documented
- [ ] Development workflow documented
- [ ] Testing workflow documented
- [ ] Deployment workflow documented
- [ ] Changelog system established

## Discoveries
### Key Insights
- [Important findings about the codebase]

### Questions / Uncertainties
- [Things that need clarification]

### Immediate Improvements Needed
- [Documentation gaps or code issues found]
```

---

## Note to Agent
This prompt is designed to be comprehensive. Focus on accuracy and usefulness over speed. The documentation you create will be the foundation for all future development work, so invest the time to make it thorough and correct.
