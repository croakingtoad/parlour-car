# Task Decomposition Patterns

Reference for breaking down tasks into parallel teammate work streams. Covers partitioning strategies, team coordination patterns, and anti-patterns.

---

## Partitioning Strategies

### Module-Based Partitioning

Assign each teammate a complete module or directory. Best for large codebases with clear boundaries.

```
Project structure:
src/
├── auth/        → Teammate: security-reviewer
├── payments/    → Teammate: payments-reviewer
├── notifications/ → Teammate: notifications-reviewer
└── users/       → Teammate: users-reviewer
```

Prompt to lead:

```
Create an agent team to review this codebase. Assign one teammate per
top-level module in src/. Each should write findings to reports/{module}.md.
Teammates should not modify any source files.
```

### Concern-Based Partitioning

Each teammate looks at the entire codebase from a different angle. Best for smaller codebases or cross-cutting concerns.

```
Concern areas:
├── Security           → Teammate 1 (injection, auth, secrets)
├── Performance        → Teammate 2 (N+1, caching, algorithms)
├── Error Handling     → Teammate 3 (uncaught, missing validation)
└── Code Quality       → Teammate 4 (duplication, naming, complexity)
```

Multiple teammates reading the same files is fine — reads never conflict. The risk comes from overlapping write targets.

### Layer-Based Partitioning

For full-stack applications, partition by architectural layer:

```
├── Frontend (components, pages)     → Teammate 1
├── API Layer (routes, controllers)  → Teammate 2
├── Business Logic (services)        → Teammate 3
└── Data Layer (models, db)          → Teammate 4
```

### Hypothesis-Based Partitioning

For debugging, each teammate investigates a different theory. Unique to agent teams: teammates can message each other to challenge findings.

```
Bug: "Users intermittently get 500 errors on checkout"

Create an agent team with 5 teammates. Each investigates a different
hypothesis. Have them talk to each other to try to disprove each other's
theories, like a scientific debate. Update a shared findings doc with
whatever consensus emerges.
```

This adversarial structure fights anchoring bias — sequential investigation tends to stop at the first plausible explanation.

---

## Task Type Patterns

### Code Review

The strongest use case. Teammates review in parallel, share findings, and the lead synthesizes.

```
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

### Test Generation

Test files are usually independent — each teammate writes tests for a different module.

```
Create an agent team to write tests for the API layer.
Split by module: auth, users, payments.
Each teammate creates tests in the appropriate __tests__ directory.
After writing, each should run their tests and fix failures before completing.
```

Constraint to include: "After writing tests, run them to verify they pass. Fix any failures before marking the task complete."

### Documentation

Each teammate documents a different component, writing to separate files.

```
Create a team to document our API. Assign one teammate per endpoint group:
- Authentication endpoints
- User management endpoints
- Payment endpoints
After all complete, synthesize into a unified API reference.
```

### Refactoring (Careful)

Works when changes are cleanly partitioned. The key risk is write conflicts.

**Safe pattern — file-isolated refactoring:**

```
Phase 1 (Lead or single teammate):
  Update shared interfaces and type definitions

Phase 2 (Parallel teammates):
  ├── Teammate 1: Update consumers in src/services/
  ├── Teammate 2: Update consumers in src/api/
  └── Teammate 3: Update consumers in src/workers/
```

Use task dependencies so Phase 2 teammates are blocked until Phase 1 completes.

**Constraint for each teammate:**

```
Only modify files within your assigned directory.
Do not modify shared configuration files.
Do not modify imports in files outside your directory.
```

### Migration

Library swaps and framework upgrades with repetitive changes across many files.

```
Create a team to migrate from Moment.js to date-fns. Split by directory.
Each teammate applies these rules consistently:
1. moment() → new Date()
2. moment(str) → parseISO(str)
3. .format('YYYY-MM-DD') → format(date, 'yyyy-MM-dd')
4. .add(n, 'days') → addDays(date, n)

After teammates finish, remove moment from package.json and run tests.
```

Include migration rules directly in each teammate's task since they don't share context.

### New Feature Development

Each teammate owns a separate piece. Use plan approval for risky work.

```
Create a team to build the notification system:
- Teammate 1: Database schema and models (src/db/notifications/)
- Teammate 2: API endpoints (src/api/notifications/)
- Teammate 3: Frontend components (src/components/notifications/)

Require plan approval for each teammate before they start implementing.
Set up task dependencies: API teammate blocked until schema is done.
```

### Cross-Layer Coordination

Changes spanning frontend, backend, and tests. Each teammate owns a layer.

```
Create a team to implement the new auth flow:
- Frontend teammate: login form, token storage, route guards
- Backend teammate: auth middleware, JWT handling, endpoints
- Test teammate: integration tests for the full flow

Have them communicate to agree on the API contract before implementing.
```

### Multi-Repository Tasks

Each teammate handles a different repo.

```
├── Teammate 1: In /repos/frontend, update the API client
├── Teammate 2: In /repos/backend, add the new auth middleware
└── Teammate 3: In /repos/shared-types, update TypeScript definitions
```

Set dependencies if repos depend on each other, or provide shared definitions directly in each prompt.

---

## Coordination Patterns

### Task Dependencies

Use the shared task list to enforce ordering:

```
Task A: "Set up database schema" (no dependencies)
Task B: "Write API endpoints" (depends on A)
Task C: "Write frontend" (depends on B)
Task D: "Write schema tests" (depends on A)
```

Tasks B and C block automatically. When A completes, B and D unblock. Teammates self-claim unblocked tasks.

### Shared Context via CLAUDE.md

Since teammates load CLAUDE.md automatically, put shared conventions there:

- Coding standards and patterns
- Architecture decisions
- API contracts
- Testing requirements

This avoids repeating context in every spawn prompt.

### Lead as Synthesizer

After teammates complete, the lead:

1. Reads all output files
2. Cross-references findings
3. Resolves contradictions
4. Produces a unified deliverable

Enable delegate mode (Shift+Tab) to prevent the lead from implementing instead of coordinating.

### Staged Execution

For work with dependencies, break into stages:

```
Stage 1 (parallel): Research and planning
  - Teammates explore different approaches
  - Each writes a proposal

Stage 2 (lead): Decision
  - Lead reviews proposals, picks approach

Stage 3 (parallel): Implementation
  - Teammates implement their assigned pieces
  - Task dependencies enforce ordering

Stage 4 (lead): Integration and testing
  - Lead merges results, runs tests
```

---

## Sizing Guidelines

| Task Size | Teammates | Example |
|-----------|-----------|---------|
| Too small | 0 (single session) | Rename a variable, fix a typo |
| Minimum viable | 2 | Review frontend + backend |
| Sweet spot | 3-4 | Review/implement 3-4 independent modules |
| Large but manageable | 5-6 | Full codebase audit across many dimensions |
| Diminishing returns | 7+ | Coordination overhead exceeds parallelism benefit |

**Rule of thumb:** 5-6 tasks per teammate keeps everyone productive. If a teammate runs out of work, it self-claims the next unblocked task.

**Task sizing:** A task should be self-contained and produce a clear deliverable — a function, a test file, a review report. Too small wastes spawn overhead. Too large means teammates work too long without check-ins, increasing risk of wasted effort.

---

## Anti-Patterns

### The Sequential Chain

```
Bad:
├── Teammate 1: "Design the database schema"
├── Teammate 2: "Build the API using the schema from #1"  ← FAILS
└── Teammate 3: "Build the frontend using the API from #2" ← FAILS
```

Teammates start simultaneously. Use task dependencies instead — teammate 2's task depends on teammate 1's task.

### The Shared Config Editor

```
Bad:
├── Teammate 1: "Add your service to docker-compose.yml"
├── Teammate 2: "Add your service to docker-compose.yml"  ← CONFLICT
└── Teammate 3: "Add your service to docker-compose.yml"  ← CONFLICT
```

Have the lead make shared config changes after teammates complete their module-specific work.

### The Micro-Tasker

```
Bad:
├── Teammate 1: "Rename variable foo to bar in file.ts"
├── Teammate 2: "Add a comment to line 42 of file.ts"
└── Teammate 3: "Fix the indentation in file.ts"
```

Tasks too small, all touching the same file. A single session handles this in seconds.

### The Context Amnesiac

```
Bad:
Spawn prompt: "Continue what we were discussing about the auth refactor."
```

Teammates start fresh with no conversation history. Include all necessary context in the spawn prompt or rely on CLAUDE.md.

### The Unmonitored Swarm

```
Bad:
Spawn 6 teammates, walk away, hope for the best.
```

Agent teams need steering. Check progress, redirect failing approaches, and synthesize findings as they arrive. Letting a team run unattended for too long increases the risk of wasted effort.

### The Over-Broadcaster

```
Bad:
Broadcast every status update to all teammates.
```

Broadcasts multiply token cost by team size. Use direct messages for teammate-specific information. Reserve broadcasts for team-wide changes (schema updates, new constraints).
