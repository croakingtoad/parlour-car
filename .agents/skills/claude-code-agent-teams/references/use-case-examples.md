# Use Case Examples

End-to-end workflows showing how to use agent teams for different types of tasks. Each example includes the prompt, expected team behavior, and tips.

---

## Example 1: Parallel Code Review

The most natural fit for agent teams. Each reviewer applies a different lens so issues don't get missed.

### The Prompt

```
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

### What Happens

1. Lead creates a team with three teammates
2. Lead creates tasks: "Security review", "Performance review", "Test coverage review"
3. Each teammate claims a task and begins reviewing
4. Teammates read the PR diff, trace affected code paths, check for issues
5. Each writes findings (inline or to a report)
6. Lead synthesizes all findings into a unified review

### Why It Works

- Reviewers don't overlap — each has a distinct concern
- All work is read-heavy — no file conflicts
- A single reviewer tends to gravitate toward one issue type; splitting forces comprehensive coverage

### Tips

- Add specific criteria: "Flag any use of unsafe DOM injection or raw SQL string concatenation"
- Specify severity ratings: "Rate each finding as high/medium/low"
- For large PRs, further split by module within each concern

---

## Example 2: Competing Hypothesis Debugging

When the root cause is unclear, parallel investigation with adversarial debate finds answers faster than sequential exploration.

### The Prompt

```
Users report the app exits after one message instead of staying connected.
Spawn 5 agent teammates to investigate different hypotheses. Have them talk
to each other to try to disprove each other's theories, like a scientific
debate. Update the findings doc with whatever consensus emerges.
```

### What Happens

1. Lead creates 5 teammates, each investigating a different theory
2. Teammates explore: connection handling, timeout config, error recovery, event loop, state management
3. As teammates find evidence, they message each other: "I found the WebSocket closes after first message — does your theory explain this?"
4. Teammates challenge and refine each other's hypotheses
5. Team converges on the actual root cause through debate
6. Lead documents the consensus finding

### Why It Works

- Sequential debugging suffers from anchoring — once one theory is explored, subsequent investigation is biased toward it
- Parallel adversarial investigation surfaces the root cause that survives scrutiny
- Inter-teammate messaging enables real-time debate without the lead as bottleneck

### Tips

- Encourage explicit disproval: "Your job is not just to investigate your theory, but to try to disprove the others"
- Set a time bound: "If no consensus in 10 minutes, each report your strongest evidence and let the lead decide"
- Use this for intermittent/non-obvious bugs, not for bugs with obvious stack traces

---

## Example 3: New Feature Development

Each teammate owns a separate piece of a new feature, with plan approval to catch problems early.

### The Prompt

```
Create a team to build the notification system:
- Teammate 1: Database schema and migration (src/db/notifications/)
- Teammate 2: API endpoints and handlers (src/api/notifications/)
- Teammate 3: Frontend components and state (src/components/notifications/)

Require plan approval for each teammate before they make any changes.
Set up task dependencies: API teammate blocked until schema is done.
Frontend teammate blocked until API is done.

Only approve plans that include error handling and test coverage.
```

### What Happens

1. Lead creates the team and task list with dependencies
2. Schema teammate enters plan mode, explores existing patterns, proposes migration
3. Lead reviews schema plan — checks for proper indexes, foreign keys, naming conventions
4. Lead approves (or rejects with feedback)
5. Schema teammate implements, marks task complete
6. API teammate unblocks, enters plan mode, designs endpoints based on the now-existing schema
7. Process repeats through all layers

### Why It Works

- Plan approval catches architectural problems before code is written
- Task dependencies enforce the correct ordering automatically
- Each teammate owns distinct directories — no file conflicts
- The lead's approval criteria ("include error handling and tests") are enforced autonomously

### Tips

- Have teammates share their API contracts with downstream teammates via messaging
- Use delegate mode so the lead coordinates instead of implementing
- If the lead starts implementing itself: "Wait for your teammates to complete their tasks before proceeding"

---

## Example 4: Cross-Layer Coordination

Changes that span frontend, backend, and tests, each owned by a different teammate.

### The Prompt

```
Create a team to implement the new auth flow:
- Frontend teammate: login form, token storage, route guards
- Backend teammate: auth middleware, JWT handling, new endpoints
- Test teammate: integration tests covering the full flow

Have frontend and backend communicate to agree on the API contract
before implementing. Test teammate should start writing test skeletons
after the contract is agreed on.
```

### What Happens

1. Lead creates team with three teammates
2. Frontend and backend teammates message each other to agree on endpoints, request/response shapes, error codes
3. Once aligned, both begin implementing their respective layers
4. Test teammate watches for the agreed contract and writes test skeletons
5. As implementation progresses, test teammate fills in assertions
6. Lead verifies integration by having test teammate run the full suite

### Why It Works

- Direct teammate-to-teammate messaging enables contract negotiation without the lead as intermediary
- Each layer is independent once the contract is agreed
- Tests validate integration automatically

### Tips

- Include specific contract negotiation instructions: "Agree on request/response types before writing any code"
- Consider having the contract documented in a shared file both can reference
- Use CLAUDE.md for authentication patterns and conventions to avoid re-explaining

---

## Example 5: Codebase Audit with Quality Gates

Using hooks to enforce standards as teammates complete their reviews.

### The Prompt

```
Create a team to audit our codebase for production readiness.
Spawn four reviewers:
- Security reviewer: auth, input validation, secrets management
- Performance reviewer: queries, caching, algorithms
- Reliability reviewer: error handling, logging, recovery
- Compliance reviewer: data privacy, PII handling, audit trails

Each reviewer must write findings to reports/{concern}-audit.md.
After all complete, synthesize into a go/no-go recommendation.
```

### Enhancing with Hooks

Add a `TaskCompleted` hook that checks report quality:

```bash
#!/bin/bash
# Reject task completion if report is missing or too short
REPORT_FILE="reports/${TASK_NAME}-audit.md"
if [ ! -f "$REPORT_FILE" ]; then
  echo "Report file not found. Write findings to $REPORT_FILE before completing."
  exit 2
fi
LINES=$(wc -l < "$REPORT_FILE")
if [ "$LINES" -lt 20 ]; then
  echo "Report too short ($LINES lines). Provide more detailed findings."
  exit 2
fi
exit 0
```

### What Happens

1. Lead creates team, assigns concern areas
2. Teammates review in parallel, writing to separate report files
3. When a teammate tries to mark their task complete, the hook validates the report
4. If the report is missing or too short, the hook rejects completion and sends feedback
5. Teammate continues working until the report meets the quality bar
6. Lead synthesizes all reports into a final recommendation

---

## Example 6: Exploring Design Alternatives

Research task where teammates explore different approaches and the lead picks the best.

### The Prompt

```
I need to add real-time features to our app. Create a research team:
- Teammate 1: Research WebSockets (pros, cons, implementation complexity)
- Teammate 2: Research Server-Sent Events (pros, cons, implementation complexity)
- Teammate 3: Research long polling (pros, cons, implementation complexity)
- Teammate 4: Evaluate our current architecture for real-time readiness

Each should write a report to reports/{approach}.md covering:
- How it works
- Pros and cons for our specific stack (Node.js, React, PostgreSQL)
- Implementation effort estimate
- Scaling considerations

After all complete, compare approaches and recommend one.
```

### What Happens

1. Each teammate researches their assigned approach independently
2. Teammate 4 evaluates current architecture constraints that affect the choice
3. All write reports to distinct files
4. Lead reads all reports, cross-references findings, and makes a recommendation considering the architecture evaluation

### Tips

- The architecture evaluator is key — without it, recommendations are generic rather than specific to the codebase
- Have teammates explicitly consider the existing stack, not just the technology in isolation
- If teammates have access to the codebase, tell them to look at relevant files (e.g., existing WebSocket config, database connection pooling)
