# Architecture and Communication

Deep dive into how agent teams work internally: components, messaging, task lifecycle, and token usage.

---

## Team Components

An agent team consists of four components:

### Team Lead

The main Claude Code session that creates the team. The lead:

- Spawns and shuts down teammates
- Creates and assigns tasks
- Synthesizes results from teammates
- Handles team cleanup when work is done

The lead is fixed for the team's lifetime — it cannot be transferred or promoted from a teammate. Only the lead can manage the team.

### Teammates

Each teammate is a full, independent Claude Code session with:

- Its own context window (no shared history with the lead)
- Full tool access (file editing, bash, MCP servers, skills)
- Project context loaded automatically (CLAUDE.md, MCP servers, skills)
- The spawn prompt from the lead as initial instructions

Teammates cannot spawn their own teams or teammates. Only the lead can manage the team hierarchy.

### Shared Task List

A persistent list of work items that coordinates the team:

- All agents (lead and teammates) can see task status
- Tasks have three states: **pending**, **in progress**, **completed**
- Tasks can declare dependencies on other tasks
- A pending task with unresolved dependencies cannot be claimed
- When a task completes, blocked tasks unblock automatically
- Task claiming uses file locking to prevent race conditions when multiple teammates try to claim the same task simultaneously

### Mailbox

A messaging system enabling direct communication between agents:

- **message** — send to one specific teammate
- **broadcast** — send to all teammates simultaneously
- Messages are delivered automatically to recipients
- The lead does not need to poll for updates

---

## How Teams Start

There are two paths to team creation:

1. **User requests a team** — describe the task and team structure in natural language. Claude creates the team based on instructions.
2. **Claude proposes a team** — if Claude determines a task benefits from parallel work, it suggests creating a team. The user confirms before proceeding.

In both cases, the user stays in control. Claude will not create a team without approval.

### What Happens at Spawn

When a teammate is spawned:

1. A new Claude Code instance is created
2. The instance loads project context: CLAUDE.md, MCP servers, skills
3. The spawn prompt from the lead is delivered as the initial message
4. The teammate begins working immediately
5. The teammate inherits the lead's permission settings

The lead's conversation history does **not** carry over. Include all necessary context in the spawn prompt.

---

## Communication Patterns

### Lead to Teammate

The lead sends instructions, assignments, and feedback to individual teammates:

```
Tell the security reviewer to also check for CORS misconfigurations.
```

### Teammate to Lead

Teammates report findings, ask questions, and send completion notifications:

- **Idle notifications** — when a teammate finishes and stops, it automatically notifies the lead
- Teammates can message the lead with questions or intermediate findings

### Teammate to Teammate

Teammates can communicate directly without going through the lead:

```
Have the frontend teammate share the API contract with the backend teammate.
```

This enables the "competing hypotheses" pattern where teammates challenge each other's findings.

### Broadcast

Send a message to all teammates simultaneously:

```
Broadcast to all teammates: the database schema has changed, pull latest before continuing.
```

Use sparingly — token costs scale with team size since each teammate processes the broadcast in its own context.

### User to Teammate

Users can interact with any teammate directly:

- **In-process mode**: Shift+Up/Down to select, type to message
- **Split-pane mode**: click into the teammate's pane

This is useful for course-correction when a teammate goes off-track.

---

## Task Lifecycle

### Task States

```
pending ──► in_progress ──► completed
   │
   └── (blocked by dependencies)
```

A task can be:
- **pending** — not yet started, available for claiming (unless blocked)
- **in_progress** — claimed by a teammate, actively being worked on
- **completed** — work is done

### Dependency Resolution

Tasks can depend on other tasks. The system manages dependencies automatically:

```
Task A: "Set up database schema"
Task B: "Write API endpoints" (depends on A)
Task C: "Write frontend" (depends on B)
Task D: "Write tests for schema" (depends on A)
```

- Tasks B and C are blocked until their dependencies complete
- When Task A completes, Tasks B and D unblock automatically
- Task C remains blocked until Task B completes

### Claiming Tasks

When a teammate finishes its current work:

1. **Lead assigns** — the lead can explicitly assign the next task
2. **Self-claim** — the teammate picks up the next unassigned, unblocked task

File locking prevents two teammates from claiming the same task simultaneously.

### Task Status Lag

Known limitation: teammates sometimes fail to mark tasks as completed, which blocks dependent tasks. If a task appears stuck:

- Check whether the work is actually done
- Update the task status manually
- Tell the lead to nudge the teammate

---

## Delegate Mode

By default, the lead may start implementing tasks itself instead of waiting for teammates. Delegate mode restricts the lead to coordination-only tools:

- Spawning teammates
- Messaging teammates
- Shutting down teammates
- Managing tasks

Toggle delegate mode with **Shift+Tab** after starting a team. This is useful when the lead should focus entirely on orchestration — breaking down work, assigning tasks, reviewing output, and synthesizing results.

---

## Plan Approval Workflow

For risky or complex tasks, require teammates to plan before implementing:

```
Spawn an architect teammate to refactor auth. Require plan approval.
```

The workflow:

1. Teammate works in **read-only plan mode** — can read files and explore but not modify
2. Teammate designs an approach and submits a **plan approval request**
3. Lead reviews the plan
4. Lead **approves** (teammate exits plan mode, begins implementation) or **rejects with feedback** (teammate revises and resubmits)

The lead makes approval decisions autonomously. Influence its judgment with criteria in the original prompt:

```
Only approve plans that include test coverage.
Reject plans that modify the database schema.
Reject plans that add new dependencies.
```

---

## Quality Gate Hooks

Enforce automated rules at key lifecycle points:

### TeammateIdle

Runs when a teammate is about to go idle (finished its current work):

- **Exit code 0** — teammate goes idle normally
- **Exit code 2** — sends feedback to the teammate and keeps it working

Use this to enforce completeness: "if tests aren't passing, keep working."

### TaskCompleted

Runs when a task is being marked as complete:

- **Exit code 0** — task marks as completed normally
- **Exit code 2** — prevents completion and sends feedback

Use this for quality gates: "if lint fails, reject completion."

See [Claude Code hooks documentation](https://code.claude.com/docs/en/hooks) for implementation details.

---

## Token Usage

Agent teams consume significantly more tokens than a single session:

- Each teammate has its own context window
- Token usage scales with the number of active teammates
- Broadcast messages multiply cost by team size
- Each teammate processes project context (CLAUDE.md, etc.) independently

### When the Extra Cost Is Worthwhile

- Research, review, and investigation — parallel exploration finds answers faster
- Independent feature development — teammates work without blocking each other
- Debugging with competing hypotheses — adversarial investigation surfaces root causes

### When a Single Session Is More Cost-Effective

- Sequential tasks with heavy dependencies
- Small tasks that don't justify coordination overhead
- Tasks where a single agent can complete the work in a few minutes

### Cost Optimization Strategies

- Use fewer teammates (2-3 instead of 4-5) unless parallelism clearly helps
- Specify lighter models for straightforward tasks: `"Use Sonnet for each teammate"`
- Use direct messages instead of broadcasts when possible
- Right-size tasks so teammates finish efficiently rather than spinning

---

## Storage and Discovery

### Team Configuration

Stored at `~/.claude/teams/{team-name}/config.json`:

- Contains a `members` array with each teammate's name, agent ID, and agent type
- Teammates can read this file to discover other team members

### Task Storage

Stored at `~/.claude/tasks/{team-name}/`:

- Shared task list accessible to all agents
- Includes task states, dependencies, and ownership

### Cleanup

When the team is done:

1. Shut down all teammates first (they can approve or reject shutdown requests)
2. Ask the lead to clean up: `"Clean up the team"`
3. The lead checks for active teammates and fails if any are still running

Always clean up through the lead — teammates should not run cleanup because their team context may not resolve correctly, potentially leaving resources in an inconsistent state.

---

## Constraints and Boundaries

| Constraint | Detail |
|------------|--------|
| One team per session | A lead can only manage one team at a time. Clean up before starting a new team. |
| No nested teams | Teammates cannot spawn their own teams or teammates. |
| Fixed lead | The creating session is the lead for the team's lifetime. |
| No session resumption | `/resume` and `/rewind` do not restore in-process teammates. |
| Permissions at spawn | All teammates start with the lead's mode. Change individually after spawn. |
