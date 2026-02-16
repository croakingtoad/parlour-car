---
name: claude-code-agent-teams
description: "Guide for orchestrating Claude Code agent teams — multiple parallel Claude Code sessions coordinated by a team lead. Use this skill when the user mentions agent teams, teammates, parallel agents, multi-agent workflows, spawning agents, coordinating agents, delegate mode, plan approval for teammates, TeammateIdle or TaskCompleted hooks, or wants to break a task into parallel independent work streams. Also trigger on questions about tmux split-pane mode, in-process teammate mode, Shift+Up/Down agent switching, shared task lists, inter-agent messaging, or designing tasks for multi-agent decomposition. This is an experimental feature requiring CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS to be enabled."
---

# Claude Code Agent Teams

Agent teams coordinate multiple Claude Code sessions working together. One session acts as the **team lead**, spawning **teammates** that work independently in their own context windows and communicate directly with each other through a shared task list and mailbox messaging system.

> Agent teams are experimental and disabled by default. Enable with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`. See [Setup and Configuration](./references/setup-and-configuration.md) for details.

## When to Use Agent Teams

Agent teams add coordination overhead and consume significantly more tokens than a single session. Use them when parallel exploration adds real value:

- **Research and review** — teammates investigate different aspects simultaneously, share and challenge findings
- **New modules or features** — teammates each own a separate piece without stepping on each other
- **Debugging with competing hypotheses** — teammates test different theories in parallel, converge on answers faster
- **Cross-layer coordination** — changes spanning frontend, backend, and tests, each owned by a different teammate

Avoid agent teams for sequential tasks, same-file edits, work with many dependencies, or tasks a single session handles efficiently. For focused workers that only report results back, use [subagents](https://code.claude.com/docs/en/sub-agents) instead.

### Agent Teams vs Subagents

|                   | Subagents                                    | Agent Teams                                     |
| :---------------- | :------------------------------------------- | :---------------------------------------------- |
| **Context**       | Own window; results return to caller         | Own window; fully independent                   |
| **Communication** | Report back to main agent only               | Teammates message each other directly            |
| **Coordination**  | Main agent manages all work                  | Shared task list with self-coordination          |
| **Best for**      | Focused tasks where only the result matters  | Complex work requiring discussion/collaboration  |
| **Token cost**    | Lower: results summarized to main context    | Higher: each teammate is a separate instance     |

## Starting a Team

Describe the task and team structure in natural language. Claude creates the team, spawns teammates, and coordinates work:

```
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

Claude can also propose a team if it determines a task benefits from parallel work. It always asks for confirmation first.

## Core Concepts

### Architecture

| Component     | Role                                                                    |
| :------------ | :---------------------------------------------------------------------- |
| **Team lead** | Main session that creates the team, spawns teammates, coordinates work  |
| **Teammates** | Separate Claude Code instances working on assigned tasks                |
| **Task list** | Shared work items teammates claim and complete, with dependency tracking |
| **Mailbox**   | Messaging system for direct communication between agents                |

Teams and tasks are stored locally at `~/.claude/teams/{team-name}/config.json` and `~/.claude/tasks/{team-name}/`. See [Architecture and Communication](./references/architecture-and-communication.md) for internals.

### Display Modes

- **In-process** (default) — all teammates run inside the main terminal. Navigate with Shift+Up/Down. Works in any terminal.
- **Split panes** — each teammate gets its own pane. Requires tmux or iTerm2. Click into any pane to interact directly.

Default is `"auto"`: split panes if already in tmux, in-process otherwise. Override with `teammateMode` in settings or `--teammate-mode` flag.

### Communication

Teammates do not inherit the lead's conversation history. Each loads project context (CLAUDE.md, MCP servers, skills) plus the spawn prompt from the lead.

- **Direct messages** — send to one specific teammate
- **Broadcast** — send to all teammates (use sparingly, costs scale with team size)
- **Idle notifications** — teammates automatically notify the lead when they finish
- **Shared task list** — all agents see task status and claim available work

### Task Management

Tasks have three states: **pending**, **in progress**, and **completed**. Tasks can depend on other tasks — a pending task with unresolved dependencies cannot be claimed until those complete.

- The lead assigns tasks explicitly, or teammates self-claim the next unassigned, unblocked task
- Task claiming uses file locking to prevent race conditions
- When a task completes, blocked tasks unblock automatically

## Lead Controls

### Delegate Mode

Restrict the lead to coordination-only tools (spawning, messaging, shutting down teammates, managing tasks) so it focuses on orchestration instead of implementing work itself. Toggle with **Shift+Tab**.

### Plan Approval

Require teammates to plan before implementing. The teammate works in read-only plan mode until the lead approves:

```
Spawn an architect teammate to refactor the authentication module.
Require plan approval before they make any changes.
```

The lead approves or rejects with feedback autonomously. Influence judgment with criteria: "only approve plans that include test coverage" or "reject plans that modify the database schema."

### Quality Gate Hooks

Enforce rules with hooks:

- **`TeammateIdle`** — runs when a teammate is about to go idle. Exit code 2 sends feedback and keeps the teammate working.
- **`TaskCompleted`** — runs when a task is being marked complete. Exit code 2 prevents completion and sends feedback.

### Interacting with Teammates

- **In-process**: Shift+Up/Down to select, type to message. Enter to view session, Escape to interrupt. Ctrl+T toggles task list.
- **Split panes**: click into any pane to interact directly.

### Shutting Down

Ask the lead to shut down specific teammates (they can approve or reject). Then ask the lead to clean up the team. Always clean up through the lead — teammates should not run cleanup.

## Task Decomposition

The key to effective agent teams is decomposing tasks into independent work streams. See [Task Decomposition Patterns](./references/task-decomposition-patterns.md) for detailed examples.

**Core principles:**

1. **Maximize independence** — zero dependencies between teammates is ideal
2. **Minimize write conflicts** — each teammate owns distinct files or directories
3. **Prefer read-heavy work** — analysis, review, and research parallelize cleanly
4. **Right-size tasks** — 5-6 tasks per teammate keeps everyone productive; too small wastes overhead, too large creates bottlenecks

**Anti-patterns to avoid:**

- Sequential chains where step 2 depends on step 1's output
- Multiple teammates editing the same file
- Micro-tasks not worth the spawn overhead
- Prompts that assume shared conversation context (teammates start fresh)

## Best Practices

- Give teammates enough context in spawn prompts — they load CLAUDE.md but not the lead's conversation
- Specify teammate count and models explicitly when needed: `"Create a team with 4 teammates. Use Sonnet for each."`
- Tell the lead to wait for teammates before proceeding if it starts implementing itself
- Start with research and review tasks to learn the coordination model before attempting parallel implementation
- Avoid file conflicts by ensuring each teammate owns different files
- Monitor and steer — check progress, redirect failing approaches, synthesize findings as they arrive
- Pre-approve common operations in permission settings to reduce interruption from teammate permission prompts

## Quick Reference

| Action | How |
|--------|-----|
| Enable feature | Set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` to `1` in env or settings.json |
| Start a team | Describe task and team structure in natural language |
| Navigate teammates | Shift+Up / Shift+Down (in-process) or click pane (split) |
| Toggle task list | Ctrl+T |
| Delegate mode | Shift+Tab |
| View teammate session | Enter (in-process mode) |
| Interrupt teammate | Escape (in-process mode) |
| Specify models | Include in prompt: `"Use Sonnet for each teammate"` |

## Reference Documentation

- [Setup and Configuration](./references/setup-and-configuration.md) — enabling, display modes, permissions, settings
- [Architecture and Communication](./references/architecture-and-communication.md) — internals, messaging, task lifecycle, token usage
- [Task Decomposition Patterns](./references/task-decomposition-patterns.md) — detailed patterns for different task types
- [Use Case Examples](./references/use-case-examples.md) — end-to-end workflows with prompts
- [Troubleshooting](./references/troubleshooting.md) — common issues, limitations, diagnostics

Official documentation: https://code.claude.com/docs/en/agent-teams
