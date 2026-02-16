# Troubleshooting

Diagnosing and resolving common issues with Claude Code agent teams, plus known limitations.

---

## Setup Issues

### Agent Teams Not Available

**Symptom:** Claude doesn't offer to create a team or says the feature isn't available.

**Checklist:**

| Check | How | Expected |
|-------|-----|----------|
| Feature flag set | Check `~/.claude/settings.json` for `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | `"1"` in the `env` block |
| Environment variable | `echo $CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | `1` |

**Fix:** Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Then restart Claude Code.

### Teammates Not Appearing

**Symptom:** Asked Claude to create a team but no teammates show up.

**Possible causes:**

- **In-process mode:** Teammates may be running but not visible. Press **Shift+Down** to cycle through active teammates.
- **Task too simple:** Claude decides whether to spawn teammates based on the task. If the task doesn't warrant a team, Claude may handle it solo. Be explicit: "Create an agent team with 3 teammates."
- **Split-pane mode issues:** If expecting split panes, verify tmux is installed:

```bash
which tmux
tmux -V
```

For iTerm2, verify the `it2` CLI is installed and the Python API is enabled.

### Split Panes Not Working

**Symptom:** Teammates run in-process when split panes were expected.

**Causes and fixes:**

- Default is `"auto"` — uses split panes only if already in tmux. Start in tmux first or set `teammateMode` to `"tmux"`.
- Split panes not supported in: VS Code integrated terminal, Windows Terminal, Ghostty. Use in-process mode.
- For iTerm2: ensure `it2` CLI is installed and Python API is enabled.

---

## Runtime Issues

### Lead Implements Instead of Delegating

**Symptom:** The lead starts writing code itself instead of assigning work to teammates.

**Fixes:**

- Enable **delegate mode** with **Shift+Tab** — restricts the lead to coordination-only tools
- Tell the lead explicitly: "Wait for your teammates to complete their tasks before proceeding"
- Structure prompts to emphasize coordination: "Your role is to coordinate the team, not to implement directly"

### Lead Shuts Down Before Work Is Done

**Symptom:** The lead decides the team is finished while tasks are still in progress.

**Fix:** Tell the lead to keep going. Also check whether tasks appear stuck (see below).

### Too Many Permission Prompts

**Symptom:** Frequent interruptions from teammate permission requests.

**Fix:** Pre-approve common operations in permission settings before spawning teammates:

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Bash(git status)",
      "Bash(npm test*)"
    ]
  }
}
```

### Teammates Stopping on Errors

**Symptom:** Teammate encounters an error and stops instead of recovering.

**Fixes:**

1. Navigate to the teammate (Shift+Up/Down or click pane)
2. Review the error and give additional instructions
3. If unrecoverable, spawn a replacement to continue the work

### Teammate Runs Indefinitely

**Symptom:** A teammate keeps working long after it should have finished.

**Fixes:**

1. Navigate to the teammate's pane
2. Observe what it's doing — stuck in a loop? exploring tangents?
3. Intervene: "Stop. Summarize your findings so far and mark the task complete."
4. If unresponsive in split-pane mode: `tmux kill-pane`

---

## Task and Coordination Issues

### Tasks Stuck as "In Progress"

**Symptom:** A task shows as in progress but the teammate has stopped working.

**Known limitation:** Teammates sometimes fail to mark tasks as completed, blocking dependent tasks.

**Fixes:**

- Check whether the work is actually done (look at output files)
- Tell the lead to update the task status manually
- Tell the lead to nudge the teammate: "Ask the reviewer to mark their task complete"

### Dependent Tasks Not Unblocking

**Symptom:** Task A is done but Task B (depends on A) stays blocked.

**Cause:** Task A not marked complete. See "Tasks Stuck" above.

**Fix:** Have the lead verify and update task statuses. Dependencies unblock automatically once the blocking task is marked complete.

### Teammates Claiming the Same Task

**Symptom:** Two teammates start working on the same task.

**This should not happen** — task claiming uses file locking. If it does occur:

1. Tell one teammate to stop and pick a different task
2. Check for filesystem permission issues preventing locking
3. Report as a bug if reproducible

---

## Communication Issues

### Messages Not Delivered

**Symptom:** Teammate sends a message but the recipient doesn't react.

**Checks:**

- Messages are delivered automatically, no relay needed
- Recipient may be mid-tool-call; messages arrive after the current turn
- Check recipient is still running (hasn't shut down or crashed)

### Broadcast Overuse

**Symptom:** High token consumption from messaging.

**Fix:** Use direct messages for teammate-specific info. Reserve broadcasts for team-wide changes. Token cost scales linearly with team size per broadcast.

---

## File Conflict Issues

### Write Conflicts

**Symptom:** Multiple teammates modified the same file; last write wins.

**Detection:**

```bash
git diff         # See what changed
git log --oneline -5  # Recent changes
```

**Prevention strategies:**

| Strategy | When to use |
|----------|------------|
| Directory isolation — each teammate owns distinct dirs | Module-based work |
| Temp file output — write to unique paths | Analysis/review tasks |
| Read-only constraint — "Do not modify source files" | Code review |
| Lead merges — teammates suggest, lead applies | Shared files unavoidable |

### Import/Dependency Conflicts

**Prevention:** Include in spawn prompts:

```
Do not modify package.json, requirements.txt, or any dependency files.
If you need a new dependency, note it in your report and the lead will install.
```

### Git Conflicts

**Prevention:** Don't have teammates commit to git. Let the lead handle all git operations after synthesizing results.

---

## Cleanup Issues

### Orphaned tmux Sessions

**Symptom:** tmux sessions persist after the team ends.

```bash
tmux ls
tmux kill-session -t <session-name>
```

### Cleanup Fails

**Symptom:** "Clean up the team" fails with active teammates.

**Fix:** Shut down all teammates first, then clean up. Teammates can approve or reject shutdown. Always clean up through the lead — teammates should not run cleanup.

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| **No session resumption** | `/resume` and `/rewind` do not restore in-process teammates. After resuming, the lead may message non-existent teammates. Tell the lead to spawn new ones. |
| **Task status lag** | Teammates sometimes fail to mark tasks complete, blocking dependents. Check and update manually. |
| **Slow shutdown** | Teammates finish current request/tool call before shutting down. |
| **One team per session** | Clean up the current team before starting a new one. |
| **No nested teams** | Teammates cannot spawn their own teams. |
| **Fixed lead** | Cannot promote a teammate or transfer leadership. |
| **Permissions at spawn** | All teammates start with lead's mode. Change individually after spawn only. |
| **Split pane terminal support** | Not supported in VS Code terminal, Windows Terminal, or Ghostty. |

---

## Diagnostic Commands

```bash
# Check feature flag
cat ~/.claude/settings.json | grep AGENT_TEAMS

# Check if in tmux
echo $TMUX

# List tmux sessions and panes
tmux list-sessions
tmux list-panes

# Check team and task storage
ls ~/.claude/teams/
ls ~/.claude/tasks/

# System resources
free -h          # Memory
nproc            # CPU cores
df -h            # Disk space

# Kill specific pane
tmux kill-pane -t {pane_number}

# Kill all panes except current
tmux kill-pane -a

# Nuclear: kill all tmux
tmux kill-server
```

---

## Common Failure Patterns

### "Said complete but nothing happened"

Teammate claims task is done, but no files changed. The teammate may have analyzed without writing output. Fix: "Write your findings to reports/security-review.md. The file doesn't exist yet."

### "Teammate went off-track"

Working on something unrelated. Insufficient scope in spawn prompt. Fix: Navigate to pane, message: "Stop. Focus only on src/auth/. Ignore other directories."

### "Inconsistent output across teammates"

No shared conventions. Fix: Add conventions to CLAUDE.md (loaded automatically by all teammates) or include format requirements in each spawn prompt.

### "High cost, low value"

Task didn't benefit from parallelism. Reserve teams for tasks with clear parallel benefit. Start with research/review before attempting parallel implementation.
