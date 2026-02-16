# Setup and Configuration

Complete guide to enabling and configuring Claude Code agent teams.

---

## Enabling Agent Teams

Agent teams are experimental and disabled by default. Enable by setting the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` environment variable to `1`.

### Option A: Settings File

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### Option B: Shell Environment

Export the variable before launching Claude Code:

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
claude
```

### Verification

After enabling, start a Claude Code session and ask:

```
What tools do you have available?
```

The response should include team-related capabilities. If not, verify the environment variable is set correctly.

---

## Display Modes

Agent teams support two display modes that control how teammates appear in the terminal.

### In-Process Mode

All teammates run inside the main terminal. This is the default when not running in tmux.

- Works in any terminal (VS Code, Windows Terminal, iTerm2, etc.)
- Navigate between teammates with **Shift+Up / Shift+Down**
- Press **Enter** to view a teammate's full session
- Press **Escape** to interrupt a teammate's current turn
- Press **Ctrl+T** to toggle the shared task list

### Split-Pane Mode

Each teammate gets its own terminal pane visible simultaneously. Requires tmux or iTerm2.

- See all teammates' output at once
- Click into any pane to interact directly with that teammate
- Better visibility but requires additional setup

### Configuring Display Mode

The default is `"auto"`, which uses split panes if already running inside a tmux session, and in-process otherwise.

**Override in settings.json:**

```json
{
  "teammateMode": "in-process"
}
```

Valid values: `"auto"`, `"in-process"`, `"tmux"`

The `"tmux"` setting enables split-pane mode and auto-detects whether to use tmux or iTerm2 based on the terminal environment.

**Override per session:**

```bash
claude --teammate-mode in-process
```

### tmux Setup

Install tmux through the system package manager:

```bash
# macOS
brew install tmux

# Ubuntu/Debian
sudo apt-get install tmux

# Verify installation
tmux -V
```

To use split-pane mode with tmux, launch Claude Code inside a tmux session:

```bash
tmux new -s claude
claude
```

To check if already inside tmux: `echo $TMUX` prints a path if inside a session.

> tmux has known limitations on certain operating systems and works best on macOS. Using `tmux -CC` in iTerm2 is the suggested entrypoint.

### iTerm2 Setup

For split-pane mode with iTerm2:

1. Install the [`it2` CLI](https://github.com/mkusaka/it2)
2. Enable the Python API: **iTerm2 > Settings > General > Magic > Enable Python API**

### Unsupported Terminals for Split Panes

Split-pane mode is not supported in:
- VS Code integrated terminal
- Windows Terminal
- Ghostty

Use in-process mode in these terminals.

---

## Permissions

Teammates inherit the lead's permission settings at spawn time:

- If the lead runs with `--dangerously-skip-permissions`, all teammates do too
- After spawning, individual teammate permission modes can be changed
- Per-teammate modes cannot be set at spawn time

### Reducing Permission Prompts

Teammate permission requests bubble up to the lead, which can create friction. To reduce interruptions, pre-approve common operations in permission settings before spawning teammates:

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

---

## Project Context

Teammates load the same project context as a regular Claude Code session:

- **CLAUDE.md** files from the working directory
- **MCP servers** configured for the project
- **Skills** installed and available

Teammates also receive the spawn prompt from the lead. However, they do **not** inherit the lead's conversation history. Include all necessary context in the spawn prompt or rely on CLAUDE.md for shared project knowledge.

---

## Storage Locations

Agent team data is stored locally:

| Data | Location |
|------|----------|
| Team configuration | `~/.claude/teams/{team-name}/config.json` |
| Task list | `~/.claude/tasks/{team-name}/` |

The team config contains a `members` array with each teammate's name, agent ID, and agent type. Teammates can read this file to discover other team members.

---

## Quick Setup Checklist

```bash
# 1. Enable agent teams
# Add to ~/.claude/settings.json:
# { "env": { "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1" } }

# 2. (Optional) Install tmux for split-pane mode
tmux -V  # Check if installed
# brew install tmux  # macOS
# sudo apt-get install tmux  # Ubuntu/Debian

# 3. (Optional) Start in tmux for split panes
tmux new -s claude

# 4. Launch Claude Code
claude

# 5. Verify by asking Claude to create a team
```
