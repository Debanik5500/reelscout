# Agent Instructions

## Token Optimization Stack

This project uses a **three-layer token optimization strategy** to reduce token consumption by up to 90% while maintaining quality:

1. **DCP (Dynamic Context Pruning)** - Conversation history compression and management
2. **opencode-ctx** - Aggressive prompt/context compression before sending to provider
3. **OpenCode Native Settings** - Compaction, tool output limits, and input restrictions

All three systems work together automatically. Follow the guidelines below to use them effectively.

---

## Layer 1: Dynamic Context Pruning (DCP)

**Plugin**: `@tarquinen/opencode-dcp`  
**Purpose**: Manages conversation context through compression, deduplication, and error pruning  
**Savings**: Reduces context by compressing stale conversation history into high-fidelity summaries

### When to Compress

- **Task boundaries**: After completing a discrete task (bug fix, feature implementation, refactor)
- **Context awareness**: Above 50k tokens, be mindful. Above 100k tokens, compression becomes important
- **Natural breaks**: After successful test runs, completed reviews, or finishing exploratory work

### Using the Compress Tool

The model has access to a `compress` tool that creates technical summaries of conversation sections.

**Range mode** (default): Compress contiguous conversation spans
```
Use compress tool when:
- A feature is complete and tested
- Debug session resolved the issue
- Exploration phase finished, moving to implementation
```

**Message mode** (experimental): Compress individual messages surgically
```
Use for:
- Removing large file dumps that are no longer needed
- Pruning verbose output from completed operations
- Selective cleanup while keeping recent context
```

### What Gets Protected

Automatically preserved during compression:
- Subagent outputs (`task` tool)
- Skill invocations
- Todo lists (`todowrite`, `todoread`)
- File write operations (`write`)
- File edits (`edit`)
- User messages (if `protectUserMessages: true` in config)
- Content in `<protect>...</protect>` tags (if `protectTags: true`)

### DCP Commands

- `/dcp` - Open DCP control panel with context stats
- `/dcp-compress [focus]` - Manually trigger compression with optional focus

### DCP Configuration

Current settings:
- **Compression mode**: range (compresses conversation spans)
- **Min context threshold**: 50,000 tokens (below this, compression is less likely)
- **Max context threshold**: 100,000 tokens (above this, compression strongly encouraged)
- **Auto-cleanup**: Deduplication and error purging enabled
- **Protected tools**: task, skill, todowrite, todoread, write, edit

---

## Layer 2: opencode-ctx Plugin

**Plugin**: `opencode-ctx`  
**Purpose**: Aggressive prompt/context compression **before** sending to provider  
**Savings**: 38-41% on Anthropic, 45-50% on OpenAI by compressing overhead and tool noise

### What It Does Automatically

1. **Tool description compression** - Shrinks OpenCode tool definitions
2. **System prompt compression** - Reduces the fixed system prompt overhead
3. **Message-history tool trim** - Collapses stale old tool output and duplicate reads
4. **Tool-output cleaning** - Removes ANSI noise, progress bars, blank lines
5. **Anthropic cache TTL upgrade** - Keeps cache breakpoints alive longer (1h vs 5m)
6. **Deduplication** - Removes duplicate tool calls with identical arguments
7. **Superseded reads** - Collapses multiple reads of the same file

### Default Behavior

All optimizations are **enabled by default**. The plugin works transparently with no manual intervention.

### Environment Variables (Optional Tuning)

Only adjust these if you notice quality issues:

```bash
# Core toggles (all ON by default)
OPENCODE_CTX_PLUGIN=0          # Disable entire plugin
OPENCODE_CTX_TRIM=0            # Skip tool/system prompt compression
OPENCODE_CTX_MSGS=0            # Skip message-history trimming
OPENCODE_CTX_CLEAN=0           # Skip tool-output cleaning
OPENCODE_CTX_TTL=0             # Skip Anthropic cache TTL upgrade
OPENCODE_CTX_DEBUG=1           # Enable debug logging to stderr

# Tuning knobs
OPENCODE_CTX_MSGS_KEEP=3       # Preserve last N tool outputs intact (default: 3)
OPENCODE_CTX_MSGS_CAP=600      # Byte threshold before trimming (default: 600)
OPENCODE_CTX_MSGS_HEAD=300     # Head bytes kept when trimming (default: 300)
OPENCODE_CTX_MSGS_TAIL=150     # Tail bytes kept when trimming (default: 150)
```

### Expected Impact

**Single-turn request**: ~40% fewer tokens  
**Multi-turn session**: ~40-50% fewer tokens  
**Context window reach**: 2x longer sessions before hitting limits

If the model loses context on long sessions, increase `OPENCODE_CTX_MSGS_KEEP=5` or disable message trimming with `OPENCODE_CTX_MSGS=0`.

---

## Layer 3: OpenCode Native Settings

**Configuration**: `opencode.json`  
**Purpose**: Core OpenCode compaction and output limiting  
**Savings**: Aggressive early compaction and tool output caps

### Active Settings

```json
{
  "compaction": {
    "auto": true,        // Auto-compact when context fills
    "prune": true,       // Remove old tool outputs
    "reserved": 8000,    // Buffer before compaction (default: 20k)
    "tail_turns": 15     // Keep last 15 turns during compaction
  },
  "tool_output": {
    "max_lines": 100,    // Limit tool output to 100 lines
    "max_bytes": 4096    // Limit tool output to 4KB
  }
}
```

### How Compaction Works

**Trigger formula**:
```
usable = limit.input - reserved
Compaction triggers when: current_tokens >= usable

With reserved=8000:
- If input limit is 25k: compaction at 17k tokens (~1-2 messages)
- If no input limit: compaction at context_window - max_output
```

### Tool Output Limiting

Large tool outputs are automatically truncated:
- **Max 100 lines** per tool call
- **Max 4KB** per tool output
- Helps prevent context explosion from large file reads or bash outputs

---

## Combined Impact

**Layer 1 (DCP)**: Compresses conversation history into summaries  
**Layer 2 (opencode-ctx)**: Compresses prompts/tools before provider sends  
**Layer 3 (Native)**: Limits tool output and triggers early compaction

**Total reduction**: Up to 90% token savings in long sessions with tool-heavy work

### Context Budget Awareness

Your models have these limits:
- Most coding models: 1,048,576 tokens (1M)
- Vision models: 200,000 tokens (200k)

With this stack:
- **200k context**: ~108 turns → **~225 turns** (2.1x longer)
- **1M context**: ~610 turns → **~1,206 turns** (2x longer)

---

## Best Practices

### 1. Complete Tasks Before Compressing
Don't compress mid-task. Finish work, verify it, then compress.

### 2. Compress Early and Often
Don't wait until context is maxed. Compress at natural boundaries.

### 3. Trust the Summaries
DCP creates technical summaries that preserve important details. Compressed content includes protected tool outputs.

### 4. Monitor Token Usage
Check `/dcp` panel regularly to see context size and compression stats.

### 5. Use Focus Hints
When manually compressing with `/dcp-compress`, add focus text to guide what should be compressed.

### 6. Let opencode-ctx Work Silently
The opencode-ctx plugin works automatically. Don't disable it unless you see quality issues.

---

## Example Workflow

```
1. Start feature work (context: 30k tokens)
   → opencode-ctx: Compresses each request by ~40%
   → Native: Tool outputs limited to 100 lines/4KB

2. Implement and test feature (context: 65k tokens)
   → DCP: Monitoring, no compression yet (below 100k threshold)
   → opencode-ctx: Still compressing each request

3. Feature complete, tests pass
   → DCP: Use compress tool to summarize implementation discussion
   → Context drops to 40k tokens with high-fidelity summary

4. Continue with next task (context: 40k tokens)
   → Cycle repeats with all three layers active
```

---

## Troubleshooting

### Context Growing Too Fast
- Compress more frequently at task boundaries with DCP
- Check if large tool outputs are being generated
- Consider increasing `tool_output.max_bytes` if truncation causes issues

### Lost Important Information
- Check if tool output was protected by DCP
- Adjust `protectedTools` in `~/.config/opencode/dcp.jsonc`
- Increase `OPENCODE_CTX_MSGS_KEEP` for opencode-ctx

### Model Not Compressing
- Context may be below 50k threshold
- Check if manual mode is enabled in DCP
- Use `/dcp` to verify DCP status

### Quality Issues After Compression
- Disable specific opencode-ctx optimizations with environment variables
- Increase `OPENCODE_CTX_MSGS_KEEP` to preserve more tool outputs
- Disable message trimming entirely: `OPENCODE_CTX_MSGS=0`

---

## Verification

To verify all systems are active:

```bash
# Check DCP is loaded
# You'll see /dcp command available

# Check opencode-ctx is active
OPENCODE_CTX_DEBUG=1 opencode run "hi"
# Should show: [ctx-plugin] active: ...

# Check native settings
# View opencode.json for compaction and tool_output settings
```

---

## Configuration Locations

- **DCP config**: `~/.config/opencode/dcp.jsonc` (auto-created on first run)
- **opencode-ctx**: Environment variables only
- **Native settings**: `~/.config/opencode/opencode.json`
- **Project instructions**: `AGENTS.md` (this file)
