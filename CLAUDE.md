# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Claude Code plugin for LangChain and LangGraph development. It provides skills, commands, and reviewer agents to help build AI applications with best practices.

## Plugin Structure

```
langchain-community-plugin/
├── .claude-plugin/
│   ├── plugin.json               # Plugin manifest (name, version, keywords)
│   └── .mcp.json                 # MCP server configuration (LangChain docs)
├── skills/                       # Core guidance loaded contextually
│   ├── langgraph/               # AI agents with StateGraph and create_agent
│   ├── langchain-rag/           # RAG pipelines with vector stores
│   └── langchain-chains/        # LCEL chains for summarization/extraction
├── commands/                     # Slash commands for scaffolding
│   ├── new-agent.md             # /langchain-community:new-agent
│   ├── add-tool.md              # /langchain-community:add-tool
│   ├── new-rag.md               # /langchain-community:new-rag
│   └── new-chain.md             # /langchain-community:new-chain
└── agents/                       # Reviewer subagents (haiku model)
    ├── langgraph-reviewer.md    # Reviews agent code for common mistakes
    └── rag-reviewer.md          # Reviews RAG code for common mistakes
```

## Architecture

**Skills** (`skills/<name>/SKILL.md`): Main guidance documents with YAML frontmatter containing `name` and `description`. Each skill has a `references/` subdirectory with detailed documentation for specific patterns (e.g., state-patterns.md, vector-stores.md).

**Commands** (`commands/<name>.md`): Slash commands that scaffold new code. Use YAML frontmatter for command metadata.

**Agents** (`agents/<name>.md`): Reviewer agents that analyze code for issues. Configured with YAML frontmatter specifying `name`, `description`, `model` (haiku), `color`, and available `tools`.

**MCP Servers** (`.claude-plugin/.mcp.json`): Configures external MCP servers. The plugin includes the official LangChain docs MCP server (`https://docs.langchain.com/mcp`) for accessing up-to-date documentation directly.

## Key Patterns

- Skills are triggered by natural language (e.g., "build an agent" triggers langgraph skill)
- Commands are invoked with `/langchain-community:<command-name>`
- Reviewer agents trigger proactively after writing relevant code or on explicit request
- All Python code examples target LangChain v1 (2025) architecture where LangChain agents use LangGraph under the hood

## Development Guidelines

When modifying this plugin:

1. **Skills**: Keep SKILL.md focused on quick decisions and gotchas. Put detailed patterns in `references/` files.
2. **Agents**: Use haiku model for reviewers to minimize cost. Include concrete examples in the description for proper triggering.
3. **Commands**: Structure prompts to ask clarifying questions before generating code.
4. **Code examples**: Always use TypedDict for LangGraph state (not Pydantic). Always include checkpointer for HITL/memory patterns.
