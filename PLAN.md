# LangChain/LangGraph Plugin Architecture Plan

## Executive Summary

After analyzing the latest LangChain documentation (v1, 2025) and the existing plugin structure:

1. **Rename plugin** from `langgraph` → `langchain-ai`
2. **Add RAG skill** with comprehensive vector store coverage
3. **Add LCEL/Chains skill** for non-agent workflows (summarization, extraction)

---

## User Decisions

- **Plugin Name**: `langchain-ai` (broader scope, discoverable)
- **Vector Stores**: All major stores (FAISS, Chroma, pgvector, Pinecone, Qdrant, Weaviate)
- **Scope**: RAG + Advanced Chains (LCEL patterns beyond agents)

---

## Deep Analysis

### Key Finding: LangChain v1 Architecture Shift

The 2025 LangChain documentation reveals a **fundamental architectural change**:

```
LangChain (high-level) → built on → LangGraph (low-level)
```

- `create_agent` is now LangChain's primary API (replaces old `createReactAgent`)
- LangChain agents use LangGraph under the hood for execution
- Legacy chains moved to `langchain_classic`

**Implication**: Your plugin already covers the core LangChain agent API (`create_agent`) because it's fundamentally a LangGraph construct.

### What the Current Plugin Covers

| Area | Coverage | Notes |
|------|----------|-------|
| `create_agent` (LangChain) | ✅ Complete | High-level agent API |
| `StateGraph` (LangGraph) | ✅ Complete | Low-level control |
| Tools & Tool Binding | ✅ Complete | Both APIs |
| State Management | ✅ Complete | TypedDict, reducers |
| HITL | ✅ Complete | Interrupts, approvals |
| Streaming | ✅ Complete | All modes |
| Error Prevention | ✅ Complete | Comprehensive |

### What's Missing: RAG

**RAG (Retrieval-Augmented Generation)** is the major gap:

```python
# RAG Pipeline (NOT covered)
documents = loader.load()                    # Document loading
chunks = splitter.split_documents(documents) # Text splitting
vectorstore.add_documents(chunks)            # Embedding & storing
retriever = vectorstore.as_retriever()       # Retrieval
chain = retriever | prompt | llm | parser    # Generation
```

This is a **different mental model**:
- Agents: Autonomous decision-making with tools
- RAG: Data pipeline for context augmentation

---

## Self-Critique & Options Analysis

### Option A: Add `langchain-rag` Skill to Existing Plugin

**Pros:**
- Single plugin for Python AI development
- Skills auto-select based on task ("build RAG" vs "build agent")
- Shared components (models, prompts) documented once
- Reflects that RAG + Agents are often combined

**Cons:**
- Plugin scope expands (but focused on same ecosystem)
- Plugin name "langgraph" becomes slightly misleading

**Verdict: RECOMMENDED** - This is how users actually work.

### Option B: Create Separate `langchain-rag` Plugin

**Pros:**
- Clear separation of concerns
- Each plugin stays focused

**Cons:**
- Users need two plugins for common workflows (RAG agent)
- Duplicated content (model init, prompts)
- Goes against "LangChain is built on LangGraph" relationship

**Verdict: Not recommended** - Creates artificial separation.

### Option C: Rename Plugin to `langchain-ecosystem` or `ai-python`

**Pros:**
- Accurate scope representation
- Room for future expansion

**Cons:**
- Loses "langgraph" brand recognition
- Breaking change for existing users
- "ai-python" is too generic

**Verdict: Optional enhancement** - Could rename, but not required.

---

## Critique of the "Do We Even Need This?" Question

**Challenge**: Is RAG complex enough to warrant a skill?

**Counter-evidence** (yes, it is):
1. **Chunking is hard**: Strategy selection (recursive, semantic, sentence)
2. **Retrieval quality**: k value, score thresholds, re-ranking
3. **Many gotchas**: Embedding dimension mismatches, document metadata loss
4. **Vector store variance**: Each has different setup patterns
5. **Hybrid patterns**: RAG + agents, multi-query retrieval

**Verdict**: RAG deserves structured guidance, not just code snippets.

---

## Final Recommendation

### Plugin Structure

```
langchainSkills/                     # Directory stays same
├── .claude-plugin/
│   └── plugin.json                  # RENAME plugin to 'langchain-ai'
├── skills/
│   ├── langgraph/                   # EXISTING - Keep as-is (agents)
│   │   ├── SKILL.md
│   │   └── references/
│   ├── langchain-rag/               # NEW - RAG pipelines
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── document-loaders.md
│   │       ├── text-splitters.md
│   │       ├── vector-stores.md     # All 6 stores
│   │       ├── retriever-patterns.md
│   │       └── common-errors.md
│   └── langchain-chains/            # NEW - LCEL & chains
│       ├── SKILL.md
│       └── references/
│           ├── lcel-fundamentals.md
│           ├── summarization.md
│           ├── extraction.md
│           └── output-parsers.md
├── commands/
│   ├── new-agent.md                 # EXISTING
│   ├── add-tool.md                  # EXISTING
│   ├── new-rag.md                   # NEW - Scaffold RAG
│   └── new-chain.md                 # NEW - Scaffold chains
└── agents/
    ├── langgraph-reviewer.md        # EXISTING
    └── rag-reviewer.md              # NEW - Review RAG code
```

### Plugin Naming Decision

**Rename to `langchain-ai`** because:
1. Broader scope now includes RAG + LCEL chains
2. More discoverable for LangChain users
3. Reflects the full ecosystem coverage

### New Skill #1: `langchain-rag`

**Trigger phrases:**
- "Build a RAG pipeline"
- "Load documents from..."
- "Create a retrieval chain"
- "Set up vector store"
- "Chunk documents"

**Content outline:**
1. Quick decision: Which vector store?
2. Document loading patterns
3. Text splitting strategies
4. Vector store setup (FAISS, Chroma, Pinecone, pgvector, Qdrant, Weaviate)
5. Retriever configuration
6. RAG chain composition (LCEL)
7. Common errors & fixes

### New Command: `/langchain-ai:new-rag`

Scaffold a complete RAG pipeline with:
- Document loader selection
- Text splitter configuration
- Vector store initialization
- Retriever setup
- Chain composition

### New Agent: `rag-reviewer`

Checks for:
- Missing metadata preservation
- Suboptimal chunk sizes
- Retriever k values
- Embedding dimension mismatches
- Missing error handling for empty retrievals

### New Skill #2: `langchain-chains`

**Trigger phrases:**
- "Create a summarization chain"
- "Build an extraction pipeline"
- "Use LCEL to..."
- "Chain multiple prompts"
- "Create a data pipeline"

**Content outline:**
1. LCEL fundamentals (pipe operator, RunnablePassthrough)
2. Summarization patterns (stuff, map-reduce, refine)
3. Extraction patterns (structured output, Pydantic)
4. Prompt chaining (multi-step reasoning)
5. Parallel execution (RunnableParallel)
6. Fallbacks and retries
7. Output parsing patterns

### New Command: `/langchain-ai:new-chain`

Scaffold common chain patterns:
- Summarization chain
- Extraction chain
- Multi-step reasoning chain

---

## Implementation Plan

### Phase 1: Rename Plugin
1. Update `.claude-plugin/plugin.json`:
   - Change `name` from "langgraph" to "langchain-ai"
   - Update description: "Build AI applications with LangChain and LangGraph. Agents, RAG pipelines, and LCEL chains with best practices."
   - Add keywords: "langchain", "rag", "retrieval", "vector", "lcel", "chains"

### Phase 2: Create langchain-rag Skill
1. Create `skills/langchain-rag/SKILL.md` with:
   - Quick start RAG example
   - Decision matrix for vector stores (all 6)
   - Critical rules and gotchas
   - Links to reference docs

2. Create reference docs:
   - `document-loaders.md` - PDF, web, CSV, JSON patterns
   - `text-splitters.md` - Recursive, semantic, sentence strategies
   - `vector-stores.md` - FAISS, Chroma, pgvector, Pinecone, Qdrant, Weaviate
   - `retriever-patterns.md` - Hybrid, multi-query, contextual compression
   - `common-errors.md` - RAG-specific error catalog

### Phase 3: Create langchain-chains Skill
1. Create `skills/langchain-chains/SKILL.md` with:
   - LCEL quick start
   - When to use chains vs agents
   - Common patterns overview

2. Create reference docs:
   - `lcel-fundamentals.md` - Pipe operator, RunnablePassthrough, RunnableParallel
   - `summarization.md` - Stuff, map-reduce, refine strategies
   - `extraction.md` - Structured output, Pydantic models
   - `output-parsers.md` - StrOutputParser, JsonOutputParser, PydanticOutputParser

### Phase 4: Create Commands
1. Create `commands/new-rag.md`:
   - Ask vector store preference
   - Generate complete RAG scaffold
   - Include best practices

2. Create `commands/new-chain.md`:
   - Ask chain type (summarization, extraction, custom)
   - Generate appropriate template

### Phase 5: Create rag-reviewer Agent
1. Create `agents/rag-reviewer.md`:
   - Check chunk sizes, overlap
   - Check retriever configuration
   - Check embedding consistency
   - Check error handling

### Phase 6: Update README
1. Document all three skills (agents, RAG, chains)
2. Show when to use each
3. Installation and usage examples

---

## Verification

After implementation:
1. **Skill triggers**:
   - "Help me build a RAG pipeline" → langchain-rag skill
   - "Create a summarization chain" → langchain-chains skill
   - "Build an agent with tools" → langgraph skill (existing)
2. **Commands**:
   - `/langchain-ai:new-rag` scaffolds RAG pipeline
   - `/langchain-ai:new-chain` scaffolds LCEL chain
   - `/langchain-ai:new-agent` still works (existing)
3. **Reviewer**:
   - Write sample RAG code with issues → rag-reviewer catches them
4. **No regressions**:
   - Existing langgraph skill triggers correctly
   - Existing commands work with new plugin name

---

## Files to Create/Modify

### Modified Files (2)
- `.claude-plugin/plugin.json` - Rename to langchain-ai, update description/keywords
- `README.md` - Document all three skills

### New Files (15)

**langchain-rag skill (6 files)**:
- `skills/langchain-rag/SKILL.md`
- `skills/langchain-rag/references/document-loaders.md`
- `skills/langchain-rag/references/text-splitters.md`
- `skills/langchain-rag/references/vector-stores.md`
- `skills/langchain-rag/references/retriever-patterns.md`
- `skills/langchain-rag/references/common-errors.md`

**langchain-chains skill (5 files)**:
- `skills/langchain-chains/SKILL.md`
- `skills/langchain-chains/references/lcel-fundamentals.md`
- `skills/langchain-chains/references/summarization.md`
- `skills/langchain-chains/references/extraction.md`
- `skills/langchain-chains/references/output-parsers.md`

**Commands (2 files)**:
- `commands/new-rag.md`
- `commands/new-chain.md`

**Agents (1 file)**:
- `agents/rag-reviewer.md`

---

## Estimated Effort

| Phase | Files | Complexity |
|-------|-------|------------|
| 1. Rename plugin | 1 | Low |
| 2. RAG skill | 6 | High (most content) |
| 3. Chains skill | 5 | Medium |
| 4. Commands | 2 | Medium |
| 5. Agent | 1 | Low |
| 6. README | 1 | Low |

**Total: 16 files** (2 modified, 14 new)
