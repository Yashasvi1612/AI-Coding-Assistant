---
title: AI Coding Assistant
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# AI Coding Assistant (Mini Copilot)

An agentic coding assistant: given a request, the LLM itself decides which
tool(s) to invoke — explain code, fix a bug, generate tests, generate docs,
review a PR diff, or chat with an indexed GitHub repo (RAG) — rather than
following one fixed prompt. It can also chain tools (e.g. fix a bug, then
generate a test for the fix).

## Why this is "agentic" and not just RAG

A plain RAG app always does: retrieve → stuff into prompt → answer.
Here, the LLM is given a menu of tools (`tools.py` → `TOOL_SCHEMAS`) and
decides for itself, per request:
- whether to call a tool at all,
- which one (or several) fit the request,
- what arguments to pass them,
- and how to combine results into a final answer.

`chat_with_repo` is the one RAG-backed tool — it's invoked automatically
by the agent whenever you ask about "the repo" / a file / a function
without pasting code yourself.

## Architecture

```
User message
     │
     ▼
Groq LLM (Llama 3.1 8B) + tool schemas ──► decides which tool(s) to call
     │
     ├── explain_code / fix_bug / generate_tests / generate_docs / review_pr_diff
     │        (direct focused LLM call on the user's pasted code/diff)
     │
     └── chat_with_repo
              │
              ▼
      FAISS similarity search over
      embedded repo chunks (sentence-transformers)
              │
              ▼
      retrieved code context ──► LLM answer, grounded in real repo content
     │
     ▼
Final synthesized answer back to user
```

## Files

- `indexer.py` — clones a GitHub repo, chunks source files, builds a FAISS index
- `tools.py` — the six coding-assistant skills + their function-calling schemas
- `agent.py` — the agent loop: sends tools to the LLM, executes whichever it picks, feeds results back
- `app.py` — Gradio UI (repo indexing + chat)

## Setup (local)

1. Get a free Groq API key: https://console.groq.com/keys
2. `pip install -r requirements.txt`
3. `export GROQ_API_KEY=your_key_here`
4. `python app.py`
5. Optionally paste a public GitHub repo URL and click "Index Repo", then chat

## Example prompts to try

- Paste a function and say: *"Explain this and write tests for it"* (chains two tools)
- Index a repo, then ask: *"What does the auth module do?"*
- Paste a `git diff` output and say: *"Review this PR"*
- Paste buggy code and say: *"This throws an IndexError, fix it"*

## Deploy to Hugging Face Spaces

1. New Space → SDK: Gradio
2. Push all four `.py` files + `requirements.txt`
3. Add `GROQ_API_KEY` under Space Settings → Secrets

## Limitations / what I'd improve with more time

- `chat_with_repo` uses flat L2 search — fine for one repo, would need a
  persistent vector DB for multi-repo/multi-session use.
- No streaming responses yet.
- Tool-calling loop is single-round (executes tools once, then answers) —
  could be extended to a full multi-round ReAct loop for more complex,
  multi-step requests.
- No sandboxed code execution — `generate_tests`/`fix_bug` output isn't
  auto-run and verified yet; that would be the natural next step (agent
  writes code, runs it, reads the error, iterates).

---

## Submission write-up 

**Project: AI Coding Assistant (Agentic Framework)**

Built an agentic coding assistant that mirrors tools like GitHub Copilot,
but is explicitly agentic rather than a single fixed prompt. The assistant
exposes six capabilities as callable tools — explain code, fix bugs,
generate tests, generate documentation, review PR diffs, and chat with a
GitHub repository — and an LLM (Llama 3.1 8B via Groq, using function
calling) decides at runtime which tool(s) to invoke based on the user's
request, including chaining multiple tools when needed (e.g. fixing a bug
and generating a regression test for it).

The "chat with repo" capability is RAG-backed: the target repository is
cloned, source files are chunked and embedded locally with
sentence-transformers, and indexed in FAISS; when the agent decides a
question is about the codebase, it retrieves the most relevant chunks and
grounds its answer in the actual repo content rather than general
knowledge.

This demonstrates the core skills behind agentic GenAI systems: tool/function
definition, LLM-driven tool selection and chaining, and RAG grounding — combined
into one deployable, interactive assistant rather than a single-purpose demo.
