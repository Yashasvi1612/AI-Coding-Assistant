"""
agent.py
--------
The agentic core: given a user message, the LLM decides which tool(s) from
tools.TOOL_SCHEMAS to call (it may call none, one, or several), we execute
them, feed results back, and let the model produce a final answer.
This is what makes this an "agentic framework" rather than a single
fixed RAG prompt.
"""

import json
from groq import Groq
from tools import (
    TOOL_SCHEMAS,
    explain_code,
    fix_bug,
    generate_tests,
    generate_docs,
    review_pr_diff,
    chat_with_repo,
)
from indexer import retrieve_context

LLM_MODEL = "llama-3.1-8b-instant"

TOOL_FUNCTIONS = {
    "explain_code": explain_code,
    "fix_bug": fix_bug,
    "generate_tests": generate_tests,
    "generate_docs": generate_docs,
    "review_pr_diff": review_pr_diff,
}

AGENT_SYSTEM_PROMPT = (
    "You are an AI coding assistant, like a mini Copilot. You have access to "
    "tools for explaining code, fixing bugs, generating tests, generating "
    "docs, reviewing PR diffs, and chatting with an indexed GitHub repo. "
    "Decide which tool(s) best satisfy the user's request and call them. "
    "If the user pasted code directly, use the tool on that code rather than "
    "chat_with_repo. If they're asking about 'the repo' / a file / a function "
    "without pasting code, use chat_with_repo. You may call more than one "
    "tool if the request needs it (e.g. fix a bug AND generate a test for the fix)."
)


def run_agent(client: Groq, user_message: str, chat_history: list, index=None, chunks=None):
    """
    Runs one turn of the agent loop.
    chat_history: list of {"role": ..., "content": ...} dicts (prior turns).
    Returns (final_answer_text, tool_trace) where tool_trace is a list of
    strings describing which tools were called (for UI transparency).
    """
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    tool_trace = []

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0.2,
    )
    msg = response.choices[0].message

    # No tool calls -> model answered directly (e.g. small talk / clarification)
    if not msg.tool_calls:
        return msg.content or "(no response)", tool_trace

    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

    tool_results = []  # (name, result) pairs, in call order

    for tool_call in msg.tool_calls:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        if name == "chat_with_repo":
            question = args.get("question", user_message)
            retrieved = retrieve_context(question, index, chunks) if index is not None else []
            if not retrieved:
                result = "No repository is indexed yet, so I can't answer questions about it."
            else:
                result = chat_with_repo(client, question, retrieved)
            tool_trace.append(f"🔧 chat_with_repo(\"{question[:60]}\")")
        elif name in TOOL_FUNCTIONS:
            fn = TOOL_FUNCTIONS[name]
            result = fn(client, **args)
            tool_trace.append(f"🔧 {name}({', '.join(args.keys())})")
        else:
            result = f"Unknown tool: {name}"

        tool_results.append((name, result))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result,
            }
        )

    # Single tool call: return its output directly. A second LLM pass here
    # tends to summarize away the actual content (e.g. code/tests/explanations)
    # instead of relaying it, so we skip it when there's nothing to combine.
    if len(tool_results) == 1:
        return tool_results[0][1], tool_trace

    # Multiple tool calls: present each tool's raw output under its own
    # heading. (We intentionally do NOT ask the LLM to "combine" these with
    # another call — smaller models tend to leak pseudo function-call syntax
    # or paraphrase away real code/tests when asked to re-synthesize.)
    sections = []
    for name, result in tool_results:
        pretty_name = name.replace("_", " ").title()
        sections.append(f"### {pretty_name}\n\n{result}")
    return "\n\n---\n\n".join(sections), tool_trace