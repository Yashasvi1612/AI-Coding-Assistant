"""
tools.py
--------
Defines the individual "skills" the coding assistant can perform.
Each tool is a focused LLM call with its own prompt. The agent (agent.py)
decides WHICH of these to call based on the user's request, and can call
several tools before producing a final answer.
"""

from groq import Groq

LLM_MODEL = "llama-3.1-8b-instant"


def _call_llm(client: Groq, system_prompt: str, user_content: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def explain_code(client: Groq, code: str) -> str:
    return _call_llm(
        client,
        "You are a senior engineer. Explain the given code clearly: what it "
        "does, how it works step by step, and call out any non-obvious parts. "
        "Use plain language, minimal jargon.",
        code,
    )


def fix_bug(client: Groq, code: str, error_description: str = "") -> str:
    user_content = f"Code:\n{code}"
    if error_description:
        user_content += f"\n\nReported issue / error:\n{error_description}"
    return _call_llm(
        client,
        "You are a senior debugger. Find the bug(s) in the given code. "
        "Explain the root cause, then provide the corrected code in full. "
        "If you're not certain of the bug without more context, say what "
        "additional info you'd need.",
        user_content,
    )


def generate_tests(client: Groq, code: str, framework: str = "pytest") -> str:
    return _call_llm(
        client,
        f"You are a test engineer. Write {framework} unit tests for the given "
        "code. Cover normal cases, edge cases, and at least one failure case. "
        "Return runnable test code only, with brief comments.",
        code,
    )


def generate_docs(client: Groq, code: str) -> str:
    return _call_llm(
        client,
        "You are a technical writer. Generate documentation for the given "
        "code: a short description, parameters/return values (or props/args), "
        "and one usage example. Use clean Markdown.",
        code,
    )


def review_pr_diff(client: Groq, diff: str) -> str:
    return _call_llm(
        client,
        "You are a strict but constructive code reviewer. Review the given "
        "git diff. Point out bugs, style issues, missing tests, and security "
        "concerns. Organize as: Blocking Issues / Suggestions / Nitpicks. "
        "Be specific and reference line context from the diff.",
        diff,
    )


def chat_with_repo(client: Groq, question: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        f"# File: {c['path']}\n{c['text']}" for c in retrieved_chunks
    )
    return _call_llm(
        client,
        "You are a codebase assistant. Answer the user's question about this "
        "repository using ONLY the provided file excerpts as context. Cite "
        "which file(s) you used. If the context doesn't contain the answer, "
        "say so rather than guessing.",
        f"Repository context:\n{context}\n\nQuestion: {question}",
    )


# Tool schemas exposed to the LLM for function-calling / routing.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "explain_code",
            "description": "Explain what a piece of code does, step by step.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_bug",
            "description": "Find and fix a bug in a piece of code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "error_description": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_tests",
            "description": "Generate unit tests for a piece of code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "framework": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_docs",
            "description": "Generate documentation for a piece of code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_pr_diff",
            "description": "Review a git diff / pull request and give structured feedback.",
            "parameters": {
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_with_repo",
            "description": (
                "Answer a question about the currently indexed GitHub repository "
                "using retrieved code context. Use this whenever the user asks "
                "about 'the repo', 'this codebase', a file, or a function without "
                "pasting the code themselves."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]