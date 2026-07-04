"""
app.py
------
Gradio UI for the AI Coding Assistant.

Tabs:
- Chat: talk to the agent (explain/fix/test/docs/review/chat-with-repo)
- Repo: index a public GitHub repo for RAG-based chat
- History: browse and switch between past chat sessions (persisted to disk)

Set GROQ_API_KEY as an environment variable (or HF Space secret).
"""

import os
import gradio as gr
from groq import Groq

from agent import run_agent
from indexer import build_repo_index
from history import (
    load_sessions,
    create_session,
    update_session_messages,
    delete_session,
    session_labels,
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

CUSTOM_CSS = """
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#title-banner {
    text-align: center;
    padding: 14px 0 8px 0;
    animation: fadeInDown 0.6s ease;
}
#title-banner h1 {
    margin-bottom: 2px;
    font-weight: 700;
    letter-spacing: -0.5px;
    font-size: 2rem;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#title-banner p {
    color: #64748b;
    font-size: 0.95rem;
}

/* Tab content fades in on switch */
.tabitem {
    animation: fadeIn 0.35s ease;
}

/* Tab labels: smooth underline/hover feel */
.tab-nav button {
    transition: color 0.15s ease, border-color 0.15s ease !important;
}

/* All buttons: gentle lift + glow on hover */
button {
    transition: transform 0.15s ease, box-shadow 0.15s ease, background-color 0.15s ease !important;
}
button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.25);
}
button:active {
    transform: translateY(0px) scale(0.98);
}

/* Text inputs: focus glow instead of a hard blue outline */
textarea:focus, input[type="text"]:focus {
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
    border-color: #6366f1 !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

/* Chat panel: rounded, soft shadow, fades in */
#main-chatbot {
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08);
    animation: fadeIn 0.4s ease;
}

/* History radio options: hover feedback */
#history-list label {
    transition: background-color 0.15s ease, transform 0.1s ease;
    border-radius: 8px !important;
}
#history-list label:hover {
    background-color: rgba(99, 102, 241, 0.08);
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size="lg",
)


# ---------- Repo indexing ----------

def index_repo(repo_url):
    if not repo_url or not repo_url.strip():
        return None, [], "Enter a public GitHub repo URL first."
    try:
        index, chunks, repo_dir, stats = build_repo_index(repo_url.strip())
        return index, chunks, stats
    except Exception as e:
        return None, [], f"Failed to index repo: {e}"


# ---------- Chat ----------

def chat(user_message, chatbot_messages, index_state, chunks_state, sessions_state, current_id):
    if not user_message.strip():
        return chatbot_messages, "", sessions_state

    chatbot_messages = chatbot_messages or []

    # Strip to role/content only -- Gradio message dicts can carry extra
    # keys (e.g. "metadata") that the Groq API rejects.
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in chatbot_messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    answer, tool_trace = run_agent(
        groq_client, user_message, clean_history, index=index_state, chunks=chunks_state
    )

    if tool_trace:
        answer = f"_Tools used: {', '.join(tool_trace)}_\n\n{answer}"

    chatbot_messages = chatbot_messages + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer},
    ]

    update_session_messages(sessions_state, current_id, chatbot_messages)
    return chatbot_messages, "", sessions_state


# ---------- History / sessions ----------

def start_new_chat(sessions_state):
    sid = create_session(sessions_state)
    labels, _ = session_labels(sessions_state)
    return [], sid, sessions_state, gr.update(choices=labels, value=labels[0] if labels else None)


def refresh_history_list(sessions_state):
    labels, _ = session_labels(sessions_state)
    return gr.update(choices=labels, value=labels[0] if labels else None)


def load_selected_session(label, sessions_state, current_id):
    labels, label_to_id = session_labels(sessions_state)
    sid = label_to_id.get(label)
    if sid is None:
        return gr.update(), current_id, gr.update()
    messages = sessions_state[sid]["messages"]
    return messages, sid, gr.Tabs(selected="chat_tab")


def delete_selected_session(label, sessions_state, current_id):
    labels, label_to_id = session_labels(sessions_state)
    sid = label_to_id.get(label)
    if sid:
        delete_session(sessions_state, sid)

    labels, label_to_id = session_labels(sessions_state)

    if not labels:
        new_id = create_session(sessions_state)
        labels, label_to_id = session_labels(sessions_state)
        return gr.update(choices=labels, value=labels[0]), sessions_state, [], new_id

    if sid == current_id:
        first_label = labels[0]
        new_current = label_to_id[first_label]
        return (
            gr.update(choices=labels, value=first_label),
            sessions_state,
            sessions_state[new_current]["messages"],
            new_current,
        )

    return gr.update(choices=labels), sessions_state, gr.update(), current_id


# ---------- Bootstrap initial state ----------

_initial_sessions = load_sessions()
if not _initial_sessions:
    _first_id = create_session(_initial_sessions)
else:
    _labels0, _label_to_id0 = session_labels(_initial_sessions)
    _first_id = _label_to_id0[_labels0[0]]

_labels0, _ = session_labels(_initial_sessions)


with gr.Blocks(title="AI Coding Assistant", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "<div id='title-banner'><h1>🤖 AI Coding Assistant</h1>"
        "<p>Explain code · Fix bugs · Generate tests · Write docs · "
        "Review PRs · Chat with a repo</p></div>"
    )

    sessions_state = gr.State(_initial_sessions)
    current_session_id = gr.State(_first_id)
    index_state = gr.State(None)
    chunks_state = gr.State([])

    with gr.Tabs() as tabs:
        with gr.Tab("💬 Chat", id="chat_tab"):
            new_chat_btn = gr.Button("🆕 New Chat", size="sm")
            chatbot = gr.Chatbot(
                label="Assistant",
                value=_initial_sessions[_first_id]["messages"],
                height=480,
                elem_id="main-chatbot",
            )
            msg = gr.Textbox(
                label="Message",
                placeholder="e.g. 'Explain this function: ...' or 'What does the parser module do?'",
            )
            send_btn = gr.Button("Send", variant="primary")

        with gr.Tab("📦 Repo", id="repo_tab"):
            gr.Markdown(
                "Index a public GitHub repo, then switch to **Chat** and ask "
                "questions about it (e.g. \"what does this repo do?\")."
            )
            repo_input = gr.Textbox(label="GitHub repo URL", placeholder="https://github.com/user/repo")
            index_btn = gr.Button("Index Repo", variant="primary")
            repo_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("🕘 History", id="history_tab"):
            gr.Markdown("Past chat sessions, saved locally to `sessions.json`.")
            history_list = gr.Radio(
                choices=_labels0,
                value=_labels0[0] if _labels0 else None,
                label="Sessions",
                elem_id="history-list",
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh")
                load_btn = gr.Button("📂 Load", variant="primary")
                delete_btn = gr.Button("🗑️ Delete", variant="stop")

    # ---------- Wiring ----------
    index_btn.click(index_repo, inputs=[repo_input], outputs=[index_state, chunks_state, repo_status])

    send_btn.click(
        chat,
        inputs=[msg, chatbot, index_state, chunks_state, sessions_state, current_session_id],
        outputs=[chatbot, msg, sessions_state],
    )
    msg.submit(
        chat,
        inputs=[msg, chatbot, index_state, chunks_state, sessions_state, current_session_id],
        outputs=[chatbot, msg, sessions_state],
    )

    new_chat_btn.click(
        start_new_chat,
        inputs=[sessions_state],
        outputs=[chatbot, current_session_id, sessions_state, history_list],
    )

    refresh_btn.click(refresh_history_list, inputs=[sessions_state], outputs=[history_list])

    load_btn.click(
        load_selected_session,
        inputs=[history_list, sessions_state, current_session_id],
        outputs=[chatbot, current_session_id, tabs],
    )

    delete_btn.click(
        delete_selected_session,
        inputs=[history_list, sessions_state, current_session_id],
        outputs=[history_list, sessions_state, chatbot, current_session_id],
    )

if __name__ == "__main__":
    demo.launch()