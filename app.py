

from pathlib import Path
import gradio as gr

from rag.generator import generator
from utils.chat_storage import (
    create_conversation,
    update_conversation,
    get_conversation,
    list_conversations,
    delete_all_conversations,
)

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "ui" / "styles.css"

if not CSS_PATH.exists():
    raise FileNotFoundError(f"CSS file not found: {CSS_PATH}")

CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8")


APP_TITLE = "NutriBot"

SUGGESTED_QUESTIONS = [
    "What are the functions of proteins?",
    "Why are carbohydrates important?",
    "What are the functions of vitamins?",
    "Why is water important for the body?",
    "Explain the importance of dietary fats.",
    "What is a balanced diet?",
]


# ---------------------------------------
# HISTORY HELPERS
# ---------------------------------------

def get_history_choices():
    conversations = list_conversations()

    return [
        (
            f"{item['title']}  ·  {item['updated_at'][:10]}",
            item["id"]
        )
        for item in conversations
    ]


def display_messages(messages):
    """Remove storage-only metadata before displaying chat."""
    return [
        {
            "role": item["role"],
            "content": item["content"]
        }
        for item in messages
        if item.get("role") in ["user", "assistant"]
    ]


def format_sources(sources):
    """Format source page references from the Groq generator."""
    if not sources:
        return "No textbook references available."

    pages = []
    for source in sources:
        if isinstance(source, dict):
            page = source.get("page", "Unknown")
        else:
            page = source
        if page not in pages:
            pages.append(page)

    return (
        "### 📚 Textbook References\n"
        + ", ".join(f"Page {page}" for page in pages)
    )


def format_passages(passages):
    """Display the context returned by the Groq generator."""
    if not passages:
        return "No retrieved passages available."

    # The Groq generator returns the retrieved passages as one formatted string.
    if isinstance(passages, str):
        return "### Retrieved Textbook Context\n\n" + passages

    # Also support a list of passage dictionaries for compatibility.
    output = ""
    for index, item in enumerate(passages, start=1):
        if isinstance(item, dict):
            output += (
                f"### Passage {index}\n"
                f"**PDF Page:** {item.get('page', 'Unknown')}  \n"
                f"**Similarity:** {item.get('similarity', 'N/A')}\n\n"
                f"{item.get('text', '')}\n\n---\n\n"
            )
        else:
            output += f"### Passage {index}\n{item}\n\n---\n\n"
    return output


# ---------------------------------------
# CHAT FUNCTION
# ---------------------------------------

def chat_response(message, history, conversation_id):

    if not message or not message.strip():
        return (
            history or [],
            "",
            "",
            "",
            conversation_id,
            gr.update(choices=get_history_choices())
        )

    try:
        if not conversation_id:
            conversation_id = create_conversation()

        result = generator.generate_answer(message.strip())

        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        passages = result.get("context", [])

        conversation = get_conversation(conversation_id)

        saved_messages = (
            conversation["messages"]
            if conversation else []
        )

        saved_messages.append({
            "role": "user",
            "content": message.strip()
        })

        # Store references along with the assistant response.
        saved_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "context": passages
        })

        update_conversation(
            conversation_id,
            saved_messages
        )

        visible_history = display_messages(saved_messages)

        return (
            visible_history,
            "",
            format_sources(sources),
            format_passages(passages),
            conversation_id,
            gr.update(choices=get_history_choices())
        )

    except Exception as error:

        print("NutriBot error:", error)

        updated_history = list(history or [])

        updated_history.append({
            "role": "assistant",
            "content": (
                "Sorry, something went wrong while generating "
                "your answer. Please try again."
            )
        })

        return (
            updated_history,
            "",
            "Unable to retrieve references.",
            "",
            conversation_id,
            gr.update(choices=get_history_choices())
        )


# ---------------------------------------
# NEW CHAT
# ---------------------------------------

def new_chat():

    conversation_id = create_conversation()

    return (
        [],
        "",
        "",
        "",
        conversation_id,
        gr.update(choices=get_history_choices(), value=None)
    )


# ---------------------------------------
# LOAD PREVIOUS CONVERSATION
# ---------------------------------------

def load_previous_chat(conversation_id):

    if not conversation_id:
        return [], None, "", ""

    conversation = get_conversation(conversation_id)

    if not conversation:
        return [], None, "", ""

    messages = conversation.get("messages", [])

    # Restore references from the latest assistant response.
    latest_assistant = next(
        (
            item for item in reversed(messages)
            if item.get("role") == "assistant"
        ),
        {}
    )

    return (
        display_messages(messages),
        conversation_id,
        format_sources(latest_assistant.get("sources", [])),
        format_passages(latest_assistant.get("context", []))
    )


# ---------------------------------------
# DELETE ALL HISTORY
# ---------------------------------------

def delete_history():

    delete_all_conversations()

    conversation_id = create_conversation()

    return (
        [],
        "",
        "",
        "",
        conversation_id,
        gr.update(choices=[], value=None)
    )

# ---------------------------------------
# APPLICATION
# ---------------------------------------

initial_conversation_id = create_conversation()


with gr.Blocks(
    title="NutriBot | Human Nutrition AI",
    css=CUSTOM_CSS
) as demo:


    current_conversation = gr.State(
        initial_conversation_id
    )

    with gr.Row(equal_height=False):

        # SIDEBAR

        with gr.Column(
            scale=1,
            min_width=240,
            elem_id="sidebar"
        ):

            gr.Markdown(
                "# 🥗 NutriBot",
                elem_id="brand"
            )

            gr.Markdown(
                "YOUR HUMAN NUTRITION ASSISTANT",
                elem_id="tagline"
            )

            gr.Markdown("---")

            new_chat_button = gr.Button(
                "＋ New Chat",
                variant="primary",
                elem_classes=["primary-btn"]
            )

            with gr.Accordion(
                "🕘 Chat History",
                open=False
            ):

                history_selector = gr.Radio(
                    choices=get_history_choices(),
                    label="Previous Conversations",
                    interactive=True
                )

                delete_history_button = gr.Button(
                    "🗑 Delete All History"
                )

            gr.Markdown("---")

            gr.Markdown("### 🧠 Knowledge Base")

            gr.Markdown(
                """
                **Source:** Human Nutrition – 2020 Edition

                **Database:** 2,511 text vectors

                **Embedding:** MiniLM (384 dimensions)

                **Search:** FAISS

                **Generator:** Groq (openai/gpt-oss-120b)
                """
            )

            gr.Markdown("---")

            with gr.Accordion(
                "ℹ️ About NutriBot",
                open=False
            ):

                gr.Markdown(
                    """
                    NutriBot is an educational human
                    nutrition assistant powered by
                    Retrieval-Augmented Generation.

                    It retrieves relevant information
                    from the Human Nutrition textbook
                    and generates answers with references.
                    """
                )

            gr.Markdown(
                "🟢 Knowledge base loaded",
                elem_id="tagline"
            )

        # MAIN PANEL

        with gr.Column(
            scale=4,
            elem_id="main-panel"
        ):

            gr.Markdown("# Welcome to NutriBot 👋")

            gr.Markdown(
                "Explore nutrition, nutrients, food, digestion "
                "and the science of human health."
            )

            gr.Markdown("### 💬 Nutrition Assistant")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=470,
                type="messages",
                elem_id="chatbot",
                allow_tags=False
            )

            with gr.Row():

                message = gr.Textbox(
                    placeholder="Ask anything about human nutrition...",
                    label="Your Question",
                    lines=2,
                    scale=5,
                    elem_id="question-box"
                )

                send_button = gr.Button(
                    "➤ Send",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"]
                )

            # SUGGESTED QUESTIONS BELOW INPUT

            gr.Markdown("### ✨ Suggested Questions")

            suggestion_buttons = []

            with gr.Row():
                for question in SUGGESTED_QUESTIONS[:3]:

                    button = gr.Button(
                        question,
                        elem_classes=["suggestion-btn"]
                    )

                    suggestion_buttons.append(
                        (button, question)
                    )

            with gr.Row():
                for question in SUGGESTED_QUESTIONS[3:]:

                    button = gr.Button(
                        question,
                        elem_classes=["suggestion-btn"]
                    )

                    suggestion_buttons.append(
                        (button, question)
                    )

            # SOURCES

            with gr.Accordion(
                "📚 Textbook References and Retrieved Passages",
                open=False
            ):

                source_output = gr.Markdown(
                    elem_id="sources-panel"
                )

                passage_output = gr.Markdown()

            gr.Markdown(
                """
                ---
                **Educational Disclaimer:** NutriBot provides
                information for educational purposes only.
                It is not a substitute for professional
                medical or dietary advice.
                """,
                elem_id="footer"
            )

    # -----------------------------------
    # EVENT OUTPUTS
    # -----------------------------------

    chat_outputs = [
        chatbot,
        message,
        source_output,
        passage_output,
        current_conversation,
        history_selector
    ]

    # SEND BUTTON

    send_button.click(
        fn=chat_response,
        inputs=[
            message,
            chatbot,
            current_conversation
        ],
        outputs=chat_outputs
    )

    # ENTER KEY

    message.submit(
        fn=chat_response,
        inputs=[
            message,
            chatbot,
            current_conversation
        ],
        outputs=chat_outputs
    )

    # NEW CHAT

    new_chat_button.click(
        fn=new_chat,
        outputs=chat_outputs
    )

    # LOAD HISTORY

    history_selector.change(
        fn=load_previous_chat,
        inputs=[history_selector],
        outputs=[
            chatbot,
            current_conversation,
            source_output,
            passage_output
        ]
    )

    # DELETE HISTORY

    delete_history_button.click(
        fn=delete_history,
        outputs=chat_outputs
    )

    # SUGGESTED QUESTIONS

    for button, question in suggestion_buttons:

        button.click(
            fn=lambda history, cid, q=question:
                chat_response(q, history, cid),
            inputs=[
                chatbot,
                current_conversation
            ],
            outputs=chat_outputs
        )


# ---------------------------------------
# START SERVER
# ---------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
