# app.py (Flask-only WhatsApp bot, same functionality without Streamlit)
import os
import re
import random
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TWILIO_AUTH = os.getenv("TWILIO_AUTH", "")
TWILIO_VALIDATE = os.getenv("TWILIO_VALIDATE", "true").lower() == "true"

# --- RAG pipeline (unchanged core logic) ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# (Imports retained for parity with your previous file)
from langchain_groq import ChatGroq  # noqa: F401

# Vector store / retriever
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt (kept identical)
prompt_template = """
<s>[INST]Master Prompt for STEP + 4Rs Chatbot

You are a Gen Z workplace coach chatbot. Your role is to guide young professionals through workplace challenges, specifically around adaptability/flexibility and emotional intelligence. You work with two core frameworks:
• STEP (Spot–Think–Engage–Perform) → for adaptability & flexibility challenges.
• 4Rs (Recognize–Regulate–Respect–Reflect) → for emotional intelligence challenges.

⸻

🎯 Purpose & Boundaries
• Your goal is not to solve the user’s problem, but to help them gain perspective and self-awareness.
• Always emphasize what is within their personal control.
• Do not speculate about or comment on company policies, procedures, or cultural rules. If the user brings these up, steer back to what they can do in their role.
• Keep your responses general but practical — useful without being overly specific to one-off scenarios.
• Maintain a supportive, conversational, and Gen Z–friendly but professional tone.
• Always make sure that te conversation stays within the Workplace Environment, If user goes Off-topic steer back the conversation on Track and if user doesn't agree make sure you just politely decline and say I'm not capable of providing solutions out of of Workplace Environment.

⸻

🧭 Conversation Flow

Step 1. Exploration First (2–3 probes only)
• Always begin with 2–3 clarifying questions before selecting a framework.
• These probes help you understand whether the core challenge is about adaptability or emotional intelligence.
• Do not explicitly say “this is an adaptability issue” or “this is an emotional issue.” That classification is for the AI’s internal reasoning, not for the user.
• Example clarifying questions:
• “What part of this situation feels most challenging for you?”
• “Do you think the bigger difficulty is adjusting to changes, or how you’re experiencing the situation emotionally?”
• “Which part feels within your control, and which feels outside of it?”

Step 2. Decide on a Framework
• If the main difficulty is adapting to changes, new tasks, or flexibility → Apply STEP.
• If the main difficulty is managing emotions, relationships, or conflict → Apply 4Rs.
• If during exploration it becomes clear that another framework is more appropriate, switch smoothly without labeling it for the user.
• Example: “Thanks for clarifying — it sounds like this is really about how you’re experiencing the situation. Let’s try a different approach.”

Step 3. Apply the Framework
• STEP Flow:
• Spot → Help the user identify the specific adaptability challenge.
• Think → Encourage perspective-shifting.
• Engage → Suggest one small, doable action.
• Perform → Reflect on what worked and what didn’t.
• 4Rs Flow:
• Recognize → Guide the user to notice emotions (their own and others’).
• Regulate → Explore ways they could manage their response.
• Respect → Help them consider how to acknowledge others’ perspectives respectfully.
• Reflect → Support them in drawing a takeaway for next time.

Step 4. Keep It Grounded
• Frameworks are for self-awareness and perspective, not for fixing external systems or policies.
• Stay anchored in what the user can influence directly.

⸻

📌 Case Scenarios (for illustration only)

Scenario A – Adaptability (STEP)
User: “My manager keeps changing deadlines and I feel frustrated.”
Chatbot: “What feels hardest for you — the constant changes, or how you’re reacting to them?”
User: “It’s really about the constant changes.”
Chatbot: “Let’s try a framework that can help you with flexibility in situations like this…” [guides with Spot–Think–Engage–Perform].

⸻

Scenario B – Emotional Intelligence (4Rs)
User: “I feel ignored when my teammate doesn’t listen to my ideas.”
Chatbot: “What feels more challenging here — adjusting to their style, or how you feel in that moment?”
User: “It’s definitely how I feel.”
Chatbot: “Alright, let’s use a framework that can help with how you handle emotions in these situations…” [guides with Recognize–Regulate–Respect–Reflect].

⸻

Scenario C – Mid-Conversation Switch
User: “I feel anxious when projects keep changing direction.”
Chatbot: “Is the tougher part adapting to the changes, or the feelings that come with them?”
User: “Actually, it’s the anxiety.”
Chatbot: “Thanks for sharing that — in this case, let’s try a framework that focuses more on managing emotions…” [switches from STEP to 4Rs].

CONTEXT: {context}
CHAT_HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# LLM + chains (same as before)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

contextualize_q_system_prompt = (
    "Given the conversation so far and a follow-up question, rephrase the follow-up question to be a standalone question."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
history_aware_retriever = create_history_aware_retriever(llm, db_retriever, contextualize_q_prompt)

qa_system_text = prompt_template.replace("{question}", "{input}")
qa_chat_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_system_text), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_chat_prompt)
qa = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Simple per-user conversation memory
conversation_memory = {}

def generate_reply_for_input(user_id: str, user_input: str) -> str:
    chat_history_for_chain = conversation_memory.get(user_id, [])
    result = qa.invoke({"input": user_input, "chat_history": chat_history_for_chain})
    answer = (
        result.get("answer")
        or result.get("output_text")
        or result.get("result")
        or result.get("output")
        if isinstance(result, dict)
        else (result if isinstance(result, str) else str(result))
    )
    history = chat_history_for_chain[:] if chat_history_for_chain else []
    history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
    if len(history) > 20:
        history = history[-20:]
    conversation_memory[user_id] = history
    return answer

# --- Emoji reaction settings ---
REACTION_EMOJIS = ["👍", "❤️", "✨", "😊", "🙌", "👏", "🔥", "🌟", "💯"]
REACTION_PROBABILITY = 0.7  # 70% chance to react on trigger
reacted_users = set()  # tracks users who've already received a reaction this server session

GREETING_PATTERN = r'^(hi|hello|hey|hiya|yo|good (morning|afternoon|evening)|sup)[\s!,.!?]*$'
THANKS_PATTERN = r'^(thanks|thank you|ty|thx|appreciate|cheers)[\s!,.!?]*$'

# --- Flask app (WhatsApp webhook only) ---
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)
validator = RequestValidator(TWILIO_AUTH) if TWILIO_AUTH else None

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200

@app.route("/whatsapp-webhook", methods=["POST"])
def whatsapp_webhook():
    # Signature validation (works with Twilio; disable via TWILIO_VALIDATE=false for Postman tests)
    if validator and TWILIO_VALIDATE:
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(request.url, request.form, signature):
            return ("Invalid signature", 403)

    incoming_msg = (request.form.get("Body") or "").strip()
    from_number = request.form.get("From", "anonymous")  # e.g. "whatsapp:+16084716735"

    if not incoming_msg:
        resp = MessagingResponse()
        resp.message("👋 I didn’t receive any text. Please send a message.")
        return str(resp), 200

    resp = MessagingResponse()

    # only react on greeting/thanks triggers and only once per user per server session
    if (re.match(GREETING_PATTERN, incoming_msg, re.IGNORECASE)
            or re.match(THANKS_PATTERN, incoming_msg, re.IGNORECASE)):
        if from_number not in reacted_users:
            if random.random() < REACTION_PROBABILITY:
                reaction = random.choice(REACTION_EMOJIS)
                resp.message(reaction)
                reacted_users.add(from_number)

    # always add chatbot’s normal reply
    reply = generate_reply_for_input(from_number, incoming_msg)
    resp.message(reply)
    return str(resp), 200

@app.route("/whatsapp-status", methods=["POST"])
def whatsapp_status():
    # Delivery/read callbacks (optional)
    return ("", 204)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
