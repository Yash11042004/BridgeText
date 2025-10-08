# app.py — Twilio WhatsApp + WhatsApp Cloud API chatbot (non-breaking update)
import os
import tempfile
import subprocess
import requests
import traceback
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TWILIO_AUTH = os.getenv("TWILIO_AUTH", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_VALIDATE = os.getenv("TWILIO_VALIDATE", "true").lower() == "true"
DEBUG_SAVE_MEDIA = os.getenv("DEBUG_SAVE_MEDIA", "false").lower() == "true"

# --- WhatsApp Cloud API config (NEW) ---
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "stepbot_verify")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "")

# --- OpenAI client ---
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- LangChain + pipeline imports (identical to yours, not modified) ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

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
• Maintain a supportive, conversational, and Gen Z–friendly but professional tone. Don't asks same questions repeatedly or in round-about manner and dont ask too may questions.
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

Critical Communication Rules
Keep It Short and Natural
Maximum 2 sentences per response (3 only if absolutely necessary)
Don't ask a question after every single sentence - sometimes just make a statement
Vary your response types: statements, questions, observations, suggestions
Sound like a real person texting, not a formal coach reading from a script
Bad Examples (Too Long, Too Many Questions):
❌ "That sounds really challenging and I can understand why you'd feel frustrated about that situation. Working in an environment where you don't feel supported can be incredibly draining on your mental health and overall wellbeing. How long have you been experiencing these feelings? What specific situations trigger the most stress for you?"
❌ "I hear you - dealing with a difficult manager can really impact your day-to-day work experience and make it hard to feel motivated. It's completely normal to feel this way when you're facing these kinds of interpersonal challenges. Have you noticed any patterns in when they behave this way? How do you typically respond when this happens?"
Good Examples (Concise, Natural):
✅ "That sounds exhausting. How long has this been going on?"
✅ "Yeah, that would stress anyone out. What part feels hardest for you?"
✅ "I get why you're frustrated. Sounds like your manager's style is really different from what you're used to."
✅ "That's a tough spot to be in. Would it help to work through a method for handling situations like this?"
Response Variety Patterns
Don't always end with questions. Mix it up:
Statement + Question
"That sounds really frustrating. What's been the hardest part?"
Just a Statement
"I can see why that would make you anxious about work."
Observation + Statement
"Sounds like you're dealing with a lot of uncertainty. That would stress anyone out."
Question Only (occasionally)
"How long has this been going on?"
Validation + Suggestion
"That's a tough situation. Would it help to try a structured approach to this?"
Make It Feel Human
Use Natural Language:
"That sucks" instead of "That sounds challenging"
"Yeah, that's tough" instead of "I can understand your frustration"
"Makes sense" instead of "That's completely understandable"
"I get it" instead of "I can see why you feel that way"
Drop Unnecessary Words:
❌ "I can definitely understand why that would be frustrating for you"
✅ "Yeah, that's frustrating"
Don't Over-Validate:
❌ "That's completely understandable and valid and anyone would feel exactly the same way"
✅ "Anyone would feel that way"
Framework Application - Stay Concise
When Introducing Frameworks:
❌ "This sounds like it's really about adapting to significant workplace changes and building your skills around flexibility and resilience in the face of organizational transformation. Would it be helpful if we used a systematic method that's specifically designed for navigating these kinds of adaptation challenges?"
✅ "This sounds like an adaptation challenge. Want to try a method designed for handling workplace changes?"
When Moving Through Phases:
❌ "You've already demonstrated excellent awareness by identifying what's changing in your environment, which is actually the first part of the framework called SPOT. Now we're going to move into the THINK phase where we work on reframing your perspective about this challenge."
✅ "You've done SPOT well - you see what's changing. Now let's work on THINK - reframing how you see this."
When Ending:
❌ "You've successfully worked through the complete STEP framework including all four phases: SPOT where you identified the changes, THINK where you reframed your perspective, ENGAGE where you identified concrete actions, and PERFORM where you planned how to measure and learn from your results."
✅ "You've worked through the full STEP approach. You can use this same method for any workplace changes."
Example Conversation (Concise & Natural)
User: "my boss keeps giving me work right when i'm about to leave"
AI: "That's super frustrating. Does this happen a lot?"
User: "yeah like every single day"
AI: "Sounds like a pattern. What's the main issue for you - the extra work or the timing?"
User: "the timing, i never know when my day actually ends"
AI: "That uncertainty would stress anyone out. I can't help change your boss's timing, but I can help you manage the stress this creates. Want to try that?"
User: "yeah"
AI: "It's called the 4Rs framework. You've already done RECOGNIZE - you know this pattern triggers stress. Now REGULATE - when they hand you work at 5pm, what could help you stay calm?"
User: "maybe just accept it's going to happen?"
AI: "Right, expecting it instead of being surprised. Now RESPECT - why might your boss do this?"
User: "probably they're disorganized or have stuff come up last minute"
AI: "Probably not personal to you. REFLECT - what boundary could you test, like 'I can start on this tomorrow morning'?"
User: "i could try that"
AI: "You've used the full 4Rs approach. Use it anytime you're managing difficult work situations."
Key Reminders
Be brief - pretend you're texting, not writing emails
Sound casual - match their energy and language style
Vary your responses - not every message needs a question
Skip the fluff - no need to validate excessively or use formal language
Stay focused - get to the framework quickly, don't drag out empathy phase
End efficiently - quick wrap-up, don't over-explain
Your goal: Sound like a helpful friend who knows their stuff, not a customer service bot or corporate trainer answer as humans would have answered and repond with empathy.

CONTEXT: {context}
CHAT_HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input", "chat_history"])
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [("system", "Rephrase follow-up question to standalone."), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
history_aware_retriever = create_history_aware_retriever(llm, db_retriever, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages(
    [("system", prompt_template), MessagesPlaceholder("chat_history"), ("human", "{input}")]
))
qa = create_retrieval_chain(history_aware_retriever, qa_chain)

conversation_memory = {}

def generate_reply_for_input(user_id: str, user_input: str) -> str:
    history = conversation_memory.get(user_id, [])
    result = qa.invoke({"input": user_input, "chat_history": history})
    answer = result.get("answer") or result.get("output_text") or str(result)
    history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
    conversation_memory[user_id] = history[-20:]
    return answer

# --- Audio utilities (unchanged) ---
def download_media(url: str, dest_path: str, auth=None, timeout=30):
    print("Downloading media:", url)
    resp = requests.get(url, auth=auth, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 10):
            f.write(chunk)

def convert_to_mp3(input_path: str, output_path: str):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-b:a", "128k", output_path], check=True)

def transcribe_with_openai(audio_file_path: str) -> str:
    try:
        with open(audio_file_path, "rb") as fh:
            resp = openai_client.audio.transcriptions.create(model="gpt-4o-transcribe", file=fh)
        return resp.text if hasattr(resp, "text") else resp.get("text", "")
    except Exception as e:
        print("Transcribe error:", e)
        traceback.print_exc()
        return ""

# --- Flask app setup ---
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
validator = RequestValidator(TWILIO_AUTH) if TWILIO_AUTH else None

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200

# ----------------------------------------------------------------
# ✅ TWILIO WhatsApp webhook (kept EXACTLY same behavior as before)
# ----------------------------------------------------------------
@app.route("/whatsapp-webhook", methods=["POST"])
def whatsapp_webhook():
    if validator and TWILIO_VALIDATE:
        signature = request.headers.get("X-Twilio-Signature", "")
        if not validator.validate(request.url, request.form, signature):
            return ("Invalid signature", 403)

    from_number = request.form.get("From", "anonymous")
    incoming_msg = (request.form.get("Body") or "").strip()
    num_media = int(request.form.get("NumMedia", "0"))

    print("Incoming Twilio message from:", from_number)
    user_input = incoming_msg

    # Handle audio (unchanged)
    if num_media and num_media > 0:
        media_url = request.form.get("MediaUrl0")
        media_content_type = request.form.get("MediaContentType0", "")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = os.path.join(tmpdir, "raw_audio")
                auth = None
                if media_url and "api.twilio.com" in media_url and TWILIO_ACCOUNT_SID and TWILIO_AUTH:
                    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH)
                download_media(media_url, raw_path, auth)
                mp3_path = os.path.join(tmpdir, "audio.mp3")
                convert_to_mp3(raw_path, mp3_path)
                transcription = transcribe_with_openai(mp3_path)
                user_input = transcription or "[voice note could not be transcribed]"
        except Exception as e:
            print("Error processing Twilio audio:", e)
            user_input = "[error processing voice message]"

    if not user_input:
        resp = MessagingResponse()
        resp.message("👋 I didn’t receive any text. Please send a message.")
        return str(resp), 200

    reply = generate_reply_for_input(from_number, user_input)
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200

# ----------------------------------------------------------------
# ✅ WhatsApp Cloud API webhook (NEW)
# ----------------------------------------------------------------
@app.route("/meta-webhook", methods=["GET"])
def verify_meta_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        print("✅ Meta webhook verified.")
        return challenge, 200
    return "Verification failed", 403

@app.route("/meta-webhook", methods=["POST"])
def handle_meta_webhook():
    data = request.get_json()
    print("Incoming Meta webhook:", data)

    try:
        if "entry" in data:
            for entry in data["entry"]:
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", [])
                    if not messages:
                        continue

                    for msg in messages:
                        from_number = msg.get("from")
                        msg_type = msg.get("type", "text")
                        user_input = None

                        if msg_type == "text":
                            user_input = msg["text"]["body"]
                        elif msg_type == "audio":
                            audio_id = msg["audio"]["id"]
                            media_url = f"https://graph.facebook.com/v19.0/{audio_id}"
                            headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
                            info = requests.get(media_url, headers=headers).json()
                            download_url = info.get("url")
                            if download_url:
                                audio_bytes = requests.get(download_url, headers=headers).content
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                                    tmp.write(audio_bytes)
                                    tmp_path = tmp.name
                                user_input = transcribe_with_openai(tmp_path)

                        if user_input:
                            reply = generate_reply_for_input(from_number, user_input)
                            send_meta_message(from_number, reply)
    except Exception as e:
        print("Error in Meta webhook:", e)
        traceback.print_exc()

    return jsonify({"status": "ok"}), 200

# ----------------------------------------------------------------
# ✅ WhatsApp Cloud API send message (NEW)
# ----------------------------------------------------------------
def send_meta_message(to_number: str, text: str):
    if not (META_ACCESS_TOKEN and META_PHONE_NUMBER_ID):
        print("⚠️ Missing META config, cannot send message.")
        return

    url = f"https://graph.facebook.com/v19.0/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text}
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        print("Meta send status:", res.status_code, res.text)
    except Exception as e:
        print("Error sending Meta message:", e)
        traceback.print_exc()

# ----------------------------------------------------------------
# Run Flask
# ----------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
