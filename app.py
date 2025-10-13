# app.py (updated + improved media handling, logging, and fallbacks)
import os
import tempfile
import subprocess
import requests
import traceback
import logging
from flask import Flask, request, jsonify
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
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_VALIDATE = os.getenv("TWILIO_VALIDATE", "true").lower() == "true"
DEBUG_SAVE_MEDIA = os.getenv("DEBUG_SAVE_MEDIA", "false").lower() == "true"

# Meta WhatsApp cloud env
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "stepbot_verify")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")

# logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("whatsapp-bot")

# --- OpenAI client (v1+) ---
openai_client = None
try:
    from openai import OpenAI

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = OpenAI()
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.exception("Could not initialize OpenAI v1 client: %s", e)
    openai_client = None

# --- RAG pipeline setup (langchain + vectorstore) ---
try:
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
    logger.info("Vector store loaded")
except Exception as e:
    logger.exception("Error initializing vector store or langchain modules: %s", e)
    # allow rest of app to run even if RAG setup failed
    embeddings = None
    db = None
    db_retriever = None

# Prompt template
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
# If you prefer to keep the full, exact prompt, replace the placeholder above with your original prompt string
from langchain.prompts import PromptTemplate as LCPromptTemplate
prompt = LCPromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# LLM + chains (guarded)
try:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")
    contextualize_q_system_prompt = "Given the conversation so far and a follow-up question, rephrase the follow-up question to be a standalone question."
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
    logger.info("LLM and retrieval chain initialized")
except Exception as e:
    logger.exception("Error initializing LLM or chains: %s", e)
    llm = None
    qa = None

# Conversation memory and tone preferences
conversation_memory = {}
tone_preferences = {}  # maps user_id -> "professional" | "casual"

def generate_reply_for_input(user_id: str, user_input: str) -> str:
    tone = tone_preferences.get(user_id)
    effective_input = user_input
    if tone:
        effective_input = f"[TONE: {tone}]\n{user_input}"
    chat_history_for_chain = conversation_memory.get(user_id, [])
    try:
        if qa:
            result = qa.invoke({"input": effective_input, "chat_history": chat_history_for_chain})
        else:
            result = {"answer": "Sorry, the assistant is temporarily unavailable."}
    except Exception as e:
        logger.exception("Error invoking QA chain: %s", e)
        result = {"answer": "Sorry, something went wrong while processing your request."}

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

# --- Helpers for audio download, conversion, transcription ---
def download_media(url: str, dest_path: str, auth: tuple = None, timeout: int = 30) -> None:
    logger.debug("Downloading media from %s to %s (auth=%s)", url, dest_path, bool(auth))
    resp = requests.get(url, auth=auth, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(10240):
            if chunk:
                f.write(chunk)
    logger.debug("Saved media to %s (%d bytes)", dest_path, os.path.getsize(dest_path))

def convert_to_mp3(input_path: str, output_path: str) -> None:
    logger.debug("Converting %s -> %s using ffmpeg", input_path, output_path)
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-b:a", "128k", output_path]
    subprocess.run(cmd, check=True)
    logger.debug("Conversion complete: %s", output_path)

def transcribe_with_openai(audio_file_path: str) -> str:
    if not openai_client:
        logger.warning("OpenAI client not available for transcription")
        return ""
    try:
        with open(audio_file_path, "rb") as fh:
            resp = openai_client.audio.transcriptions.create(model="gpt-4o-transcribe", file=fh)
        text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
        if text:
            logger.debug("Transcription (gpt-4o-transcribe) succeeded")
            return text
    except Exception as e:
        logger.debug("gpt-4o-transcribe failed or unavailable: %s", e)

    try:
        with open(audio_file_path, "rb") as fh:
            resp = openai_client.audio.transcriptions.create(model="whisper-1", file=fh)
        text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
        if text:
            logger.debug("Transcription (whisper-1) succeeded")
        return text or ""
    except Exception as e:
        logger.exception("Whisper transcription failed: %s", e)
        return ""

# --- Meta WhatsApp helpers ---
def send_whatsapp_reaction(to_number: str, message_id: str, emoji: str, phone_number_id: str, access_token: str):
    url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "reaction",
        "reaction": {"message_id": message_id, "emoji": emoji},
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    logger.debug("Sent reaction %s to %s for message %s", emoji, to_number, message_id)

def should_react_with_heart(user_input: str) -> bool:
    triggers = ["hi", "hello", "hey", "my name is"]
    return any(trigger in user_input.lower() for trigger in triggers)

def send_meta_text(to_number: str, text: str):
    url = f"https://graph.facebook.com/v17.0/{META_PHONE_NUMBER_ID}/messages"
    payload = {"messaging_product": "whatsapp", "to": to_number, "text": {"body": text}}
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        logger.exception("Error sending meta text: %s -- response: %s", e, resp.text if resp is not None else None)
    return resp

def send_meta_interactive_tone_choice(to_number: str):
    url = f"https://graph.facebook.com/v17.0/{META_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": "Would you like replies in a Professional or Casual tone?"},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "tone_professional", "title": "Professional"}},
                    {"type": "reply", "reply": {"id": "tone_casual", "title": "Casual"}},
                ]
            },
        },
    }
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        logger.exception("Error sending meta interactive: %s -- response: %s", e, resp.text if resp is not None else None)
    return resp

# --- Flask app ---
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)
validator = RequestValidator(TWILIO_AUTH) if TWILIO_AUTH else None

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200

# --- Twilio Webhook ---
@app.route("/whatsapp-webhook", methods=["POST"])
def whatsapp_webhook():
    try:
        if validator and TWILIO_VALIDATE:
            signature = request.headers.get("X-Twilio-Signature", "")
            if not validator.validate(request.url, request.form, signature):
                logger.warning("Invalid Twilio signature")
                return ("Invalid signature", 403)

        from_number = request.form.get("From", "anonymous")
        incoming_msg = (request.form.get("Body") or "").strip()
        num_media = int(request.form.get("NumMedia", "0"))

        user_input = None
        if num_media > 0:
            media_url = request.form.get("MediaUrl0")
            media_ct = request.form.get("MediaContentType0", "")  # e.g. audio/ogg, audio/mpeg
            # map common content-types to extensions
            ext = ".bin"
            if "ogg" in media_ct or "opus" in media_ct:
                ext = ".ogg"
            elif "mpeg" in media_ct or "mp3" in media_ct:
                ext = ".mp3"
            elif "wav" in media_ct:
                ext = ".wav"
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    raw_path = os.path.join(tmpdir, f"incoming_media{ext}")
                    # Twilio media URLs often require HTTP Basic auth with account SID and auth token
                    auth_tuple = None
                    if TWILIO_ACCOUNT_SID and TWILIO_AUTH:
                        auth_tuple = (TWILIO_ACCOUNT_SID, TWILIO_AUTH)
                    download_media(media_url, raw_path, auth=auth_tuple)
                    transcription = transcribe_with_openai(raw_path)
                    if not transcription:
                        mp3_path = os.path.join(tmpdir, "converted.mp3")
                        try:
                            convert_to_mp3(raw_path, mp3_path)
                            transcription = transcribe_with_openai(mp3_path)
                        except subprocess.CalledProcessError as cpe:
                            logger.exception("ffmpeg conversion failed: %s", cpe)
                    user_input = transcription or "[voice message received but could not transcribe]"
                    if DEBUG_SAVE_MEDIA:
                        save_dest = os.path.join(os.getcwd(), f"debug_media_{os.path.basename(raw_path)}")
                        with open(raw_path, "rb") as rfh, open(save_dest, "wb") as wfh:
                            wfh.write(rfh.read())
                        logger.info("Saved debug media to %s", save_dest)
            except Exception as e:
                logger.exception("Error processing incoming media: %s", e)
                user_input = "[error processing voice message]"
        else:
            user_input = incoming_msg

        if not user_input:
            resp = MessagingResponse()
            resp.message("👋 I didn’t receive any text. Please send a message.")
            return str(resp), 200

        reply = generate_reply_for_input(from_number, user_input)
        resp = MessagingResponse()
        resp.message(reply)
        return str(resp), 200
    except Exception as e:
        logger.exception("Unhandled error in whatsapp_webhook: %s", e)
        resp = MessagingResponse()
        resp.message("Sorry, something went wrong processing your message.")
        return str(resp), 500

@app.route("/whatsapp-status", methods=["POST"])
def whatsapp_status():
    return ("", 204)

# --- Meta Webhook ---
@app.route("/meta-webhook", methods=["GET", "POST"])
def meta_webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        challenge = request.args.get("hub.challenge")
        verify_token = request.args.get("hub.verify_token")
        if mode == "subscribe" and verify_token == META_VERIFY_TOKEN:
            logger.info("META webhook verified!")
            return str(challenge), 200
        return "Verification token mismatch", 403

    try:
        data = request.get_json(silent=True)
        logger.debug("Incoming Meta webhook: %s", data)
        if not data:
            return jsonify({"status": "no data"}), 200

        if data.get("object") == "whatsapp_business_account":
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    messages = value.get("messages", []) or []
                    for msg in messages:
                        from_number = msg.get("from") or "anonymous"
                        message_id = msg.get("id")
                        user_input = None

                        # text messages
                        if msg.get("type") == "text":
                            user_input = msg["text"]["body"].strip()
                            if user_input.lower() in ("hi", "hello", "hey"):
                                try:
                                    if message_id:
                                        send_whatsapp_reaction(
                                            from_number, message_id, "❤️", META_PHONE_NUMBER_ID, META_ACCESS_TOKEN
                                        )
                                    else:
                                        send_meta_text(from_number, "❤️")
                                except Exception:
                                    logger.exception("Error sending reaction on greeting")
                                try:
                                    send_meta_interactive_tone_choice(from_number)
                                except Exception:
                                    logger.exception("Error sending interactive tone choice")
                                continue

                        # audio/voice handling
                        elif msg.get("type") in ("audio", "voice"):
                            # Meta Graph: fetch media object to get the URL
                            try:
                                media_obj = msg.get(msg["type"], {})  # contains id
                                media_id = media_obj.get("id")
                                if media_id:
                                    media_url_fetch = f"https://graph.facebook.com/v17.0/{media_id}"
                                    params = {"access_token": META_ACCESS_TOKEN, "fields": "url"}
                                    media_resp = requests.get(media_url_fetch, params=params, timeout=15)
                                    media_resp.raise_for_status()
                                    media_json = media_resp.json()
                                    media_link = media_json.get("url") or media_json.get("secure_url")
                                    if not media_link:
                                        # older Graph responses sometimes return a 'data' object with 'url'
                                        media_link = media_json.get("data", {}).get("url")
                                    if media_link:
                                        with tempfile.TemporaryDirectory() as tmpdir:
                                            raw_path = os.path.join(tmpdir, "voice_input")
                                            # determine extension from content-type header if possible
                                            # fetch the media binary
                                            download_media(media_link, raw_path)
                                            # try transcription; convert if needed
                                            transcription = transcribe_with_openai(raw_path)
                                            if not transcription:
                                                mp3_path = os.path.join(tmpdir, "voice.mp3")
                                                try:
                                                    convert_to_mp3(raw_path, mp3_path)
                                                    transcription = transcribe_with_openai(mp3_path)
                                                except subprocess.CalledProcessError as cpe:
                                                    logger.exception("ffmpeg conversion failed for meta media: %s", cpe)
                                            user_input = transcription or "[voice message received but could not transcribe]"
                                    else:
                                        logger.warning("No media link returned for media id %s", media_id)
                                        user_input = "[voice message received but could not fetch media]"
                                else:
                                    logger.warning("No media id found in message object: %s", msg)
                                    user_input = "[voice message received but media id missing]"
                            except Exception as e:
                                logger.exception("Error fetching/transcribing meta audio: %s", e)
                                user_input = "[voice message received but could not transcribe]"

                        # interactive payload (button replies)
                        interactive = msg.get("interactive")
                        if interactive:
                            i_type = interactive.get("type")
                            if i_type == "button_reply":
                                br = interactive.get("button_reply", {})
                                button_id = br.get("id", "")
                                button_title = br.get("title", "").lower()
                                chosen_tone = None
                                if "professional" in button_id or "professional" in button_title:
                                    chosen_tone = "professional"
                                elif "casual" in button_id or "casual" in button_title:
                                    chosen_tone = "casual"
                                if chosen_tone:
                                    tone_preferences[from_number] = chosen_tone
                                    try:
                                        send_meta_text(
                                            from_number,
                                            f"Got it — I'll reply in a {chosen_tone.capitalize()} tone. How can I help today?",
                                        )
                                    except Exception:
                                        logger.exception("Error sending tone acknowledgement")
                                    continue

                        if not user_input:
                            # fallback skip (locations, stickers etc.)
                            user_input = None

                        if user_input:
                            try:
                                if should_react_with_heart(user_input):
                                    if message_id:
                                        send_whatsapp_reaction(
                                            from_number, message_id, "❤️", META_PHONE_NUMBER_ID, META_ACCESS_TOKEN
                                        )
                            except Exception:
                                logger.exception("Error sending reaction for user_input")

                            reply_text = generate_reply_for_input(from_number, user_input)
                            send_url = f"https://graph.facebook.com/v17.0/{META_PHONE_NUMBER_ID}/messages"
                            payload = {
                                "messaging_product": "whatsapp",
                                "to": from_number,
                                "text": {"body": reply_text},
                            }
                            headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
                            try:
                                resp = requests.post(send_url, json=payload, headers=headers)
                                logger.debug("Meta reply sent: %s -- %s", resp.status_code, resp.text)
                            except Exception:
                                logger.exception("Error sending Meta reply")

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.exception("Error in Meta webhook: %s", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
