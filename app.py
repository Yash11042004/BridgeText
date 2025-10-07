# app.py (Flask WhatsApp bot — updated to support voice-note STT while preserving existing behavior)
import os
import tempfile
import subprocess
import requests
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
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")  # optional, used for fetching Twilio-hosted media if needed
TWILIO_VALIDATE = os.getenv("TWILIO_VALIDATE", "true").lower() == "true"
DEBUG_SAVE_MEDIA = os.getenv("DEBUG_SAVE_MEDIA", "false").lower() == "true"

# --- OpenAI client (for STT) ---
import openai
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

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

# --- Helpers for audio download, conversion, transcription ---
def download_media(url: str, dest_path: str, auth: tuple = None, timeout: int = 30) -> None:
    """
    Download media to dest_path.
    If auth is provided (username, password), requests will use HTTP Basic Auth.
    """
    with requests.get(url, auth=auth, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 10):
                if chunk:
                    f.write(chunk)

def convert_to_mp3(input_path: str, output_path: str) -> None:
    """
    Convert any input audio to an mp3 with settings friendly for STT (16k mono).
    Requires ffmpeg installed and available on PATH.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "128k",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_with_openai(audio_file_path: str) -> str:
    """
    Use OpenAI Whisper (openai.Audio.transcribe) to transcribe audio.
    Returns transcript string or empty string on failure.
    """
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set; cannot transcribe.")
        return ""
    try:
        with open(audio_file_path, "rb") as fh:
            resp = openai.Audio.transcribe("whisper-1", fh)
        # Try to extract text robustly
        if isinstance(resp, dict):
            return resp.get("text", "") or resp.get("transcript", "") or ""
        # Some client versions return an object with .text
        return getattr(resp, "text", "") or ""
    except Exception as e:
        print("OpenAI transcription error:", e)
        return ""

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

    from_number = request.form.get("From", "anonymous")  # e.g. "whatsapp:+16084716735"
    incoming_msg = (request.form.get("Body") or "").strip()
    num_media = int(request.form.get("NumMedia", "0"))

    user_input = None

    # If there's media, attempt to handle audio (voice note)
    if num_media and num_media > 0:
        media_url = request.form.get("MediaUrl0")
        media_content_type = request.form.get("MediaContentType0", "").lower()
        print(f"Incoming media from {from_number}: url={media_url} content_type={media_content_type}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_path = os.path.join(tmpdir, "incoming_media")
                # Determine auth for Twilio-hosted media: Twilio requires basic auth with Account SID + Auth Token
                auth = None
                if TWILIO_ACCOUNT_SID and TWILIO_AUTH:
                    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH)

                # Download media
                download_media(media_url, raw_path, auth=auth)
                if DEBUG_SAVE_MEDIA:
                    debug_dir = os.path.join(os.getcwd(), "debug_media")
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_raw = os.path.join(debug_dir, f"{from_number.replace(':','_')}_raw")
                    with open(raw_path, "rb") as rfh, open(debug_raw, "wb") as wfh:
                        wfh.write(rfh.read())
                    print("Saved raw media to debug_media")

                # First attempt: try direct transcription (some formats may work)
                transcription = transcribe_with_openai(raw_path)

                # If direct transcription failed, try convert to mp3 and transcribe again
                if not transcription:
                    mp3_path = os.path.join(tmpdir, "converted.mp3")
                    try:
                        convert_to_mp3(raw_path, mp3_path)
                        transcription = transcribe_with_openai(mp3_path)
                        if DEBUG_SAVE_MEDIA and transcription:
                            debug_mp3 = os.path.join(os.getcwd(), "debug_media", f"{from_number.replace(':','_')}_converted.mp3")
                            with open(mp3_path, "rb") as rfh, open(debug_mp3, "wb") as wfh:
                                wfh.write(rfh.read())
                            print("Saved converted mp3 to debug_media")
                    except subprocess.CalledProcessError as ce:
                        print("ffmpeg conversion failed:", ce)
                        # try wav fallback
                        try:
                            wav_path = os.path.join(tmpdir, "converted.wav")
                            cmd = ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path]
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            transcription = transcribe_with_openai(wav_path)
                            if DEBUG_SAVE_MEDIA and transcription:
                                debug_wav = os.path.join(os.getcwd(), "debug_media", f"{from_number.replace(':','_')}_converted.wav")
                                with open(wav_path, "rb") as rfh, open(debug_wav, "wb") as wfh:
                                    wfh.write(rfh.read())
                                print("Saved converted wav to debug_media")
                        except Exception as e:
                            print("Fallback conversion/transcription failed:", e)
                            transcription = None
                    except Exception as e:
                        print("Conversion error:", e)
                        transcription = None

            # Decide what to pass to the pipeline
            if transcription and transcription.strip():
                user_input = transcription.strip()
                print(f"Transcription successful: {user_input}")
            else:
                # If transcription not available, optionally ask user to re-send text or pass a placeholder
                user_input = "[voice message received but could not transcribe]"
                print("Transcription empty or failed; using placeholder input")
        except Exception as e:
            print("Error processing incoming media:", e)
            user_input = "[error processing voice message]"
    else:
        user_input = incoming_msg

    # If still no input (empty text and no media), keep previous behavior
    if not user_input:
        resp = MessagingResponse()
        resp.message("👋 I didn’t receive any text. Please send a message.")
        return str(resp), 200

    # Pass user_input to existing pipeline and respond
    reply = generate_reply_for_input(from_number, user_input)
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200

@app.route("/whatsapp-status", methods=["POST"])
def whatsapp_status():
    # Delivery/read callbacks (optional)
    return ("", 204)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    # Bind to 0.0.0.0 for Render / external access
    app.run(host="0.0.0.0", port=port, debug=False)
