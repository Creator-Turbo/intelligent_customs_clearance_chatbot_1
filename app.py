from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect
from src.prompt import *
import os
import tempfile
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract

# ----------------------------- Flask App Setup -----------------------------
app = Flask(__name__)
load_dotenv()

# ----------------------------- API Keys -----------------------------
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not PINECONE_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("❌ Missing API keys. Check your .env file.")

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
print("✅ API keys loaded successfully!")

# ----------------------------- Embeddings & Pinecone -----------------------------
embeddings = download_hugging_face_embeddings()
index_name = "customs-clearance-chatbot"

docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------- LLM Setup -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=512,
    max_retries=3,
)

# ----------------------------- Prompt Template -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

# ----------------------------- Chat Memory -----------------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ----------------------------- RAG Chain -----------------------------
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# ----------------------------- Upload Folder -----------------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------------- Routes -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------- Chat API -----------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("🧍 User:", msg)

    # --- Detect and limit language support ---
    try:
        detected_lang = detect(msg)
    except:
        detected_lang = "en"

    supported_langs = {"en": "en", "hi": "hi", "ne": "ne","mai": "mai"}
    user_lang = supported_langs.get(detected_lang, "en")

    # Translate user message to English (only for Hindi/Nepali)
    if user_lang != "en":
        translated_input = GoogleTranslator(source=user_lang, target="en").translate(msg)
    else:
        translated_input = msg

    print("🔤 Translated Input:", translated_input)
    session_id = "default_user"

    # --- Generate response ---
    response = rag_chain_with_memory.invoke(
        {"input": translated_input},
        config={"configurable": {"session_id": session_id}}
    )

    english_answer = response["answer"]

    # Translate response back to user's language (only for Hindi/Nepali)
    if user_lang in ["hi", "ne", "mai"]:
        translated_answer = GoogleTranslator(source="en", target=user_lang).translate(english_answer)
    else:
        translated_answer = english_answer

    return str(translated_answer)

# ----------------------------- File Upload + Document Verification -----------------------------
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        file_path = temp_file.name

    ext = filename.rsplit('.', 1)[1].lower()
    extracted_text = ""

    try:
        # Extract text from supported document formats
        if ext == "pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted_text += page.extract_text() or ""

        elif ext == "docx":
            doc = Document(file_path)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"

        elif ext in ["jpg", "jpeg", "png"]:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)

        os.remove(file_path)  # Clean up temp file

        if not extracted_text.strip():
            return jsonify({"reply": "No readable text found in document."})

        session_id = "default_user"

        # # Step 1: Verify Document Authenticity
        # verification_prompt = (
        #     "You are a customs document verification assistant. "
        #     "Analyze the following text and check if it looks like a valid customs document "
        #     "(invoice, bill of lading, or customs declaration). "
        #     "Identify if any key details are missing (like HS code, country of origin, product description, or value). "
        #     "Reply with '✅ Verified: [reason]' if correct, or '❌ Invalid: [reason]' if issues found.\n\n"
        #     f"Document Text:\n{extracted_text[:3000]}"
        # )

        verification_prompt = (
    "You are a **Customs Document Verification Assistant**.\n"
    "Your task is to carefully analyze the provided document text and determine:\n"
    "1️⃣ Whether it is a valid customs-related document (e.g., Invoice, Bill of Lading, Customs Declaration, or Packing List).\n"
    "2️⃣ Whether all essential trade details are present:\n"
    "   - HS Code\n"
    "   - Product Description\n"
    "   - Country of Origin\n"
    "   - Quantity & Value\n"
    "   - Exporter/Importer Details\n"
    "3️⃣ Whether the document appears **authentic and complete** (no missing or inconsistent data).\n\n"
    "Respond strictly in the following format:\n"
    "📄 Document Type: [Invoice / Bill of Lading / Customs Declaration / Other / Unknown]\n"
    "📋 Document Status: [✅ Correct / ❌ Incorrect]\n"
    "🔍 Verification Result: [✅ Verified: reason] or [❌ Invalid: reason]\n"
    "💡 Missing or Suspicious Details (if any): [List clearly]\n\n"
    f"Document Text:\n{extracted_text[:3000]}"
)


        verification_response = llm.invoke(verification_prompt)
        verification_result = verification_response.content.strip()
        print("📄 Verification Result:", verification_result)

        # Step 2: Use RAG to generate customs explanation
        response = rag_chain_with_memory.invoke(
            {"input": extracted_text},
            config={"configurable": {"session_id": session_id}}
        )

        english_answer = response["answer"]

        # --- Detect and restrict translation languages ---
        try:
            detected_lang = detect(extracted_text)
        except:
            detected_lang = "en"

        supported_langs = {"en": "en", "hi": "hi", "ne": "ne","mai": "mai"}
        user_lang = supported_langs.get(detected_lang, "en")

        if user_lang in ["hi", "ne","mai"]:
            translated_answer = GoogleTranslator(source="en", target=user_lang).translate(english_answer)
            translated_verification = GoogleTranslator(source="en", target=user_lang).translate(verification_result)
        else:
            translated_answer = english_answer
            translated_verification = verification_result

        return jsonify({
            "verification": translated_verification,
            "analysis": translated_answer
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
