import os
import re
import base64
import io
from datetime import datetime
import requests
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
import fitz
from PIL import Image
import pytesseract
import numpy as np
import faiss
from tqdm import tqdm
import pickle

# Initialize logger
class Logger:
    def __init__(self):
        self.log_file = "cert_extraction_log.txt"
    
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
        print(f"[{timestamp}] [{level}] {message}")

logger = Logger()

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
TOKEN = os.getenv("AUTH_TOKEN")
DOC_API_URL = "https://api.doctoralliance.com/document/getfile?docId.id="
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def is_scanned_pdf(pdf_bytes):
    """Check if a PDF is scanned by analyzing its content"""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:3]:
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    return True
        return False
    except Exception as e:
        logger.log(f"Error checking if PDF is scanned: {e}")
        return True

def extract_text_with_ocr(pdf_bytes):
    """Extract text from scanned PDF using OCR"""
    try:
        logger.log("Using OCR to extract text from scanned document")
        text_parts = []
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            text_parts.append(page_text)
            
        pdf_document.close()
        full_text = "\n".join(text_parts)
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        return cleaned_text
    except Exception as e:
        logger.log(f"Error in OCR text extraction: {e}", "ERROR")
        raise

def get_pdf_text(doc_id):
    """Fetch and extract PDF text"""
    logger.log(f"Fetching document {doc_id}")
    try:
        response = requests.get(f"{DOC_API_URL}{doc_id}", headers=HEADERS)
        
        if response.status_code != 200:
            logger.log(f"Failed to fetch document: {response.text}", "ERROR")
            raise Exception("Failed to fetch document")
            
        document_buffer = response.json()["value"]["documentBuffer"]
        pdf_bytes = base64.b64decode(document_buffer)
        
        # Check if document is scanned 
        if is_scanned_pdf(pdf_bytes):
            logger.log("Document appears to be scanned, using OCR")
            text = extract_text_with_ocr(pdf_bytes)
        else:
            logger.log("Document appears to be digital, using PDFPlumber")
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                
        edited = re.sub(r'\b\d[A-Z][A-Z0-9]\d[A-Z][A-Z0-9]\d[A-Z]{2}(?:\d{2})?\b', '', text)
        return edited
        
    except Exception as e:
        logger.log(f"Error extracting PDF text: {e}", "ERROR")
        raise

# List of known CMS-485 doc IDs
KNOWN_CMS_485_DOC_IDS = [
    "9288276", "9287312", "9287204", "9287200", "9287190", "9287185",
    "9287179", "9287177", "9287167", "9287164", "9285299", "9285267",
    "9285067", "9285057", "9285033", "9285011", "9284880", "9283844",
    "9291787", "9291780", "9282768", "9282680", "9282675", "9290438",
    "9290409", "9290364", "9290352", "9290329", "9289588"
]

# File paths for saving
INDEX_PATH = "cms485_vector_index.index"
DOC_MAP_PATH = "cms485_doc_map.pkl"

def get_gemini_embedding(text):
    """Get embedding vector from Gemini for a given text"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Document Embedding"
        )
        return result["embedding"]
    except Exception as e:
        logger.log(f"Error generating embedding: {e}", "ERROR")
        raise

def build_and_save_vector_store(doc_ids):
    """Build FAISS index and save to disk"""
    logger.log("Building and saving vector store...")
    embeddings = []
    doc_map = {}

    for idx, doc_id in enumerate(tqdm(doc_ids)):
        try:
            full_text = get_pdf_text(doc_id)
            embedding = get_gemini_embedding(full_text)
            embeddings.append(embedding)
            doc_map[idx] = doc_id
        except Exception as e:
            logger.log(f"Error processing {doc_id}: {e}", "ERROR")

    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Build FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    # Save index and mapping
    faiss.write_index(index, INDEX_PATH)
    with open(DOC_MAP_PATH, "wb") as f:
        pickle.dump(doc_map, f)

    logger.log("Vector store saved to disk.")
    return index, doc_map

def load_vector_store():
    """Load FAISS index and doc map from disk if available"""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOC_MAP_PATH):
        logger.log("No saved index found. Building new one.", "WARNING")
        return build_and_save_vector_store(KNOWN_CMS_485_DOC_IDS)

    logger.log("Loading vector store from disk...")
    index = faiss.read_index(INDEX_PATH)
    with open(DOC_MAP_PATH, "rb") as f:
        doc_map = pickle.load(f)
    logger.log("Vector store loaded successfully.")
    return index, doc_map

def classify_cms_485(doc_id, index, doc_map, threshold=0.8):
    """Classify if a new document is CMS-485 based on similarity to known examples"""
    logger.log(f"Classifying document {doc_id}...")

    try:
        full_text = get_pdf_text(doc_id)
        query_embedding = get_gemini_embedding(full_text)
        query_np = np.array([query_embedding]).astype('float32')

        # Search in FAISS
        D, I = index.search(query_np, k=3)

        # Calculate cosine similarity approximation
        similarities = 1 - (D[0] / 2)  # L2 distance → approx Cosine Similarity
        avg_similarity = np.mean(similarities)

        # Map results
        nearest_docs = [(doc_map[i], sim) for i, sim in zip(I[0], similarities)]

        logger.log(f"Nearest matches: {nearest_docs}")
        logger.log(f"Average similarity: {avg_similarity:.2f}")

        if avg_similarity >= threshold:
            return {"is_cms_485": True, "matches": nearest_docs}
        else:
            return {"is_cms_485": False, "matches": nearest_docs}

    except Exception as e:
        logger.log(f"Error classifying {doc_id}: {e}", "ERROR")
        return {"is_cms_485": False, "error": str(e)}

if __name__ == "__main__":
    logger.log("Starting CMS-485 classification pipeline...", "INFO")

    # Step 1: Load or build vector store
    index, doc_map = load_vector_store()

    # Step 2: Classify unknown document
    test_doc_id = "9174425"
    result = classify_cms_485(test_doc_id, index, doc_map)

    # Step 3: Output final result
    print(f"\n✅ Document {test_doc_id} is CMS-485? {'YES' if result['is_cms_485'] else 'NO'}")
    print("Top Matches:", result["matches"])