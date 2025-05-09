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
import json
# Initialize logger (simplified version)
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
            # Check first few pages 
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
                
        # Clean up the text
        edited = re.sub(r'\b\d[A-Z][A-Z0-9]\d[A-Z][A-Z0-9]\d[A-Z]{2}(?:\d{2})?\b', '', text)
        return edited
        
    except Exception as e:
        logger.log(f"Error extracting PDF text: {e}", "ERROR")
        raise

def extract_cert_period(text):
    """Extract certification period using Gemini"""
    logger.log("Extracting certification period using Gemini")
    
    prompt = """
    You are a medical document analysis expert. Please extract the certification period from this medical document.
    
    Look specifically for any mention of:
    - Certification period
    - Start of care date
    - Episode start date
    - Episode end date
    - Recertification dates
    
    Return your response in exactly this JSON format:
    {
      "cert_start": "MM/DD/YYYY",
      "cert_end": "MM/DD/YYYY"
    }
    
    If only one date is found, include just that one date. Only extract what is explicitly present in the document.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([text, prompt])
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
        
    except Exception as e:
        logger.log(f"Error extracting certification period: {e}", "ERROR")
        return {}

def process_document(doc_id):
    """Process a single document to extract certification period"""
    try:
        # Get PDF text
        full_text = get_pdf_text(doc_id)
        
        # Extract certification period
        cert_period = extract_cert_period(full_text)
        
        # Log result
        logger.log(f"Certification period for document {doc_id}: {cert_period}")
        return cert_period
        
    except Exception as e:
        logger.log(f"Error processing document {doc_id}: {e}", "ERROR")
        return {}

if __name__ == "__main__":
    # Example usage - replace with actual doc ID
    doc_id = "9208994"
    result = process_document(doc_id)
    print(f"Certification Period: {result}")