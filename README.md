# 🧾 Doctor Alliance Billing Automation Bot

This project automates the verification of medical billing eligibility from a private web portal using **browser automation**, **OCR**, and **AI**.

> 🚫 **Note:** Core logic is redacted to comply with buyer company policies. This README documents the design and flow. Code involving portal URLs, DOM structure, and real credentials must not be shared publicly.

---

## 🚀 Overview

The automation system validates if a patient is billable under specific certification codes (e.g., G0180, G0179) by:

- Logging into the Doctor Alliance web portal
- Searching for patients using Name and Date of Birth
- Opening certification PDFs
- Extracting document content using **Tesseract OCR**
- Sending extracted text to **Gemini Pro** for reasoning
- Making a final billing decision based on AI output

---

## ⚙️ Tech Stack

| Tool            | Purpose                                       |
|-----------------|-----------------------------------------------|
| **Python 3.10+**| Core scripting language                        |
| **Selenium**    | Automates login, search, navigation            |
| **Tesseract OCR**| Extracts text from scanned certification PDFs |
| **Gemini Pro**  | AI reasoning on cert type/date for billing     |
| **ChromeDriver**| Browser automation driver                      |
| **Logging, OS** | I/O operations and logging                     |

---

## 🧰 Key Features

- 🔐 Secure, environment-based login (no hardcoded credentials)
- 🧑‍⚕️ Patient search using form automation
- 📄 OCR-based document scanning
- 🧠 Gemini-powered eligibility reasoning
- ✅ Structured logging for each patient billing outcome

---

## 🏁 Setup Instructions

1. **Clone repository** (internal only).

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt

   # 🧾 Execution Flow

1. Load patient list (Name and DOB).  
2. Launch browser via Selenium.  
3. Log in to the secure Doctor Alliance portal.  
4. Search for each patient.  
5. Open their certification document (PDF).  
6. Use Tesseract OCR to extract text from the document.  
7. Send the extracted text to Gemini Pro with a structured prompt.  
8. Gemini analyzes:
   - Certification type (e.g., G0180 / G0179)  
   - Certification start and end dates  
   - Whether the certification is valid for billing on the target date  
9. Log result as `BILLABLE` or `NOT BILLABLE`.

---

# 🤖 Gemini Prompt Example

```plaintext
Extracted Text:
"Certification Type: G0180
Start Date: 01/01/2024
End Date: 04/01/2024"

Prompt to Gemini:
"Based on this certification document text, is the patient billable under G0180 as of 02/15/2024?
Return YES/NO with reasoning."

Gemini Response:
"YES. G0180 is valid and active on 02/15/2024. The certification period covers the billing date."
```

# DIAGRAM
<img width="1169" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-07-26-134833" src="https://github.com/user-attachments/assets/7f8ef2ce-7b42-489c-9a20-a7d1cdd0682c" />

# 🛡️ Compliance & Legal

- Patient data is handled in a secure, sandboxed environment.  
- No Protected Health Information (PHI) is stored or transmitted externally.  
- Gemini AI is used in a stateless, read-only capacity strictly for reasoning over extracted text.  
- The tool adheres to internal organizational data handling policies and is compliant with buyer company confidentiality agreements.  
- All actions are performed within a closed, authenticated system to maintain data integrity and privacy.

