
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image, UnidentifiedImageError
import io
import speech_recognition as sr
import os
import google.generativeai as genai
import uuid
from datetime import datetime
import logging
import base64  # <--- MISSING IMPORT
import re
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# In-memory storage
stored_texts = {}

def store_extracted_text(text: str, summary: str):
    """Store extracted text along with AI summary"""
    entry_id = str(uuid.uuid4())
    stored_texts[entry_id] = {
        "id": entry_id,
        "text": text,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }
    return entry_id

def is_valid_uuid(uuid_str):
    """Check if string is a valid UUID"""
    uuid_regex = re.compile(r'^[a-f0-9]{8}-?[a-f0-9]{4}-?[4][a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}$', re.I)
    return bool(uuid_regex.match(uuid_str))

def extract_text(image_data):
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip() or "No text found in image"
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def generate_academic_summary(text: str) -> str:
    """Generate academic summary using Gemini API"""
    if not GEMINI_API_KEY:
        return "Summary unavailable: API key missing"
    
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""Analyze this academic transcript and provide a concise summary with:
        1. Overall performance assessment
        2. Strongest and weakest subjects
        3. Notable achievements
        4. Areas for improvement
        5. Final TGPA interpretation
        
        Transcript: {text[:3000]}"""  # Limit input size
        
        response = model.generate_content(prompt)
        return response.text if response.text else "No summary generated"
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return f"Summary generation failed: {str(e)}"

async def speak_output(text):
    """Text-to-speech using Base64 encoding for safe transmission"""
    try:
        # Clean markdown formatting and newlines
        cleaned = re.sub(r"[#\-*]", "", text)  # Remove markdown symbols
        cleaned = cleaned.replace("\n", " ")  # Replace newlines with spaces
        
        # Encode text to avoid PowerShell escaping issues
        encoded_text = base64.b64encode(cleaned.encode('utf-16-le')).decode()
        
        # Create PowerShell command
        command = (
            'powershell -Command "'
            '$text = [System.Text.Encoding]::Unicode.GetString('
            '[System.Convert]::FromBase64String(\'' + encoded_text + '\'));'
            'Add-Type -AssemblyName System.Speech;'
            '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            '$speak.Speak($text)"'
        )

        # Execute command with proper error handling
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for completion and check result
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"TTS failed: {stderr.decode().strip()}")
            raise HTTPException(status_code=500, detail="Speech synthesis failed")

    except Exception as e:
        logger.error(f"TTS error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image, generate academic summary, and speak the summary"""
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files are allowed"}
        )
    
    try:
        # Process image
        image_data = await file.read()
        extracted_text = extract_text(image_data)
        
        # Generate academic summary
        summary = generate_academic_summary(extracted_text)
        
        # Store extracted text and summary
        entry_id = store_extracted_text(extracted_text, summary)
        
        # Automatically call text_to_speech after processing
        await speak_output(summary)  # Trigger speech immediately after generating summary
        
        return {
            "success": True,
            "text": extracted_text,
            "summary": summary,
            "storage_id": entry_id,
            "message": "Summary spoken successfully"
        }
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
