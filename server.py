
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import pytesseract
# from PIL import Image, UnidentifiedImageError
# import io
# import speech_recognition as sr
# import os
# from typing import Dict
# import uuid
# from datetime import datetime

# app = FastAPI()

# # In-memory storage for extracted texts (you might want to use a database in production)
# stored_texts: Dict[str, dict] = {}
# current_id = 1

# # Configure Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def extract_text(image_data):
#     try:
#         image = Image.open(io.BytesIO(image_data))
#         text = pytesseract.image_to_string(image)
#         return text.strip() or "No text found in image"
#     except UnidentifiedImageError:
#         raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

# def store_extracted_text(text: str):
#     global current_id
#     entry_id = str(uuid.uuid4())
#     timestamp = datetime.now().isoformat()
#     stored_texts[entry_id] = {
#         "id": entry_id,
#         "text": text,
#         "timestamp": timestamp
#     }
#     return entry_id

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     if not file.content_type.startswith('image/'):
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Only image files are allowed"}
#         )
    
#     try:
#         image_data = await file.read()
#         extracted_text = extract_text(image_data)
#         entry_id = store_extracted_text(extracted_text)
        
#         return {
#             "success": True,
#             "text": extracted_text,
#             "storage_id": entry_id,
#             "message": "Text stored successfully"
#         }
#     except HTTPException as he:
#         return JSONResponse(
#             status_code=he.status_code,
#             content={"error": he.detail}
#         )
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Unexpected error: {str(e)}"}
#         )

# @app.get("/texts/", summary="Get all stored texts")
# async def get_all_texts():
#     return {
#         "success": True,
#         "count": len(stored_texts),
#         "results": list(stored_texts.values())
#     }

# @app.get("/texts/{text_id}", summary="Get specific stored text by ID")
# async def get_text_by_id(text_id: str):
#     entry = stored_texts.get(text_id)
#     if not entry:
#         return JSONResponse(
#             status_code=404,
#             content={"error": "Text entry not found"}
#         )
#     return {"success": True, "result": entry}

# # ... (keep existing voice and speak endpoints unchanged)
# @app.get("/voice/")
# async def get_voice_command():
#     try:
#         command = recognize_speech()
#         return {"success": True, "command": command}
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Voice recognition failed: {str(e)}"}
#         )

# @app.get("/speak/")
# async def text_to_speech(text: str):
#     try:
#         speak_output(text)
#         return {"success": True, "message": "Text spoken successfully"}
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Text-to-speech failed: {str(e)}"}
#         )
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image, UnidentifiedImageError
import io
import speech_recognition as sr
import os
import requests
from typing import Dict
import uuid
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("sk-438b8f8f9a9a4a3084aceb0277163218")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# In-memory storage
stored_texts: Dict[str, dict] = {}

def extract_text(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip() or "No text found in image"
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def describe_with_deepseek(text: str) -> str:
    """Generate description using DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "Description unavailable: API key missing"
    
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    prompt = f"Analyze this text extracted from an image: {text}"
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        return f"Description generation failed: {str(e)}"

def recognize_speech():
    """Voice recognition function"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            logger.info("Listening for voice command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10)
            
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        return "No speech detected"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech service error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def speak_output(text):
    """Text-to-speech function"""
    try:
        # For Windows
        if os.name == 'nt':
            os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{text}\')"')
        # For Linux/MacOS
        else:
            os.system(f'espeak "{text}"')
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")

def store_extracted_text(text: str, description: str):
    entry_id = str(uuid.uuid4())
    stored_texts[entry_id] = {
        "id": entry_id,
        "text": text,
        "description": description,
        "timestamp": datetime.now().isoformat()
    }
    return entry_id

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only image files are allowed"}
        )
    
    try:
        image_data = await file.read()
        extracted_text = extract_text(image_data)
        description = describe_with_deepseek(extracted_text)
        entry_id = store_extracted_text(extracted_text, description)
        
        return {
            "success": True,
            "text": extracted_text,
            "description": description,
            "storage_id": entry_id
        }
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"error": he.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/voice/")
async def get_voice_command():
    try:
        command = recognize_speech()
        return {"success": True, "command": command}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Voice recognition failed: {str(e)}"}
        )

@app.get("/speak/")
async def text_to_speech(text: str):
    try:
        speak_output(text)
        return {"success": True, "message": "Text spoken successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Text-to-speech failed: {str(e)}"}
        )

@app.get("/texts/{text_id}")
async def get_text_by_id(text_id: str):
    if text_id not in stored_texts:
        return JSONResponse(
            status_code=404,
            content={"error": "Text entry not found"}
        )
    return {"success": True, "result": stored_texts[text_id]}

@app.get("/texts/")
async def get_all_texts():
    return {
        "success": True,
        "count": len(stored_texts),
        "results": list(stored_texts.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)