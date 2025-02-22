
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image, UnidentifiedImageError
import io
import speech_recognition as sr
import os
from typing import Dict
import uuid
from datetime import datetime

app = FastAPI()

# In-memory storage for extracted texts (you might want to use a database in production)
stored_texts: Dict[str, dict] = {}
current_id = 1

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip() or "No text found in image"
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def store_extracted_text(text: str):
    global current_id
    entry_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    stored_texts[entry_id] = {
        "id": entry_id,
        "text": text,
        "timestamp": timestamp
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
        entry_id = store_extracted_text(extracted_text)
        
        return {
            "success": True,
            "text": extracted_text,
            "storage_id": entry_id,
            "message": "Text stored successfully"
        }
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"error": he.detail}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )

@app.get("/texts/", summary="Get all stored texts")
async def get_all_texts():
    return {
        "success": True,
        "count": len(stored_texts),
        "results": list(stored_texts.values())
    }

@app.get("/texts/{text_id}", summary="Get specific stored text by ID")
async def get_text_by_id(text_id: str):
    entry = stored_texts.get(text_id)
    if not entry:
        return JSONResponse(
            status_code=404,
            content={"error": "Text entry not found"}
        )
    return {"success": True, "result": entry}

# ... (keep existing voice and speak endpoints unchanged)
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
