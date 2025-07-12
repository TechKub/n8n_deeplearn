from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, shutil, os
import torch
from dotenv import load_dotenv
import whisperx
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing HUGGINGFACE_TOKEN")

app = FastAPI()

# Global variables for models
model = None
device = None
compute_type = None

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:        
        # Load audio and transcribe
        logger.info("Loading audio...")
        audio = whisperx.load_audio(tmp_path)
        
        logger.info("Transcribing...")
        result = model.transcribe(audio, batch_size=16)
        
        if not result.get("segments"):
            return JSONResponse(content={"error": "No speech detected"})

        # Alignment
        logger.info("Aligning...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result_aligned = whisperx.align(
            result["segments"], model_a, metadata, audio, device
        )

        # Diarization
        logger.info("Diarizing...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HUGGINGFACE_TOKEN, device=device
        )
        diarize_segments = diarize_model(tmp_path)

        # Assign speakers to words
        logger.info("Assigning speakers...")
        result_with_speakers = whisperx.assign_word_speakers(
            diarize_segments, result_aligned
        )

        return JSONResponse(content=result_with_speakers)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})            

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)