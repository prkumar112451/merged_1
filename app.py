from fastapi import FastAPI, Form, HTTPException
from typing import Optional
import logging
import subprocess
import traceback
import uvicorn
from ner_simple import ner_simple
from ner_sequential import ner_sequential
from summarize import summarize_text
from pydantic import BaseModel

app = FastAPI()

# Configure logging
logging.basicConfig(
    filename='app.log',  # Log to a file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger(__name__)

class NERRequest(BaseModel):
    Text: str


class TranslationRequest(BaseModel):
    Text: str
    FromLanguageCode : str
    ToLanguageCode : str

class SummarizeRequest(BaseModel):
    Text: str

def get_gpu_metrics():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            gpu_data = []
            for line in output_lines:
                metrics = line.split(',')
                if len(metrics) == 4:  # Check if the expected number of metrics is present
                    gpu_data.append({
                        'Utilisation': float(metrics[0]),
                        'Temperature': float(metrics[1]),
                        'MemoryUsed': int(metrics[2]),
                        'MemoryTotal': int(metrics[3])
                    })
            return gpu_data
        else:
            logger.error(f"Error running nvidia-smi: {result.stderr}")
            return []
    except Exception as e:
        logger.error("Error getting GPU metrics: %s", e)
        return []

@app.get('/gpu/metrics')
def get_gpu_metrics_route():
    gpu_metrics = get_gpu_metrics()
    return gpu_metrics

@app.post('/ner/sequential-entities')
async def named_entity_recognition(request : NERRequest):
    try:
        sentence = request.Text
        logger.info(f"Received sentence: {sentence}")
        if sentence is None:
            raise HTTPException(status_code=400, detail="No sentence provided")
        return ner_sequential(sentence)
    except Exception as e:
        logger.error("Error processing NER request: %s", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error processing NER request")

@app.post('/ner/simple-entities')
async def named_entity_recognition(request : NERRequest):
    try:
        sentence = request.Text
        logger.info(f"Received sentence: {sentence}")
        if sentence is None:
            raise HTTPException(status_code=400, detail="No sentence provided")
        return ner_simple(sentence)
    except Exception as e:
        logger.error("Error processing NER request: %s", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error processing NER request")

@app.post('/summarization')
async def summarization(request : SummarizeRequest):
    try:
        sentence = request.Text
        logger.info(f"Received sentence: {sentence}")
        if sentence is None:
            raise HTTPException(status_code=400, detail="No sentence provided")
        return summarize_text(sentence)
    except Exception as e:
        logger.error("Error processing NER request: %s", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error processing NER request")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Added port for clarity
