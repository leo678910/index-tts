import io
import os
import soundfile as sf
import hashlib
import uuid
import threading
import time
import datetime
import logging
import traceback
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException, Path, status, Request
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel
from typing import Dict, Any

# V2-CHANGE: 导入 IndexTTS2 而不是 IndexTTS
try:
    from indextts.infer_v2 import IndexTTS2
    TTS_AVAILABLE = True
    logging.info("IndexTTS2 library imported successfully.")
except ImportError as e:
    logging.error(f"Failed to import IndexTTS2: {e}. TTS functionality will be disabled.")
    IndexTTS2 = None
    TTS_AVAILABLE = False

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

app = FastAPI(title="IndexTTS v2 API - Async")

# --- Global State & Configuration (保持不变) ---
tasks_status: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()
PROMPTS_DIR = "prompts"
OUTPUT_DIR = "synthesized_audio"
CHECKPOINTS_DIR = "checkpoints"
CONFIG_PATH = os.path.join(CHECKPOINTS_DIR, "config.yaml")

logger.info(f"PROMPTS_DIR: {os.path.abspath(PROMPTS_DIR)}")
logger.info(f"OUTPUT_DIR: {os.path.abspath(OUTPUT_DIR)}")
logger.info(f"CHECKPOINTS_DIR: {os.path.abspath(CHECKPOINTS_DIR)}")
logger.info(f"CONFIG_PATH: {os.path.abspath(CONFIG_PATH)}")

os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# --- TTS Model Initialization (V2-CHANGE) ---
tts_model = None
if TTS_AVAILABLE:
    logger.info("Attempting to initialize IndexTTS2 model at app startup...")
    try:
        if not os.path.isdir(CHECKPOINTS_DIR):
            raise FileNotFoundError(f"Model directory '{CHECKPOINTS_DIR}' not found.")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file '{CONFIG_PATH}' not found.")
        
        # V2-CHANGE: 初始化 IndexTTS2，并开启所有性能开关
        tts_model = IndexTTS2(
            model_dir=CHECKPOINTS_DIR,
            cfg_path=CONFIG_PATH,
            use_fp16=True,         # 开启FP16
            use_cuda_kernel=True,  # 开启自定义CUDA Kernel
            use_deepspeed=True     # 开启DeepSpeed
        )
        logger.info("IndexTTS2 model initialized successfully at app startup.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize IndexTTS2 model: {e}")
        logger.error(traceback.format_exc())
        tts_model = None
        TTS_AVAILABLE = False
else:
     logger.warning("IndexTTS2 library not found. TTS endpoints will be unavailable.")

# --- Utility Functions (保持不变) ---
def hash_filename(filename: str) -> str:
    logger.debug(f"Hashing filename: {filename}")
    try:
        normalized_filename = os.path.basename(filename)
        ext = os.path.splitext(normalized_filename)[1].lower()
        h = hashlib.md5(normalized_filename.encode('utf-8')).hexdigest()
        hashed = f"{h}{ext}"
        logger.debug(f"Hashed '{filename}' to '{hashed}'")
        return hashed
    except Exception as e:
        logger.error(f"Error hashing filename '{filename}': {e}")
        raise ValueError(f"Could not hash filename: {filename}") from e

# --- Background Task Function (V2-CHANGE) ---
def run_synthesis_task(task_id: str, text: str, original_ref_filename: str):
    thread_name = threading.current_thread().name
    logger.info(f"[{task_id}] V2 Background task started in thread '{thread_name}'.")
    start_time_iso = datetime.datetime.now().isoformat()
    output_filepath = None

    # 更新状态为 RUNNING
    try:
        with tasks_lock:
             tasks_status[task_id]["status"] = "RUNNING"
             tasks_status[task_id]["start_time"] = start_time_iso
             logger.info(f"[{task_id}] Status updated to RUNNING.")
    except Exception as e:
         logger.error(f"[{task_id}] CRITICAL: Failed to update task status to RUNNING: {e}")
         return
         
    # 执行合成
    try:
        if not TTS_AVAILABLE or tts_model is None:
            raise RuntimeError("TTS Model is not available for inference.")

        hashed_ref_filename = hash_filename(original_ref_filename)
        ref_audio_path = os.path.join(PROMPTS_DIR, hashed_ref_filename)
        output_filename = f"{task_id}.wav"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        logger.info(f"[{task_id}] Ref path: {ref_audio_path}, Output path: {output_filepath}")

        if not os.path.isfile(ref_audio_path):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

        logger.info(f"[{task_id}] Starting IndexTTS2 inference...")
        
        result_path = tts_model.infer(
            intelligent_split_len=25,
            #num_beams=1,
            spk_audio_prompt=ref_audio_path, 
            text=text, 
            output_path=output_filepath, # 直接指定输出路径
            verbose=True # 在日志中打印详细推理信息
        )
        
        # V2-CHANGE: 检查返回路径是否与期望一致
        if result_path != output_filepath:
             logger.warning(f"[{task_id}] Inference returned a different path '{result_path}' than expected '{output_filepath}'. Using returned path.")
             # 如果路径不同，你可能需要决定如何处理，这里我们信任返回的路径
             # 但为了统一，最好是 tts_model.infer 直接使用我们给的路径
        
        logger.info(f"[{task_id}] IndexTTS2 inference finished. Audio saved at: {result_path}")

        # 更新状态为 SUCCESS
        with tasks_lock:
             if task_id in tasks_status:
                 tasks_status[task_id].update({
                     "status": "SUCCESS",
                     "result": output_filepath, # 依然使用我们定义的路径结构
                     "end_time": datetime.datetime.now().isoformat()
                 })
                 logger.info(f"[{task_id}] Status updated to SUCCESS.")
             else:
                  logger.warning(f"[{task_id}] Task disappeared before status update to SUCCESS.")

    except Exception as e:
        error_message = f"Synthesis failed: {type(e).__name__} - {str(e)}"
        logger.error(f"[{task_id}] FAILURE: {error_message}")
        logger.error(traceback.format_exc())
        with tasks_lock:
             if task_id in tasks_status:
                tasks_status[task_id].update({
                    "status": "FAILURE",
                    "result": error_message,
                    "end_time": datetime.datetime.now().isoformat()
                })
        # 清理可能产生的失败文件
        if output_filepath and os.path.exists(output_filepath):
             try: os.remove(output_filepath)
             except OSError: pass
    finally:
        logger.info(f"[{task_id}] Background task finished.")
        sys.stdout.flush()
        sys.stderr.flush()

# --- Pydantic Models (保持不变) ---
class TextToSpeechRequest(BaseModel):
    text: str
    filename: str

class SynthesisTaskSubmitResponse(BaseModel):
    task_id: str
    status: str
    status_url: str
    result_url: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Any | None = None
    start_time: str | None = None
    end_time: str | None = None

# --- API Endpoints (保持不变, 逻辑自适应) ---
@app.post("/v1/synthesize", status_code=status.HTTP_202_ACCEPTED, response_model=SynthesisTaskSubmitResponse)
async def synthesize_speech_submit(request_body: TextToSpeechRequest, http_request: Request):
    # ... 这部分代码与 v1 完全相同，因为它的设计是通用的 ...
    logger.info(f"--- [synthesize] Request received for '{request_body.filename}' ---")
    if not TTS_AVAILABLE or tts_model is None:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTS Service Unavailable")
    if not request_body.text or not request_body.filename:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'text' or 'filename'")
    try:
        hashed_ref_filename = hash_filename(request_body.filename)
        ref_audio_path = os.path.join(PROMPTS_DIR, hashed_ref_filename)
        if not os.path.isfile(ref_audio_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Reference audio '{request_body.filename}' not found.")
            
        task_id = str(uuid.uuid4())
        logger.info(f"[{task_id}] Generated Task ID.")
        
        with tasks_lock:
            tasks_status[task_id] = {
                "status": "PENDING", "result": None, "submit_time": datetime.datetime.now().isoformat(),
                "start_time": None, "end_time": None
            }
        
        thread = threading.Thread(
            target=run_synthesis_task,
            args=(task_id, request_body.text, request_body.filename),
            name=f"SynthTask-{task_id[:8]}",
            daemon=True
        )
        thread.start()
        
        logger.info(f"[{task_id}] Background thread started. Returning 202 Accepted.")
        base_url = str(http_request.base_url).rstrip('/')
        status_url = f"{base_url}{app.url_path_for('get_synthesis_status', task_id=task_id)}"
        result_url = f"{base_url}{app.url_path_for('get_synthesis_result', task_id=task_id)}"
        
        return SynthesisTaskSubmitResponse(
            task_id=task_id, status="PENDING", status_url=status_url, result_url=result_url
        )
    except HTTPException:
         raise
    except Exception as e:
         logger.error(f"[synthesize] Unexpected error: {e}")
         logger.error(traceback.format_exc())
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

# 其他端点 (@app.get("/v1/synthesize/status/{task_id}", ...), @app.get("/v1/synthesize/result/{task_id}", ...))
# 完全保持不变，因为它们只与 tasks_status 这个字典交互，与TTS模型无关。
@app.get("/v1/synthesize/status/{task_id}", response_model=TaskStatusResponse)
async def get_synthesis_status(task_id: str = Path(..., title="The ID of the task to check")):
    logger.debug(f"[status] Request for Task ID: {task_id}")
    with tasks_lock:
        task_info = tasks_status.get(task_id, {}).copy()
    if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    return TaskStatusResponse(task_id=task_id, **task_info)

@app.get("/v1/synthesize/result/{task_id}")
async def get_synthesis_result(task_id: str = Path(..., title="The ID of the completed task")):
    logger.info(f"[result] Request for Task ID: {task_id}")
    with tasks_lock:
        task_info = tasks_status.get(task_id, {}).copy()
    if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    
    current_status = task_info.get("status")
    result_data = task_info.get("result")

    if current_status != "SUCCESS":
        detail = f"Task '{task_id}' not complete. Status: {current_status}."
        if current_status == "FAILURE": detail += f" Error: {result_data}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    if not isinstance(result_data, str) or not os.path.isfile(result_data):
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Result file not found for task '{task_id}'.")
    
    return FileResponse(path=result_data, media_type='audio/wav', filename=f"{task_id}_result.wav")


@app.get("/v1/models")
async def get_model(): return {"code": 200, "model_list": []}

@app.get("/v1/check/audio")
async def check_audio(file_name :str):
    hashed_name = hash_filename(file_name)
    audio_path = os.path.join(PROMPTS_DIR, hashed_name)
    return JSONResponse(content={"exists": os.path.isfile(audio_path)})

class UploadResp(BaseModel): code: int; msg: str
@app.post("/v1/upload_audio", response_model=UploadResp)
async def upload_audio(audio: UploadFile = File(...)):
    content = await audio.read()
    hashed_name = hash_filename(audio.filename)
    save_path = os.path.join(PROMPTS_DIR, hashed_name)
    with open(save_path, "wb") as f: f.write(content)
    return UploadResp(code=200, msg="上传成功!")

@app.get("/")
async def info(): return {"message": "IndexTTS v2 API - Async"}

if __name__ == "__main__":
    import uvicorn
    logger.info("--- Starting Uvicorn server for IndexTTS v2 API ---")
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, log_level="info", reload=False)
