
import os
import time
import uuid
import asyncio
import functools
from dotenv import load_dotenv
from typing import List
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import redis.asyncio as async_redis
import redis  # Separate sync Redis for threaded operations
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from together import Together

# Load .env file
load_dotenv()
os.environ["TOGETHER_API_KEY"] = os.getenv("Together")

# Setup ASYNC Redis client for main application
async_redis_client = async_redis.Redis(
    host=os.getenv("Redis_endpoint", "").split(":")[0],
    port=int(os.getenv("Redis_port", "17037")),
    username="default",
    password=os.getenv("Redis_password"),
    decode_responses=True,
    ssl=False  # Based on your working config
)

# Setup SYNC Redis client for threaded operations
sync_redis_client = redis.Redis(
    host=os.getenv("Redis_endpoint", "").split(":")[0],
    port=int(os.getenv("Redis_port", "17037")),
    username="default",
    password=os.getenv("Redis_password"),
    decode_responses=True,
    ssl=False
)

# Test connection function
async def test_redis_connection():
    try:
        # Test async client
        await async_redis_client.set('test_key', 'test_value')
        result = await async_redis_client.get('test_key')
        await async_redis_client.delete('test_key')
        print(f"✅ Async Redis connection successful: {result}")
        
        # Test sync client
        sync_result = sync_redis_client.ping()
        print(f"✅ Sync Redis connection successful: {sync_result}")
        
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

# API and routing
router = APIRouter()
task_queue = asyncio.Queue()
thread_pool = ThreadPoolExecutor(max_workers=3)
NUM_WORKERS = 3

class PromptList(BaseModel):
    prompts: List[str]

# LLM task processor (synchronous function run in thread)
def process_llm_task(task_id: str, prompt: str):
    try:
        print(f"🚀 Starting task {task_id[:8]}...")
        
        # Initialize Together client with API key
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        
        # Mark 25% progress - use SYNC client in threaded function
        sync_redis_client.hset(f"task:{task_id}", "progress", "25")
        print(f"📊 Task {task_id[:8]} - 25% complete")

        # Call LLM API
        response = client.chat.completions.create(
            model=os.getenv("Model"),
            messages=[{
                "role": "user",
                "content": prompt
            }],
            stream=False,
            max_tokens=1000,
            temperature=0.7
        )

        # Mark 75% progress
        sync_redis_client.hset(f"task:{task_id}", "progress", "75")
        print(f"📊 Task {task_id[:8]} - 75% complete")

        # Final result and status
        result_content = response.choices[0].message.content
        sync_redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "completed",
                "result": result_content,
                "progress": "100",
                "completed_at": str(int(time.time()))
            }
        )
        
        print(f"✅ Task {task_id[:8]} completed successfully")

    except Exception as e:
        print(f"❌ Task {task_id[:8]} failed with error: {e}")
        sync_redis_client.hset(
            f"task:{task_id}",
            mapping={
                "status": "failed", 
                "error": str(e),
                "failed_at": str(int(time.time()))
            }
        )

# Async worker that runs the LLM task
async def background_worker(worker_id: int):
    print(f"🔄 Worker {worker_id} started")
    while True:
        try:
            task_id, prompt = await task_queue.get()
            print(f"👷 Worker {worker_id} picked up task {task_id[:8]}")
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                thread_pool,
                functools.partial(process_llm_task, task_id, prompt)
            )
            task_queue.task_done()
            
        except Exception as e:
            print(f"❌ Worker {worker_id} error: {e}")

# Lifespan event for starting background workers
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application starting up...")
    
    # Test Redis connection
    connection_ok = await test_redis_connection()
    if not connection_ok:
        raise Exception("❌ Cannot start without Redis connection")
    
    # Start background workers
    workers = []
    for i in range(NUM_WORKERS):
        worker = asyncio.create_task(background_worker(i + 1))
        workers.append(worker)
    
    print(f"✅ Started {NUM_WORKERS} background workers")
    print("🎉 Application ready!")
    
    yield  # App runs here
    
    # Cleanup
    print("🛑 Shutting down workers...")
    for worker in workers:
        worker.cancel()
    
    try:
        await async_redis_client.aclose()
        print("✅ Redis connections closed")
    except:
        pass

# FastAPI app with lifespan manager
app = FastAPI(
    title="LLM Task Queue API",
    description="Async LLM processing with Redis queue",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json"  # OpenAPI schema
)

# Health check endpoint
@router.get("/health", tags=["Health"], summary="Check API health")
async def health_check():
    try:
        ping_result = await async_redis_client.ping()
        return {
            "status": "healthy",
            "redis": "connected",
            "queue_size": task_queue.qsize(),
            "redis_ping": ping_result
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e)
        }

# Endpoint to queue new tasks
@router.post("/tasks", tags=["Tasks"], summary="Queue new LLM tasks")
async def create_tasks(prompts: PromptList):
    try:
        task_ids = []
        for prompt in prompts.prompts:
            task_id = str(uuid.uuid4())
            
            # Use AWAIT with async Redis client
            await async_redis_client.hset(
                f"task:{task_id}",
                mapping={
                    "status": "queued", 
                    "progress": "0",
                    "created_at": str(int(time.time())),
                    "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt
                }
            )
            
            await task_queue.put((task_id, prompt))
            task_ids.append(task_id)
        
        return {
            "task_ids": task_ids, 
            "status": "queued",
            "message": f"Successfully queued {len(task_ids)} tasks",
            "queue_size": task_queue.qsize()
        }
        
    except Exception as e:
        return {
            "error": f"Failed to queue tasks: {str(e)}",
            "status": "error"
        }

# Endpoint to query task status
@router.get("/tasks", tags=["Tasks"], summary="Get task status by IDs")
async def get_task_status(task_ids: str):
    try:
        task_id_list = [tid.strip() for tid in task_ids.split(",")]
        results = {}
        
        for task_id in task_id_list:
            # Use AWAIT with async Redis client
            task_data = await async_redis_client.hgetall(f"task:{task_id}")
            
            if task_data:
                # Clean up response (remove prompt_preview for cleaner output)
                clean_data = {k: v for k, v in task_data.items() if k != 'prompt_preview'}
                results[task_id] = clean_data
            else:
                results[task_id] = {"status": "not_found"}
        
        return {
            "tasks": results,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get task status: {str(e)}",
            "status": "error"
        }

# Endpoint to get specific task result
@router.get("/tasks/{task_id}/result", tags=["Tasks"], summary="Get specific task result")
async def get_task_result(task_id: str):
    try:
        task_data = await async_redis_client.hgetall(f"task:{task_id}")
        
        if not task_data:
            return {"error": "Task not found", "status": "not_found"}
        
        return {
            "task_id": task_id,
            "status": task_data.get("status"),
            "progress": task_data.get("progress"),
            "result": task_data.get("result"),
            "error": task_data.get("error"),
            "created_at": task_data.get("created_at"),
            "completed_at": task_data.get("completed_at"),
            "failed_at": task_data.get("failed_at")
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get task result: {str(e)}",
            "status": "error"
        }

# Endpoint to get queue statistics
@router.get("/queue/stats", tags=["Queue"], summary="Get queue statistics")
async def get_queue_stats():
    try:
        # Get all task keys
        task_keys = await async_redis_client.keys("task:*")
        
        stats = {"total": len(task_keys), "queued": 0, "completed": 0, "failed": 0}
        
        for key in task_keys:
            task_data = await async_redis_client.hgetall(key)
            status = task_data.get("status", "unknown")
            if status in ["queued", "completed", "failed"]:
                stats[status] += 1
        
        stats["current_queue_size"] = task_queue.qsize()
        
        return {"stats": stats, "status": "success"}
        
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}", "status": "error"}

# Include router in app
app.include_router(router, prefix="/api/v1", tags=["Tasks"])

# Root endpoint (this should be directly on app, not router)
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🤖 LLM Task Queue API",
        "version": "1.0.0",
        "status": "running",
        "swagger_ui": "http://localhost:8000/docs",
        "redoc": "http://localhost:8000/redoc",
        "endpoints": {
            "health": "/api/v1/health",
            "queue_tasks": "POST /api/v1/tasks",
            "get_status": "GET /api/v1/tasks?task_ids=id1,id2",
            "get_result": "GET /api/v1/tasks/{task_id}/result",
            "queue_stats": "GET /api/v1/queue/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LLM Task Queue API...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True
    )