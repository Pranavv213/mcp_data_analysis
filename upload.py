# upload.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="File Upload API",
    description="API for uploading CSV files",
    version="1.0.0"
)

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store uploaded file info
uploaded_files = {}

@app.get("/")
async def root():
    return {
        "message": "File Upload API",
        "status": "running",
        "endpoints": {
            "/upload": "POST - Upload a CSV file",
            "/files": "GET - List all uploaded files",
            "/file/{filename}": "GET - Get file info",
            "/file/{filename}": "DELETE - Delete a file"
        }
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file - saves with original name in the current directory
    
    Args:
        file: CSV file to upload
    
    Returns:
        File information including path and metadata
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only CSV files are allowed"}
            )
        
        # Use the original filename directly
        original_filename = file.filename
        
        # Save file directly in the current working directory (same folder as server)
        file_path = os.path.join(os.getcwd(), original_filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            logger.warning(f"⚠️ File {original_filename} already exists, overwriting...")
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Store file info
        file_info = {
            "original_name": original_filename,
            "saved_name": original_filename,
            "path": file_path,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "uploaded_at": datetime.now().isoformat(),
            "exists": True
        }
        uploaded_files[original_filename] = file_info
        
        logger.info(f"✅ File uploaded: {original_filename} to {file_path}")
        
        return {
            "message": "File uploaded successfully",
            "filename": original_filename,
            "path": file_path,
            "size_mb": file_info["size_mb"],
            "uploaded_at": file_info["uploaded_at"]
        }
        
    except Exception as e:
        logger.error(f"❌ Error uploading file: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to upload file: {str(e)}"}
        )

@app.get("/files")
async def list_files():
    """
    List all uploaded files
    """
    # Check if files still exist on disk
    existing_files = []
    for filename, info in uploaded_files.items():
        if os.path.exists(info["path"]):
            info["exists"] = True
            existing_files.append(info)
        else:
            info["exists"] = False
            # Optionally remove from memory if file no longer exists
            # del uploaded_files[filename]
    
    return {
        "total": len(existing_files),
        "files": existing_files
    }

@app.get("/file/{filename}")
async def get_file_info(filename: str):
    """
    Get information about a specific file
    """
    if filename not in uploaded_files:
        # Check if file exists on disk but not in memory
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path) and filename.endswith('.csv'):
            # Add it to memory
            file_size = os.path.getsize(file_path)
            file_info = {
                "original_name": filename,
                "saved_name": filename,
                "path": file_path,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "uploaded_at": "Unknown (pre-existing file)",
                "exists": True
            }
            uploaded_files[filename] = file_info
            return file_info
        
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    # Check if file still exists on disk
    if not os.path.exists(uploaded_files[filename]["path"]):
        uploaded_files[filename]["exists"] = False
        return JSONResponse(
            status_code=404,
            content={"error": "File not found on disk"}
        )
    
    uploaded_files[filename]["exists"] = True
    return uploaded_files[filename]

@app.delete("/file/{filename}")
async def delete_file(filename: str):
    """
    Delete a specific file
    """
    # Check in memory first
    if filename in uploaded_files:
        file_path = uploaded_files[filename]["path"]
    else:
        # Check if file exists on disk
        file_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": "File not found"}
            )
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ File deleted: {filename}")
        
        # Remove from memory if exists
        if filename in uploaded_files:
            del uploaded_files[filename]
        
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"❌ Error deleting file: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete file: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "working_directory": os.getcwd(),
        "files_count": len(uploaded_files),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          📁 FILE UPLOAD API SERVER                       ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Host:        0.0.0.0                                    ║
    ║  Port:        8001                                       ║
    ║  URL:         http://localhost:8001                      ║
    ║  Upload Dir:  Same as server directory                   ║
    ║  Directory:   {os.getcwd():<40} ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "upload:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )