#!/usr/bin/env python3
"""
Startup script for the Therapy Prediction API Server
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies(auto_install: bool = True) -> bool:
    """Ensure required dependencies are installed.
    If missing and auto_install=True, install from requirements.txt once.
    """
    def _try_imports():
        try:
            import flask  # noqa: F401
            import tensorflow  # noqa: F401
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import reportlab  # noqa: F401
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False

    if _try_imports():
        print("✅ All required dependencies are installed")
        return True

    if not auto_install:
        print("Please install dependencies using: pip install -r requirements.txt")
        return False

    # Attempt a one-time install
    req_file = Path(__file__).with_name('requirements.txt')
    if not req_file.exists():
        print("⚠️ requirements.txt not found; cannot auto-install.")
        return False
    try:
        print("📦 Installing missing Python packages (one-time)...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(req_file)])
    except subprocess.CalledProcessError as e:
        print(f"❌ Auto-install failed: {e}")
        return False

    ok = _try_imports()
    if ok:
        print("✅ Dependencies installed successfully")
    return ok

def check_model_file():
    """Check if the model file exists"""
    model_path = "E:/new laptop/mega project/final_res_vs_nonres.h5"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        return True
    else:
        print(f"⚠️ Model file not found at: {model_path}")
        print("The system will create a compatible model architecture")
        return False

def main():
    """Main startup function"""
    print("🚀 Starting Therapy Prediction API Server...")
    print("=" * 50)
    
    # Check dependencies (auto-install only when missing)
    if not check_dependencies(auto_install=True):
        sys.exit(1)
    
    # Check model file
    check_model_file()
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    print("✅ Created necessary directories")
    
    # Start the Flask app
    print("\n🌐 Starting Flask server on http://localhost:5000")
    print("📊 API endpoints available:")
    print("   - GET  /api/health")
    print("   - POST /api/predict")
    print("   - GET  /api/download-report/<filename>")
    print("   - GET  /api/models/status")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
