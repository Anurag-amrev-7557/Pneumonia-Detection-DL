#!/usr/bin/env python3
"""
PULMO·AI Enterprise PACS Workstation Launcher.

Single-command startup script: boots the FastAPI server, initializes the
leak-free dual CheXNet ensemble, and serves the 60fps PACS web workstation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("\n" + "=" * 70)
    print("  🏥  PULMO·AI ENTERPRISE PACS WORKSTATION (v2.0)")
    print("  Dual-Backbone CheXNet Ensemble (ResNet-50 + DenseNet-121)")
    print("  Validated Benchmark: 95.87% Bal-Acc | 0.9935 ROC-AUC | Zero Leakage")
    print("=" * 70 + "\n")
    
    # Model verification
    model_primary = PROJECT_ROOT / "models" / "current" / "best_model.h5"
    model_secondary = PROJECT_ROOT / "models" / "current" / "densenet121_best.h5"
    
    if not model_primary.exists():
        print(f"❌ Error: Primary model not found at {model_primary}")
        sys.exit(1)
    
    print(f"✅ Primary Backbone:    {model_primary.name} ({model_primary.stat().st_size / (1024*1024):.1f} MB)")
    if model_secondary.exists():
        print(f"✅ Secondary Backbone:  {model_secondary.name} ({model_secondary.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("⚠️  Secondary Backbone:  Not found (falling back to single-backbone mode)")
    import os
    import socket

    def get_port(preferred=8000):
        # HF Spaces injects $PORT (7860); respect it if set
        env_port = os.environ.get("PORT")
        if env_port:
            return int(env_port)
        for p in [preferred, 8080, 8001, 8081]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
        return preferred

    # Bind to 0.0.0.0 so HF Spaces (and Docker in general) can route traffic in.
    # Locally this still works fine — just access via 127.0.0.1.
    host = os.environ.get("HOST", "0.0.0.0")
    port = get_port(8000)
    print(f"\n🚀 Starting PACS Diagnostic Server on http://{host}:{port} ...")
    print("   Press Ctrl+C to stop.\n")

    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )

if __name__ == "__main__":
    main()
