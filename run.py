#!/usr/bin/env python3
"""
TerraBLOOM RAG System Runner
Simple script to start the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    print("🌱 Starting TerraBLOOM RAG System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the terrabloom project directory")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Start the Streamlit app
    print("🚀 Launching Streamlit application...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Try running manually: streamlit run app.py")

if __name__ == "__main__":
    main()