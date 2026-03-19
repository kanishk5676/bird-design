"""
Dashboard Launcher - Start Streamlit Applications
==================================================
Convenient script to launch both dashboards simultaneously
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("="*70)
    print("BIRD-INSPIRED AIRFOIL SYSTEM - DASHBOARD LAUNCHER")
    print("="*70)
    
    streamlit_dir = Path(__file__).parent / "streamlit_apps"
    
    print("\nStarting dashboards...")
    print("-" * 70)
    
    # Launch main dashboard on port 8501
    print("\n1. Main Bird Analysis Dashboard")
    print("   URL: http://localhost:8501")
    dashboard1 = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        str(streamlit_dir / "dashboard_example.py"),
        "--server.port", "8501"
    ])
    
    time.sleep(2)
    
    # Launch airfoil visualizer on port 8502
    print("\n2. Airfoil Visualizer")
    print("   URL: http://localhost:8502")
    dashboard2 = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        str(streamlit_dir / "airfoil_visualizer.py"),
        "--server.port", "8502"
    ])
    
    print("\n" + "="*70)
    print("✅ BOTH DASHBOARDS RUNNING")
    print("="*70)
    print("\nMain Dashboard:     http://localhost:8501")
    print("Airfoil Visualizer: http://localhost:8502")
    print("\nPress Ctrl+C to stop both dashboards")
    print("="*70)
    
    try:
        dashboard1.wait()
        dashboard2.wait()
    except KeyboardInterrupt:
        print("\n\nStopping dashboards...")
        dashboard1.terminate()
        dashboard2.terminate()
        dashboard1.wait()
        dashboard2.wait()
        print("✅ Dashboards stopped")

if __name__ == "__main__":
    main()
