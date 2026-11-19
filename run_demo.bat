@echo off
cd /d "%~dp0"
echo Starting the demo. A browser tab opens at http://localhost:8501
echo Close this window to stop the demo.
"venv\Scripts\python.exe" -m streamlit run demo_app.py --server.port 8501 --browser.gatherUsageStats false
pause
