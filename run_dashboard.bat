@echo off
echo ========================================
echo CAMS GTS EV v2.0 Dashboard
echo ========================================
echo.
echo Starting local server on http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run streamlit_app.py --server.port 8501 --server.headless true

pause
