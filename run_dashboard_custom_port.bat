@echo off
echo ========================================
echo CAMS GTS EV v2.0 Dashboard
echo Custom Port Configuration
echo ========================================
echo.

set /p PORT="Enter port number (default 8501): "
if "%PORT%"=="" set PORT=8501

echo.
echo Starting local server on http://localhost:%PORT%
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run streamlit_app.py --server.port %PORT% --server.headless true

pause
