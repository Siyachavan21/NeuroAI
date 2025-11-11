@echo off
echo Starting NeuroAI Healthcare Application...
echo ========================================

echo.
echo Starting Backend Server (in new window)...
start "Backend Server" cmd /k "cd backend && ..\venv\Scripts\activate && python app.py"

echo.
echo Waiting for backend to start...
timeout /t 8 /nobreak > nul

echo.
echo Starting Frontend Server (in new window)...
start "Frontend Server" cmd /k "cd frontendd && npm start"

echo.
echo Both servers are starting in separate windows...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to close this window...
pause > nul
