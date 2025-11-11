@echo off
echo Starting NeuroAI Healthcare Application...
echo ========================================

echo.
echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && python start_server.py"

echo.
echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Frontend Application...
start "Frontend App" cmd /k "cd frontendd && npm start"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to close this window...
pause > nul
