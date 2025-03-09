:: Start the backend in a new window
start uvicorn main:app --reload --port 8000

:: Move into the 'src' folder, install dependencies, and start the frontend
cd src
npm install
npm start
