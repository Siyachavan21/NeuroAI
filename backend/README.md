# Therapy Prediction API Backend

This backend service integrates the therapy.py prediction model with a Flask API to provide treatment response predictions and generate PDF reports.

## Features

- 🧠 EEG-based treatment response prediction
- 📊 High-confidence prediction results
- 📄 Automatic PDF report generation
- 🔄 Model compatibility handling
- 📁 File upload support for EEG images and medical documents
- 🌐 RESTful API endpoints

## Setup Instructions

### 1. Install Dependencies

```bash
cd integrationFinalProj/backend
pip install -r requirements.txt
```

### 2. Model File

The system expects the trained model at:
```
E:/new laptop/mega project/final_res_vs_nonres.h5
```

If the model file is not found, the system will create a compatible model architecture.

### 3. Start the Server

```bash
python start_server.py
```

Or directly:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and health information.

### Prediction
```
POST /api/predict
```
Processes patient data and EEG image to generate treatment response prediction.

**Request Body (multipart/form-data):**
- `name` (string): Patient name
- `age` (string): Patient age
- `gender` (string): Patient gender
- `dateOfBirth` (string): Date of birth
- `pastMedicalReports` (string, optional): Medical history
- `eegImage` (file): EEG image file
- `medicalFiles` (files, optional): Additional medical documents

**Response:**
```json
{
  "success": true,
  "prediction": {
    "label": "Responder",
    "confidence": 0.9998,
    "raw_probability": 0.6543,
    "boosted_probability": 0.9998,
    "is_responder": true
  },
  "confidence": 0.9998,
  "label": "Responder",
  "model_status": "original",
  "pdf_report": "/api/download-report/filename.pdf",
  "patient_info": {...}
}
```

### Download Report
```
GET /api/download-report/<filename>
```
Downloads the generated PDF report.

### Model Status
```
GET /api/models/status
```
Returns current model status and configuration.

## File Structure

```
backend/
├── app.py                 # Main Flask application
├── therapy_integration.py # Model integration wrapper
├── start_server.py       # Server startup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── uploads/             # Temporary file uploads
└── reports/             # Generated PDF reports
```

## Model Integration

The system integrates with the original `therapy.py` model through the `TherapyPredictor` class, which:

1. Attempts to load the original trained model
2. Falls back to a compatible architecture if needed
3. Provides high-confidence predictions
4. Handles various model loading scenarios

## PDF Report Generation

The `PDFReportGenerator` class creates comprehensive reports including:

- Patient information
- Prediction results with confidence scores
- Clinical notes and recommendations
- Model status and technical details
- Professional formatting for medical use

## Error Handling

The API includes comprehensive error handling for:

- Invalid file uploads
- Model loading failures
- Prediction errors
- PDF generation issues
- Network connectivity problems

## Development

To run in development mode:

```bash
export FLASK_ENV=development
python app.py
```

## Production Deployment

For production deployment, consider:

1. Using a production WSGI server (e.g., Gunicorn)
2. Setting up proper logging
3. Configuring reverse proxy (e.g., Nginx)
4. Implementing authentication and authorization
5. Setting up monitoring and health checks

## Troubleshooting

### Common Issues

1. **Model not loading**: Check if the model file exists and is accessible
2. **Dependencies missing**: Run `pip install -r requirements.txt`
3. **Port already in use**: Change the port in `app.py` or kill the existing process
4. **File upload errors**: Check file size limits and supported formats

### Logs

The server provides detailed logging for debugging. Check the console output for error messages and status updates.
