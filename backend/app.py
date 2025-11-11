from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
from datetime import datetime
import json
from werkzeug.utils import secure_filename
from therapy_integration import TherapyPredictor, EmotionPredictor, CognitivePredictor, PDFReportGenerator
try:
    from dotenv import load_dotenv  # optional in production
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, skip silently; rely on OS env vars
    pass
import cv2
import numpy as np
import threading
import tkinter as tk
import random
from cognitive import GAME_LIBRARY
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
HEATMAPS_SUBDIR = 'heatmaps'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'doc', 'docx', 'txt', 'xls', 'xlsx'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, HEATMAPS_SUBDIR), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize predictors
therapy_predictor = TherapyPredictor()
emotion_predictor = EmotionPredictor()
cognitive_predictor = CognitivePredictor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Therapy Prediction API is running'})

@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploads(filename):
    # Serve files from the uploads directory (e.g., generated heatmaps)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path)

@app.route('/api/predict', methods=['POST'])
def predict_treatment_response():
    try:
        # Get form data
        patient_data = {
            'name': request.form.get('name'),
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'dateOfBirth': request.form.get('dateOfBirth'),
            'pastMedicalReports': request.form.get('pastMedicalReports', ''),
        }
        
        # Validate required fields
        required_fields = ['name', 'age', 'gender', 'dateOfBirth']
        for field in required_fields:
            if not patient_data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        # Handle EEG image file
        if 'eegImage' not in request.files:
            return jsonify({'error': 'EEG image is required'}), 400
        
        eeg_file = request.files['eegImage']
        if eeg_file.filename == '':
            return jsonify({'error': 'No EEG image selected'}), 400
        
        if not allowed_file(eeg_file.filename):
            return jsonify({'error': 'Invalid EEG image file type'}), 400
        
        # Save EEG image temporarily
        eeg_filename = secure_filename(f"eeg_{uuid.uuid4()}_{eeg_file.filename}")
        eeg_path = os.path.join(app.config['UPLOAD_FOLDER'], eeg_filename)
        eeg_file.save(eeg_path)
        
        # Handle medical files (optional)
        medical_files = []
        medical_files_info = []
        
        if 'medicalFiles' in request.files:
            medical_files_list = request.files.getlist('medicalFiles')
            for medical_file in medical_files_list:
                if medical_file.filename != '' and allowed_file(medical_file.filename):
                    medical_filename = secure_filename(f"medical_{uuid.uuid4()}_{medical_file.filename}")
                    medical_path = os.path.join(app.config['UPLOAD_FOLDER'], medical_filename)
                    medical_file.save(medical_path)
                    medical_files.append(medical_path)
                    medical_files_info.append({
                        'filename': medical_file.filename,
                        'path': medical_path,
                        'size': os.path.getsize(medical_path)
                    })
        
        # Determine selected models
        # Frontend may send as JSON string or comma-separated
        selected_models_raw = request.form.get('selectedModels', '')
        selected_models = []
        if selected_models_raw:
            try:
                selected_models = json.loads(selected_models_raw) if selected_models_raw.strip().startswith('[') else [m.strip() for m in selected_models_raw.split(',') if m.strip()]
            except Exception:
                selected_models = [m.strip() for m in selected_models_raw.split(',') if m.strip()]
        # Defaults to Therapy if none provided
        if not selected_models:
            selected_models = ['Predictive Treatment Response']

        results = {}
        # Run chosen models
        if any(m.lower().startswith('predictive') or m.lower() == 'therapy' for m in selected_models):
            therapy_result = therapy_predictor.predict(eeg_path, patient_data)
            if not therapy_result['success']:
                # Clean up uploaded files
                os.remove(eeg_path)
                for medical_file in medical_files:
                    os.remove(medical_file)
                return jsonify({'error': therapy_result['error']}), 500
            results['therapy'] = therapy_result

        if any('emotion' in m.lower() for m in selected_models):
            emotion_result = emotion_predictor.predict(eeg_path)
            results['emotion'] = emotion_result
        if any('cognitive' in m.lower() for m in selected_models):
            cognitive_result = cognitive_predictor.predict(eeg_path)
            results['cognitive'] = cognitive_result
        
        # Generate PDF report
        report_generator = PDFReportGenerator()
        pdf_path = report_generator.generate_report(
            patient_data=patient_data,
            prediction_result=results.get('therapy', {'prediction': {'label': 'N/A', 'confidence': 0, 'raw_probability': 0, 'boosted_probability': 0}, 'model_status': 'N/A'}),
            eeg_image_path=eeg_path,
            medical_files_info=medical_files_info,
            additional_results=results
        )
        
        # Clean up uploaded files
        os.remove(eeg_path)
        for medical_file in medical_files:
            os.remove(medical_file)
        
        # Return prediction result with PDF download link
        response_body = {
            'success': True,
            'pdf_report': f"/api/download-report/{os.path.basename(pdf_path)}",
            'patient_info': patient_data,
            'results': results,
        }
        # Preserve top-level fields for backward compatibility if therapy ran
        if 'therapy' in results and results['therapy'].get('success'):
            response_body.update({
                'prediction': results['therapy']['prediction'],
                'confidence': results['therapy']['confidence'],
                'label': results['therapy']['label'],
                'model_status': results['therapy']['model_status'],
            })
        return jsonify(response_body)
        
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/api/convert-heatmap', methods=['POST'])
def convert_heatmap():
    """
    Accepts an uploaded EEG image (eegImage), applies a heatmap colormap,
    saves it in uploads/heatmaps, and returns a URL that the frontend can render.
    Note: If you have a specific pipeline using eeg_dataset.h5, replace the
    OpenCV-based conversion below with your actual invocation and save the output
    into the same folder, returning the relative URL.
    """
    try:
        if 'eegImage' not in request.files:
            return jsonify({'error': 'EEG image is required'}), 400
        eeg_file = request.files['eegImage']
        if eeg_file.filename == '':
            return jsonify({'error': 'No EEG image selected'}), 400
        if not allowed_file(eeg_file.filename):
            return jsonify({'error': 'Invalid EEG image file type'}), 400

        # Save original upload temporarily
        original_filename = secure_filename(f"eeg_{uuid.uuid4()}_{eeg_file.filename}")
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        eeg_file.save(original_path)

        # Read image in grayscale, then apply heatmap
        img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Cleanup and fail if unreadable
            os.remove(original_path)
            return jsonify({'error': 'Failed to read EEG image'}), 400

        # Normalize for consistent visualization
        img_float = img.astype('float32')
        if img_float.max() > 0:
            img_norm = (img_float / img_float.max() * 255.0).astype('uint8')
        else:
            img_norm = img.astype('uint8')
        heatmap_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)

        # Write heatmap file
        heatmap_filename = secure_filename(f"heatmap_{uuid.uuid4()}.png")
        heatmap_path_rel = os.path.join(HEATMAPS_SUBDIR, heatmap_filename)
        heatmap_path_abs = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_path_rel)
        cv2.imwrite(heatmap_path_abs, heatmap_color)

        # We can optionally remove the original upload
        try:
            os.remove(original_path)
        except Exception:
            pass

        return jsonify({
            'success': True,
            'heatmap_url': f"/uploads/{heatmap_path_rel.replace('\\', '/')}"
        })
    except Exception as e:
        print(f"Error in convert-heatmap endpoint: {e}")
        return jsonify({'error': 'Heatmap conversion failed'}), 500

@app.route('/api/download-report/<filename>', methods=['GET'])
def download_report(filename):
    try:
        file_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=f"prediction_report_{filename}")
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': 'Error downloading report'}), 500

@app.route('/api/models/status', methods=['GET'])
def get_model_status():
    try:
        status = therapy_predictor.get_model_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': 'Error getting model status'}), 500

@app.route('/api/email/config-status', methods=['GET'])
def email_config_status():
    try:
        sender_email = os.environ.get('GMAIL_USER')
        sender_password_set = bool(os.environ.get('AI_36Neuro') or os.environ.get('GMAIL_APP_PASSWORD'))
        return jsonify({
            'gmail_user_set': bool(sender_email),
            'password_set': sender_password_set,
            'using_password_var': 'AI_36Neuro' if os.environ.get('AI_36Neuro') else ('GMAIL_APP_PASSWORD' if os.environ.get('GMAIL_APP_PASSWORD') else None)
        })
    except Exception as e:
        return jsonify({'error': 'Failed to read email config'}), 500

# Global variable to track active game thread
active_game_thread = None

@app.route('/api/start-game', methods=['POST'])
def start_game():
    try:
        data = request.get_json(force=True)
        state = (data.get('state') or '').lower()  # expected: 'focus' | 'relax' | 'stress'
        index = int(data.get('index', 0))  # 0 or 1
        if state not in GAME_LIBRARY:
            return jsonify({'error': 'Invalid state'}), 400
        games = GAME_LIBRARY[state]
        if index < 0 or index >= len(games):
            return jsonify({'error': 'Invalid game index'}), 400

        # Check if a game is already running
        global active_game_thread
        if active_game_thread and active_game_thread.is_alive():
            return jsonify({'error': 'A game is already running. Please wait for it to complete.'}), 409

        # Launch the complete sequence: game -> analysis window
        def run_game():
            try:
                root = tk.Tk(); root.withdraw()
                game_class = games[index]
                
                # Launch game and wait for it to complete
                instance = game_class(root)
                root.wait_window(instance.window)
                
                # After game completes, show analysis window
                from cognitive import GameAnalysisWindow
                analysis_window = GameAnalysisWindow(root, state, game_class, instance.final_score)
                root.wait_window(analysis_window.window)
                
                try:
                    root.destroy()
                except Exception:
                    pass
            except Exception as e:
                print(f"Game sequence error: {e}")
            finally:
                # Clear the active thread when complete sequence finishes
                global active_game_thread
                active_game_thread = None

        active_game_thread = threading.Thread(target=run_game, daemon=True)
        active_game_thread.start()
        return jsonify({'success': True})
    except Exception as e:
        print(f"/api/start-game error: {e}")
        return jsonify({'error': 'Failed to start game'}), 500

@app.route('/api/send-contact-email', methods=['POST'])
def send_contact_email():
    """
    Send contact form email to neuroai36@gmail.com
    Requires Gmail App Password to be set in environment variable
    """
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        message = data.get('message', '').strip()
        
        # Validate input
        if not name or not email or not message:
            return jsonify({'error': 'All fields are required'}), 400
        
        recipient_email = 'neuroai36@gmail.com'

        # Prefer Resend if configured
        resend_api_key = os.environ.get('RESEND_API_KEY')
        if resend_api_key:
            try:
                payload = {
                    'from': 'NeuroAI <no-reply@neuroai.app>',
                    'to': [recipient_email],
                    'subject': f'New Contact Form Message from {name}',
                    'html': f"""
                        <strong>Name:</strong> {name}<br>
                        <strong>Email:</strong> {email}<br><br>
                        <strong>Message:</strong><br>
                        {message.replace(chr(10), '<br>')}<br><br>
                        <em>Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em>
                    """
                }
                headers = {
                    'Authorization': f'Bearer {resend_api_key}',
                    'Content-Type': 'application/json'
                }
                r = requests.post('https://api.resend.com/emails', json=payload, headers=headers, timeout=15)
                if r.status_code in (200, 201):
                    print("✅ Contact email sent via Resend")
                    return jsonify({'success': True})
                else:
                    print(f"❌ Resend error: {r.status_code} - {r.text}")
                    return jsonify({'error': 'Email provider error (Resend).'}), 502
            except Exception as e:
                print(f"❌ Resend exception: {e}")
                return jsonify({'error': 'Email provider exception (Resend).'}), 502

        # Fallback to Gmail SMTP
        sender_email = os.environ.get('GMAIL_USER', 'neuroai36@gmail.com')
        sender_password = os.environ.get('AI_36Neuro', '') or os.environ.get('GMAIL_APP_PASSWORD', '')
        if not sender_password:
            print("/api/send-contact-email: Missing email password env (AI_36Neuro or GMAIL_APP_PASSWORD)")
            return jsonify({'error': 'Email service not configured. Set RESEND_API_KEY or AI_36Neuro.'}), 500

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'New Contact Form Message from {name}'
        msg['From'] = sender_email
        msg['To'] = recipient_email

        text_content = f"""
New Contact Form Submission

Name: {name}
Email: {email}

Message:
{message}

---
This email was sent from your NeuroAI website contact form.
Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        html_content = f"""
<html>
<body>
    <h3>New Contact Form Submission</h3>
    <p><strong>Name:</strong> {name}</p>
    <p><strong>Email:</strong> {email}</p>
    <p><strong>Message:</strong><br>{message.replace(chr(10), '<br>')}</p>
    <p style="color:#666;">Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("✅ Contact email sent via Gmail SMTP")
        return jsonify({'success': True})
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ SMTP Authentication failed: {e}")
        return jsonify({'error': 'Email authentication failed. Check AI_36Neuro app password and GMAIL_USER.'}), 500
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return jsonify({'error': 'Failed to send email. Please try again later.'}), 500

if __name__ == '__main__':
    print("🚀 Starting Therapy Prediction API Server...")
    print("📊 Therapy Model Status:", therapy_predictor.get_model_status())
    print("🎭 Emotion Model Status:", emotion_predictor.load_status)
    print("🧩 Cognitive Model Status:", cognitive_predictor.load_status)
    app.run(debug=True, host='0.0.0.0', port=5000)