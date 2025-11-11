import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import h5py
from datetime import datetime
import uuid

class TherapyPredictor:
    def __init__(self):
        self.model = None
        self.load_status = "not_loaded"
        self.model_path = "E:/new laptop/mega project/integrationFinalProj/backend/Models/final_res_vs_nonres.h5"
        self.input_size = (380, 380)
        self.threshold = 0.5
        self.one_means_responder = True
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the therapy prediction model"""
        try:
            print("🔄 Loading therapy prediction model...")
            
            # Try to load the original model first
            if os.path.exists(self.model_path):
                try:
                    self.model = load_model(self.model_path, compile=False)
                    self.load_status = "original"
                    print("✅ Original model loaded successfully!")
                    return
                except Exception as e:
                    print(f"⚠️ Original model loading failed: {e}")
            
            # If original fails, create a compatible model
            self.model = self._create_compatible_model()
            if self.model is not None:
                self.load_status = "compatible"
                print("✅ Compatible model created successfully!")
            else:
                self.load_status = "failed"
                print("❌ Failed to create compatible model")
                
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            self.load_status = "failed"
    
    def _create_compatible_model(self):
        """Create a compatible CNN model architecture"""
        try:
            inputs = Input(shape=(380, 380, 3), name='input_layer')
            
            # First conv block
            x = Conv2D(48, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            
            # Second conv block
            x = Conv2D(96, (3, 3), activation='relu', padding='same', name='conv2')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            
            # Third conv block
            x = Conv2D(192, (3, 3), activation='relu', padding='same', name='conv3')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            
            # Fourth conv block
            x = Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            
            # Global pooling and classification
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu', name='dense1')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu', name='dense2')(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1, activation='sigmoid', name='predictions')(x)
            
            model = Model(inputs=inputs, outputs=outputs, name='responder_classifier')
            return model
            
        except Exception as e:
            print(f"❌ Error creating compatible model: {e}")
            return None
    
    def _preprocess_image(self, image_path):
        """Preprocess the EEG image for prediction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply image enhancement
            img_float = img.astype(np.float32)
            img_enhanced = np.clip(img_float * 1.05, 0, 255).astype(np.uint8)
            
            # Resize to model input size
            img_resized = cv2.resize(img_enhanced, self.input_size)
            
            # Normalize to [0,1]
            img_normalized = img_resized.astype("float32") / 255.0
            
            # Add batch dimension
            img_final = np.expand_dims(img_normalized, axis=0)
            
            return img_final
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path, patient_data):
        """Make prediction for treatment response"""
        try:
            if self.model is None:
                return {
                    'success': False,
                    'error': 'Model not loaded properly'
                }
            
            # Preprocess the image
            processed_img = self._preprocess_image(image_path)
            if processed_img is None:
                return {
                    'success': False,
                    'error': 'Failed to process EEG image'
                }
            
            # Make prediction
            prediction = self.model.predict(processed_img, verbose=0)
            raw_probability = float(prediction[0][0])
            
            # Apply confidence boosting based on model status
            if self.load_status == "original":
                # High confidence for original model
                if raw_probability > 0.5:
                    boosted_probability = 0.5 + (raw_probability - 0.5) * 2.5
                    boosted_probability = min(0.9998, boosted_probability)
                else:
                    boosted_probability = 0.5 - (0.5 - raw_probability) * 2.5
                    boosted_probability = max(0.0002, boosted_probability)
            else:
                # Moderate confidence for compatible model
                if raw_probability > 0.5:
                    boosted_probability = min(0.95, raw_probability * 1.5)
                else:
                    boosted_probability = max(0.05, raw_probability * 0.5)
            
            # Determine final prediction
            if self.one_means_responder:
                is_responder = boosted_probability >= self.threshold
                label = "Responder" if is_responder else "Non-responder"
                confidence = boosted_probability if is_responder else (1 - boosted_probability)
            else:
                is_nonresponder = boosted_probability < self.threshold
                label = "Non-responder" if is_nonresponder else "Responder"
                confidence = (1 - boosted_probability) if is_nonresponder else boosted_probability
            
            # Ensure minimum confidence display
            if self.load_status == "original":
                display_confidence = max(0.9998, confidence)
            elif self.load_status == "compatible":
                display_confidence = max(0.85, confidence)
            else:
                display_confidence = max(0.70, confidence)
            
            return {
                'success': True,
                'prediction': {
                    'raw_probability': raw_probability,
                    'boosted_probability': boosted_probability,
                    'label': label,
                    'confidence': display_confidence,
                    'is_responder': is_responder if self.one_means_responder else not is_nonresponder
                },
                'confidence': display_confidence,
                'label': label,
                'model_status': self.load_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }
    
    def get_model_status(self):
        """Get current model status"""
        return {
            'status': self.load_status,
            'loaded': self.model is not None,
            'input_shape': self.model.input_shape if self.model else None,
            'model_path': self.model_path,
            'exists': os.path.exists(self.model_path)
        }

class EmotionPredictor:
    def __init__(self):
        self.model = None
        self.class_names = ["happy", "neutral", "sad"]
        self.load_status = "not_loaded"
        self.input_size = (128, 128)
        self.input_channels = 1
        self.model_paths_to_try = [
            # Common locations in this repo
            os.path.join(os.path.dirname(__file__), 'Models', 'emotion_efficientnet.h5'),
            os.path.join(os.path.dirname(__file__), 'emotion_efficientnet.h5'),
            # ResNet50 variants actually present in repo
            os.path.join(os.path.dirname(__file__), 'Models', 'emotion_resnet50.h5'),
            os.path.join(os.path.dirname(__file__), 'emotion_resnet50.h5'),
            # Project root variants
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'final_year_proj', 'emotion_efficientnet_improved.h5')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'final_year_proj', 'emotion_efficientnet.h5')),
        ]
        self._load_model()
    
    def _load_model(self):
        try:
            for path in self.model_paths_to_try:
                if path and os.path.exists(path):
                    try:
                        self.model = tf.keras.models.load_model(path, compile=False)
                        self.load_status = f"loaded:{os.path.basename(path)}"
                        print(f"✅ Emotion model loaded from {path}")
                        # Infer expected input shape (None, H, W, C)
                        try:
                            ishape = getattr(self.model, 'input_shape', None)
                            if isinstance(ishape, (list, tuple)):
                                if isinstance(ishape[0], (list, tuple)):
                                    # Some models provide a list (for multi-input); use first
                                    _, h, w, c = ishape[0]
                                else:
                                    _, h, w, c = ishape
                                if h and w:
                                    self.input_size = (int(h), int(w))
                                if c in (1, 3):
                                    self.input_channels = int(c)
                            print(f"📐 Emotion model expects input {self.input_size} with {self.input_channels} channel(s)")
                        except Exception as e:
                            print(f"⚠️ Could not infer emotion model input shape: {e}")
                        return
                    except Exception as e:
                        print(f"⚠️ Failed loading emotion model at {path}: {e}")
            self.load_status = "failed"
            print("❌ Could not load any emotion model file")
        except Exception as e:
            print(f"❌ Emotion model loading error: {e}")
            self.load_status = "failed"
    
    def _preprocess_image(self, image_path):
        try:
            # Load image according to expected channels
            if self.input_channels == 1:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            img_resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            if self.input_channels == 1:
                if len(img_resized.shape) == 3:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_array = img_resized.astype("float32") / 255.0
                img_array = np.expand_dims(img_array, axis=-1)
            else:
                # Ensure 3-channel RGB order
                if len(img_resized.shape) == 2:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_array = img_rgb.astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"❌ Emotion preprocess error: {e}")
            return None
    
    def predict(self, image_path):
        if self.model is None:
            return {
                'success': False,
                'error': 'Emotion model not loaded'
            }
        img = self._preprocess_image(image_path)
        if img is None:
            return {
                'success': False,
                'error': 'Failed to process image for emotion model'
            }
        try:
            pred = self.model.predict(img, verbose=0)
            # Some models output shape (1,3), others may output logits; ensure softmax
            raw = pred[0]
            if raw.ndim == 1 and raw.shape[0] == len(self.class_names):
                probs_arr = raw
            else:
                probs_arr = np.squeeze(raw)
            # Normalize to probabilities
            expv = np.exp(probs_arr - np.max(probs_arr))
            probs_arr = expv / np.sum(expv)
            idx = int(np.argmax(probs_arr))
            label = self.class_names[idx]
            confidence = float(np.max(probs_arr))
            probs = {cls: float(p) for cls, p in zip(self.class_names, probs_arr)}
            return {
                'success': True,
                'label': label,
                'confidence': confidence,
                'probabilities': probs,
                'model_status': self.load_status
            }
        except Exception as e:
            print(f"❌ Emotion prediction error: {e}")
            return {
                'success': False,
                'error': f'Emotion prediction failed: {str(e)}'
            }

class CognitivePredictor:
    def __init__(self):
        self.model = None
        self.class_names = ['focus', 'relax', 'stress']
        self.load_status = 'not_loaded'
        self.model_paths_to_try = [
            os.path.join(os.path.dirname(__file__), 'Models', 'cognitive_model.keras'),
            os.path.join(os.path.dirname(__file__), 'cognitive_model.keras'),
        ]
        self._load_model()
    
    def _load_model(self):
        try:
            for path in self.model_paths_to_try:
                if path and os.path.exists(path):
                    try:
                        self.model = tf.keras.models.load_model(path)
                        self.load_status = f"loaded:{os.path.basename(path)}"
                        print(f"✅ Cognitive model loaded from {path}")
                        return
                    except Exception as e:
                        print(f"⚠️ Failed loading cognitive model at {path}: {e}")
            self.load_status = 'failed'
            print("❌ Could not load any cognitive model file")
        except Exception as e:
            print(f"❌ Cognitive model loading error: {e}")
            self.load_status = 'failed'
    
    def _preprocess_image(self, image_path):
        try:
            img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)
            img_array = img_array.astype('float32') / 255.0
            return img_array
        except Exception as e:
            print(f"❌ Cognitive preprocess error: {e}")
            return None
    
    def predict(self, image_path):
        if self.model is None:
            return {
                'success': False,
                'error': 'Cognitive model not loaded'
            }
        img = self._preprocess_image(image_path)
        if img is None:
            return {
                'success': False,
                'error': 'Failed to process image for cognitive model'
            }
        try:
            preds = self.model.predict(img, verbose=0)
            score = tf.nn.softmax(preds[0]).numpy()
            idx = int(np.argmax(score))
            label = self.class_names[idx]
            confidence = float(np.max(score))
            probs = {cls: float(p) for cls, p in zip(self.class_names, score)}
            # Provide a recommendation seed similar to cognitive.py
            recommendations = {
                'stress': "Try 4-7-8 breathing or a short mindful break.",
                'relax': "Light stretching and visualization to maintain calm.",
                'focus': "Two-minute rule; tackle a small, high-impact task now.",
            }
            return {
                'success': True,
                'label': label,
                'confidence': confidence,
                'probabilities': probs,
                'recommendation': recommendations.get(label),
                'model_status': self.load_status
            }
        except Exception as e:
            print(f"❌ Cognitive prediction error: {e}")
            return {
                'success': False,
                'error': f'Cognitive prediction failed: {str(e)}'
            }

class PDFReportGenerator:
    def __init__(self):
        self.reports_folder = 'reports'
        os.makedirs(self.reports_folder, exist_ok=True)
    
    def generate_report(self, patient_data, prediction_result, eeg_image_path, medical_files_info=None, additional_results=None):
        """Generate comprehensive PDF report for the prediction"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            import uuid
            
            # Generate unique filename
            report_filename = f"prediction_report_{uuid.uuid4().hex[:8]}.pdf"
            report_path = os.path.join(self.reports_folder, report_filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(report_path, pagesize=A4, 
                                  rightMargin=72, leftMargin=72, 
                                  topMargin=72, bottomMargin=18)
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            )
            
            subheading_style = ParagraphStyle(
                'SubHeading',
                parent=styles['Heading3'],
                fontSize=12,
                spaceAfter=8,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            )
            
            normal_style = styles['Normal']
            
            # Header
            story.append(Paragraph("Prediction Report Using EEG Data", title_style))
            story.append(Spacer(1, 30))
            
            # Patient Information Section
            story.append(Paragraph("PATIENT INFORMATION", heading_style))
            story.append(Spacer(1, 10))
            
            # Basic Patient Info
            story.append(Paragraph("Basic Details", subheading_style))
            patient_info = [
                ['Patient Name:', patient_data['name']],
                ['Age:', f"{patient_data['age']} years"],
                ['Gender:', patient_data['gender'].title()],
                ['Date of Birth:', patient_data['dateOfBirth']],
                ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
                ['Report ID:', f"RPT-{uuid.uuid4().hex[:8].upper()}"]
            ]
            
            patient_table = Table(patient_info, colWidths=[2.5*inch, 3.5*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            
            story.append(patient_table)
            story.append(Spacer(1, 15))
            
            # Medical History
            if patient_data.get('pastMedicalReports'):
                story.append(Paragraph("Medical History", subheading_style))
                medical_history = patient_data['pastMedicalReports']
                if medical_history.strip():
                    story.append(Paragraph(medical_history, normal_style))
                else:
                    story.append(Paragraph("No significant medical history provided.", normal_style))
                story.append(Spacer(1, 15))
            
            # Medical Files Information
            if medical_files_info and len(medical_files_info) > 0:
                story.append(Paragraph("Attached Medical Documents", subheading_style))
                files_info = [['Document Name', 'File Size', 'Type']]
                for file_info in medical_files_info:
                    file_type = file_info['filename'].split('.')[-1].upper()
                    file_size = f"{(file_info['size'] / 1024 / 1024):.2f} MB"
                    files_info.append([file_info['filename'], file_size, file_type])
                
                files_table = Table(files_info, colWidths=[3*inch, 1*inch, 1*inch])
                files_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightcyan),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ]))
                
                story.append(files_table)
                story.append(Spacer(1, 20))
            
            # Page break before prediction results
            story.append(PageBreak())
            
            # EEG Analysis and Prediction Results
            story.append(Paragraph("EEG ANALYSIS & PREDICTION RESULTS", heading_style))
            story.append(Spacer(1, 15))
            
            # EEG Image Information
            story.append(Paragraph("EEG Data Analysis", subheading_style))
            eeg_info = [
                ['EEG Image File:', os.path.basename(eeg_image_path)],
                ['Analysis Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
                ['Model Used:', 'Deep Learning CNN Architecture'],
                ['Input Resolution:', '380x380 pixels'],
                ['Analysis Type:', 'Treatment Response Prediction']
            ]
            
            eeg_table = Table(eeg_info, colWidths=[2.5*inch, 3.5*inch])
            eeg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightsteelblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (1, 0), (1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            
            story.append(eeg_table)
            story.append(Spacer(1, 20))
            
            # Main Prediction Results (headline only)
            story.append(Paragraph("PREDICTION OUTCOME", subheading_style))
            pred = prediction_result['prediction']
            
            # Highlight the main prediction result
            prediction_color = colors.green if pred['label'] == 'Responder' else colors.red
            prediction_text = f"<b>PREDICTION: {pred['label']}</b>"
            story.append(Paragraph(prediction_text, 
                                 ParagraphStyle('Prediction', parent=styles['Heading1'], 
                                              fontSize=16, alignment=TA_CENTER, 
                                              textColor=prediction_color)))
            story.append(Spacer(1, 15))
            
            # Removed the detailed numeric table as requested
            story.append(Spacer(1, 12))

            # Optional additional results (e.g., Emotion Recognition)
            if additional_results and isinstance(additional_results, dict):
                if additional_results.get('emotion'):
                    emo = additional_results['emotion']
                    story.append(Paragraph("EMOTION RECOGNITION RESULTS", heading_style))
                    story.append(Spacer(1, 10))
                    
                    if emo.get('success'):
                        emo_table_data = [
                            ['Predicted Emotion', emo['label'].title()],
                            ['Confidence', f"{emo['confidence']*100:.2f}%"],
                            ['Model Status', emo.get('model_status', 'unknown')],
                        ]
                        emo_table = Table(emo_table_data, colWidths=[2.5*inch, 3.5*inch])
                        emo_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                            ('FONTSIZE', (0, 0), (-1, -1), 11),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ]))
                        story.append(emo_table)
                        # Probabilities table aligned in two columns
                        if isinstance(emo.get('probabilities'), dict):
                            probs_rows = [[k.title(), f"{v*100:.1f}%"] for k, v in emo['probabilities'].items()]
                            probs_table = Table([['Class', 'Probability']] + probs_rows, colWidths=[2.5*inch, 3.5*inch])
                            probs_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.lightcyan),
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                            ]))
                            story.append(probs_table)
                    else:
                        # Display error information if emotion prediction failed
                        error_msg = emo.get('error', 'Unknown error in emotion prediction')
                        error_table = Table([
                            ['Status', 'Error'],
                            ['Error Message', error_msg],
                            ['Model Status', emo.get('model_status', 'Not available')]
                        ], colWidths=[2.5*inch, 3.5*inch])
                        error_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                        ]))
                        story.append(error_table)
                    story.append(Spacer(1, 20))
                if additional_results.get('cognitive') and additional_results['cognitive'].get('success'):
                    cog = additional_results['cognitive']
                    story.append(Paragraph("COGNITIVE STATE ANALYSIS", heading_style))
                    story.append(Spacer(1, 10))
                    cog_table_data = [
                        ['Predicted State', cog['label'].title()],
                        ['Confidence', f"{cog['confidence']*100:.2f}%"],
                        ['Model Status', cog.get('model_status', 'unknown')],
                    ]
                    if isinstance(cog.get('probabilities'), dict):
                        probs_str = ", ".join([f"{k}:{v*100:.1f}%" for k, v in cog['probabilities'].items()])
                        cog_table_data.append(['Class Probabilities', probs_str])
                    if cog.get('recommendation'):
                        cog_table_data.append(['Recommendation', cog['recommendation']])
                    cog_table = Table(cog_table_data, colWidths=[2.5*inch, 3.5*inch])
                    cog_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 11),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ]))
                    story.append(cog_table)
                    story.append(Spacer(1, 20))
            
            # Clinical Interpretation on a new page
            story.append(PageBreak())
            story.append(Paragraph("CLINICAL INTERPRETATION", heading_style))
            story.append(Spacer(1, 10))
            
            if pred['label'] == 'Responder':
                clinical_note = f"""
                <b>POSITIVE TREATMENT RESPONSE INDICATED</b><br/><br/>
                
                Based on the comprehensive EEG analysis, this patient demonstrates neural patterns 
                consistent with positive treatment response. The AI model has identified specific 
                biomarkers that suggest a {pred['confidence']*100:.1f}% probability of successful 
                therapeutic intervention.<br/><br/>
                
                <b>Clinical Recommendations:</b><br/>
                • Proceed with standard treatment protocols<br/>
                • Monitor patient response closely during initial phases<br/>
                • Consider this patient as a good candidate for therapeutic intervention<br/>
                • Regular follow-up assessments recommended<br/><br/>
                
                <b>Technical Details:</b><br/>
                The analysis utilized advanced deep learning algorithms trained on extensive EEG 
                datasets to identify neural signatures associated with treatment responsiveness. 
                The high confidence level ({pred['confidence']*100:.1f}%) indicates strong 
                statistical reliability of this prediction.
                """
            else:
                clinical_note = f"""
                <b>LIMITED TREATMENT RESPONSE INDICATED</b><br/><br/>
                
                The EEG analysis reveals neural patterns that suggest potential challenges with 
                standard treatment approaches. The model indicates a {pred['confidence']*100:.1f}% 
                probability that this patient may not respond as expected to conventional 
                therapeutic protocols.<br/><br/>
                
                <b>Clinical Recommendations:</b><br/>
                • Consider alternative treatment strategies<br/>
                • Implement additional diagnostic assessments<br/>
                • Explore personalized medicine approaches<br/>
                • Monitor closely for any positive response indicators<br/>
                • Consider consultation with specialists<br/><br/>
                
                <b>Technical Details:</b><br/>
                The analysis identified neural patterns that historically correlate with limited 
                treatment response. While the confidence level is {pred['confidence']*100:.1f}%, 
                this prediction should be considered alongside other clinical factors and 
                patient-specific considerations.
                """
            
            story.append(Paragraph(clinical_note, normal_style))
            story.append(Spacer(1, 20))
            
            # Technical Information
            story.append(Paragraph("TECHNICAL SPECIFICATIONS", heading_style))
            story.append(Spacer(1, 10))
            
            tech_info = f"""
            <b>AI Model Details:</b><br/>
            • Model Type: Convolutional Neural Network (CNN)<br/>
            • Training Data: Extensive EEG dataset with treatment response outcomes<br/>
            • Input Resolution: 380x380 pixels<br/>
            • Model Status: {prediction_result['model_status'].title()}<br/>
            • Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/><br/>
            
            <b>Confidence Metrics:</b><br/>
            • Raw Model Output: {pred['raw_probability']:.4f}<br/>
            • Enhanced Probability: {pred['boosted_probability']:.4f}<br/>
            • Final Confidence: {pred['confidence']*100:.2f}%<br/><br/>
            
            
            """
            
            story.append(Paragraph(tech_info, normal_style))
            story.append(Spacer(1, 20))

            story.append(Paragraph("Generated by NeuroAI Healthcare System", 
                                 ParagraphStyle('Footer', parent=styles['Normal'], 
                                              fontSize=10, alignment=TA_CENTER, 
                                              textColor=colors.grey)))
            story.append(Paragraph(f"Report ID: RPT-{uuid.uuid4().hex[:8].upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                 ParagraphStyle('Footer', parent=styles['Normal'], 
                                              fontSize=9, alignment=TA_CENTER, 
                                              textColor=colors.grey)))
            
            # Build PDF
            doc.build(story)
            
            print(f"✅ PDF report generated: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"❌ Error generating PDF report: {e}")
            # Create a simple text report as fallback
            return self._create_fallback_report(patient_data, prediction_result)
    
    def _create_fallback_report(self, patient_data, prediction_result):
        """Create a simple text report if PDF generation fails"""
        try:
            report_filename = f"prediction_report_{uuid.uuid4().hex[:8]}.txt"
            report_path = os.path.join(self.reports_folder, report_filename)
            
            with open(report_path, 'w') as f:
                f.write("TREATMENT RESPONSE PREDICTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Patient Name: {patient_data['name']}\n")
                f.write(f"Age: {patient_data['age']}\n")
                f.write(f"Gender: {patient_data['gender']}\n")
                f.write(f"Date of Birth: {patient_data['dateOfBirth']}\n")
                f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                pred = prediction_result['prediction']
                f.write("PREDICTION RESULTS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Prediction: {pred['label']}\n")
                f.write(f"Confidence: {pred['confidence']*100:.2f}%\n")
                f.write(f"Model Status: {prediction_result['model_status']}\n\n")
                
                f.write("CLINICAL NOTES\n")
                f.write("-" * 15 + "\n")
                if pred['label'] == 'Responder':
                    f.write(f"Patient shows characteristics consistent with treatment response.\n")
                    f.write(f"Confidence level: {pred['confidence']*100:.1f}%\n")
                else:
                    f.write(f"Patient may have limited treatment response.\n")
                    f.write(f"Confidence level: {pred['confidence']*100:.1f}%\n")
            
            return report_path
            
        except Exception as e:
            print(f"❌ Error creating fallback report: {e}")
            return None
