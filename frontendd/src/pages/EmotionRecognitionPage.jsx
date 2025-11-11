import React from 'react';
import ModelLayout from '../components/ModelLayout';
import PatientForm from '../components/PatientForm';
import './ModelPage.css';

const EmotionRecognitionPage = ({ onModelSwitch }) => {
  const handlePatientSubmit = async (formData, formDataObj) => {
    try {
      // Here you would typically send the data to your backend API
      console.log('Submitting patient data for Emotion Recognition:', formDataObj);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Process the form data
      const patientInfo = {
        name: formDataObj.name,
        age: formDataObj.age,
        gender: formDataObj.gender,
        dateOfBirth: formDataObj.dateOfBirth,
        pastMedicalReports: formDataObj.pastMedicalReports,
        eegImage: formDataObj.eegImage
      };

      console.log('Patient data processed:', patientInfo);
      
      // You can add your ML model processing logic here
      // For now, we'll just show a success message
      return { success: true, message: 'Patient data processed successfully for Emotion Recognition analysis.' };
      
    } catch (error) {
      console.error('Error processing patient data:', error);
      throw new Error('Failed to process patient data. Please try again.');
    }
  };

  const goHome = () => { window.location.hash = '#/home'; };

  return (
    <ModelLayout currentModel="emotion-recognition" onModelSwitch={onModelSwitch}>
      <div className="model-page">
        <div className="page-header">
          <button className="page-close-btn" onClick={goHome}>×</button>
          <h1>📊 Emotion Recognition Using EEG</h1>
        </div>

        <div className="model-description">
          <div className="description-card">
            <h3>About This Model</h3>
            <p>
              Built for clinical use, this tool organizes patient signals into easy-to-read emotional 
              trends (e.g., calm, stressed, elevated) and highlights patterns you can review with the 
              patient to inform supportive care, counseling, and follow‑up planning.
            </p>
          </div>
        </div>

        <PatientForm 
          onSubmit={handlePatientSubmit}
          modelName="Emotion Recognition Using EEG"
        />
      </div>
    </ModelLayout>
  );
};

export default EmotionRecognitionPage;
