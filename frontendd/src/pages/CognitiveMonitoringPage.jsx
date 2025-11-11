import React from 'react';
import ModelLayout from '../components/ModelLayout';
import PatientForm from '../components/PatientForm';
import './ModelPage.css';

const CognitiveMonitoringPage = ({ onModelSwitch }) => {
  const handlePatientSubmit = async (formData, formDataObj) => {
    try {
      // Here you would typically send the data to your backend API
      console.log('Submitting patient data for Cognitive Monitoring:', formDataObj);
      
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
      return { success: true, message: 'Patient data processed successfully for Cognitive Monitoring analysis.' };
      
    } catch (error) {
      console.error('Error processing patient data:', error);
      throw new Error('Failed to process patient data. Please try again.');
    }
  };

  const goHome = () => { window.location.hash = '#/home'; };

  return (
    <ModelLayout currentModel="cognitive-monitoring" onModelSwitch={onModelSwitch}>
      <div className="model-page">
        <div className="page-header">
          <button className="page-close-btn" onClick={goHome}>×</button>
          <h1>🔍 Cognitive Monitoring System</h1>
        </div>

        <div className="model-description">
          <div className="description-card">
            <h3>About This Model</h3>
            <p>
              Built for clinical workflows, this tool summarizes patient state into easy-to-read insights 
              (attention, relaxation, fatigue) and surfaces practical, wellness-oriented guidance you can 
              discuss with patients during routine visits or follow‑ups.
            </p>
          </div>
        </div>

        <PatientForm 
          onSubmit={handlePatientSubmit}
          modelName="Cognitive Monitoring System"
        />
      </div>
    </ModelLayout>
  );
};

export default CognitiveMonitoringPage;
