import React from 'react';
import ModelLayout from '../components/ModelLayout';
import PatientForm from '../components/PatientForm';
import './ModelPage.css';

const PredictiveTreatmentPage = ({ onModelSwitch }) => {
  const handlePatientSubmit = async (formData, formDataObj, predictionResult) => {
    try {
      console.log('Submitting patient data for Predictive Treatment:', formDataObj);
      
      if (predictionResult) {
        console.log('Prediction result received:', predictionResult);
        
        // Display detailed prediction information
        const predictionDetails = {
          label: predictionResult.label,
          confidence: predictionResult.confidence,
          modelStatus: predictionResult.model_status,
          timestamp: predictionResult.timestamp
        };
        
        console.log('Prediction details:', predictionDetails);
        
        return { 
          success: true, 
          message: `Prediction completed: ${predictionResult.label} (${(predictionResult.confidence * 100).toFixed(2)}% confidence)`,
          prediction: predictionDetails
        };
      }
      
      // Fallback if no prediction result
      return { success: true, message: 'Patient data processed successfully for Predictive Treatment Response analysis.' };
      
    } catch (error) {
      console.error('Error processing patient data:', error);
      throw new Error('Failed to process patient data. Please try again.');
    }
  };

  const goHome = () => { window.location.hash = '#/home'; };

  return (
    <ModelLayout currentModel="predictive-treatment" onModelSwitch={onModelSwitch}>
      <div className="model-page">
        <div className="page-header">
          <button className="page-close-btn" onClick={goHome}>×</button>
          <h1>🧠 Predictive Treatment Response</h1>
        </div>

        <div className="model-description">
          <div className="description-card">
            <h3>About This Model</h3>
            <p>
              Tailored for clinical practice, this tool summarizes available patient information into a
              clear indication of likely treatment response. It is designed to complement—not replace—
              clinical judgment and patient preferences during consultations.
            </p>
          </div>
        </div>

        <PatientForm 
          onSubmit={handlePatientSubmit}
          modelName="Predictive Treatment Response"
        />
      </div>
    </ModelLayout>
  );
};

export default PredictiveTreatmentPage;
