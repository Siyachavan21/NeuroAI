// VerticalSlider.jsx
import React, { useState } from 'react';
import InfoCard from './InfoCard';
import ModelLayout from './ModelLayout';
import PatientForm from './PatientForm';
import './VerticalSlider.css';
import TreatmentImg from '../assets/Treatment.png';
import EmotionsImg from '../assets/Emotions.png';
import StressImg from '../assets/Stress.png';

const VerticalSlider = () => {
  const [selectedModel, setSelectedModel] = useState(null);

  const navigate = (route) => {
    window.location.hash = `#/${route}`;
  };

  const handlePredictiveTreatment = () => {
    // Navigate to predictive route
    navigate('predictive');
  };

  const handleEmotionRecognition = () => {
    navigate('emotion');
  };

  const handleCognitiveMonitoring = () => {
    navigate('cognitive');
  };

  const closeModel = () => {
    setSelectedModel(null);
  };

  const handleModelSwitch = (modelId) => {
    setSelectedModel(modelId);
  };

  const getModelContent = () => {
    if (!selectedModel) return null;

    const modelConfigs = {
      'predictive-treatment': {
        title: '🧠 Predictive Treatment Response',
        subtitle: 'Classify responders vs. non-responders using ML models with SHAP/LIME explainability',
        description: 'This predictive model analyzes patient EEG data and medical history to determine the likelihood of treatment response. It uses advanced machine learning algorithms including Random Forest, XGBoost, and Neural Networks to provide accurate predictions.',
        features: [
          'Treatment response classification (Responder/Non-responder)',
          'SHAP (SHapley Additive exPlanations) for model interpretability',
          'LIME (Local Interpretable Model-agnostic Explanations)',
          'Patient outcome predictions with confidence scores',
          'Treatment optimization recommendations'
        ]
      },
      'emotion-recognition': {
        title: '📊 Emotion Recognition Using EEG',
        subtitle: 'Real-time emotion classification using CNN/LSTM on EEG data with interactive feedback',
        description: 'This advanced emotion recognition system analyzes EEG signals to identify and classify human emotions in real-time. It combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to achieve high accuracy in emotion detection.',
        features: [
          'Real-time emotion classification (Happy, Sad, Angry, Fear, Surprise, Disgust)',
          'CNN-LSTM hybrid architecture for optimal feature extraction',
          'Interactive feedback system for emotion validation',
          'Patient emotional state monitoring and tracking',
          'Emotion intensity scoring and analysis',
          'Multi-modal emotion recognition capabilities'
        ]
      },
      'cognitive-monitoring': {
        title: '🔍 Cognitive Monitoring System',
        subtitle: 'EEG-based focus/relaxation detection with personalized cognitive health suggestions',
        description: 'This cognitive monitoring system analyzes EEG signals to assess cognitive states including focus, attention, relaxation, and mental fatigue. It provides personalized recommendations for cognitive health improvement and performance optimization.',
        features: [
          'Real-time focus and attention monitoring',
          'Relaxation state detection and analysis',
          'Mental fatigue assessment and alerts',
          'Personalized cognitive health suggestions',
          'Performance optimization recommendations',
          'Long-term cognitive health tracking'
        ]
      }
    };

    const config = modelConfigs[selectedModel];
    if (!config) return null;

    const handlePatientSubmit = async (formData, formDataObj) => {
      try {
        console.log(`Submitting patient data for ${config.title}:`, formDataObj);
        await new Promise(resolve => setTimeout(resolve, 2000));
        alert('Patient data submitted successfully!');
        return { success: true, message: `Patient data processed successfully for ${config.title} analysis.` };
      } catch (error) {
        console.error('Error processing patient data:', error);
        throw new Error('Failed to process patient data. Please try again.');
      }
    };

    return (
      <ModelLayout currentModel={selectedModel} onModelSwitch={handleModelSwitch}>
        <div className="model-page">
          <div className="page-header">
            <h1>{config.title}</h1>
            <p>{config.subtitle}</p>
          </div>

          <div className="model-description">
            <div className="description-card">
              <h3>About This Model</h3>
              <p>{config.description}</p>
              <div className="model-features">
                <h4>Key Features:</h4>
                <ul>
                  {config.features.map((feature, index) => (
                    <li key={index}>{feature}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <PatientForm 
            onSubmit={handlePatientSubmit}
            modelName={config.title}
          />
        </div>
      </ModelLayout>
    );
  };

  return (
    <>
      <section className="slider">
        <div className="card-row">
          <InfoCard
            title="🧠 Predictive Treatment Response"
            description="Classify responders vs. non-responders using ML models with SHAP/LIME explainability"
            features={[]}
            onGenerate={() => handlePredictiveTreatment()}
            backgroundImage={TreatmentImg}
          />

          <InfoCard
            title="📊 Emotion Recognition Using EEG"
            description="Real-time emotion classification using CNN/LSTM on EEG data with interactive feedback"
            features={[]}
            onGenerate={() => handleEmotionRecognition()}
            backgroundImage={EmotionsImg}
          />

          <InfoCard
            title="🔍 Cognitive Monitoring System"
            description="EEG-based focus/relaxation detection with personalized cognitive health suggestions"
            features={[]}
            onGenerate={() => handleCognitiveMonitoring()}
            backgroundImage={StressImg}
          />
        </div>
      </section>

      {/* Model Page Modal */}
      {selectedModel && (
        <div className="model-modal-overlay" onClick={closeModel}>
          <div className="model-modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-model-btn" onClick={closeModel}>
              ×
            </button>
            {getModelContent()}
          </div>
        </div>
      )}
    </>
  );
};

export default VerticalSlider;
