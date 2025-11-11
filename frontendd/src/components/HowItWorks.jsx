import React from 'react';
import './HowItWorks.css';

const STEPS = [
  { title: '1. EEG Input', text: 'Collect non-invasive EEG signals from supported devices.' },
  { title: '2. Processing', text: 'Clean and transform signals into features in real-time.' },
  { title: '3. Inference', text: 'Run deep models (CNN/LSTM) to infer emotion and cognitive state.' },
  { title: '4. Guidance', text: 'Provide recommendations: relaxation, focus games, or mindfulness.' },
];

const HowItWorks = () => {
  return (
    <section className="how-section" id="how-it-works">
      <div className="how-container">
        <h2 className="how-title">How It Works</h2>
        <div className="how-grid">
          {STEPS.map((s, i) => (
            <div key={i} className="how-card">
              <h3 className="how-step">{s.title}</h3>
              <p className="how-text">{s.text}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;


