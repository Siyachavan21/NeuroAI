import React, { useEffect, useState } from 'react';
import { AuthProvider } from './context/AuthContext';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import VerticalSlider from './components/VerticalSlider';
import AboutUs from './components/AboutUs';
import ContactUs from './components/ContactUs';
import SignInPage from './pages/SignInPage.jsx';
import PredictiveTreatmentPage from './pages/PredictiveTreatmentPage.jsx';
import EmotionRecognitionPage from './pages/EmotionRecognitionPage.jsx';
import CognitiveMonitoringPage from './pages/CognitiveMonitoringPage.jsx';
import './App.css';
import ExercisePage from './pages/ExercisePage.jsx';

function App() {
  const [hash, setHash] = useState(window.location.hash);

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash);
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const route = (hash || '').replace('#/', '');

  const navigateTo = (route) => {
    window.location.hash = `#/${route}`;
  };

  const handleModelSwitch = (modelId) => {
    if (modelId === 'predictive-treatment') navigateTo('predictive');
    else if (modelId === 'emotion-recognition') navigateTo('emotion');
    else if (modelId === 'cognitive-monitoring') navigateTo('cognitive');
  };

  let content = (
    <>
      <Hero />
      <VerticalSlider />
      <AboutUs />
      <ContactUs />
    </>
  );

  if (route === 'signin') {
    content = <SignInPage />;
  } else if (route === 'predictive') {
    content = <PredictiveTreatmentPage onModelSwitch={handleModelSwitch} />;
  } else if (route === 'emotion') {
    content = <EmotionRecognitionPage onModelSwitch={handleModelSwitch} />;
  } else if (route === 'cognitive') {
    content = <CognitiveMonitoringPage onModelSwitch={handleModelSwitch} />;
  } else if (route === 'exercise') {
    content = <ExercisePage />;
  } else if (route === 'home') {
    content = (
      <>
        <Hero />
        <VerticalSlider />
        <AboutUs />
        <ContactUs />
      </>
    );
  }

  return (
    <AuthProvider>
      <div className="app-container">
        <Navbar />
        {content}
        <footer className="app-footer">
          © 2025 NeuroAI. All rights reserved.
        </footer>
      </div>
    </AuthProvider>
  );
}

export default App;
