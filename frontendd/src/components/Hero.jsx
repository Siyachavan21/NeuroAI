import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import './Hero.css';
import NeurologistSignIn from './NeurologistSignIn';

const Hero = () => {
  const { isAuthenticated } = useAuth();
  const [isSignInOpen, setIsSignInOpen] = useState(false);

  useEffect(() => {
    if (!window || !window.location) return;
    const hash = window.location.hash;
    const isEmptyHash = hash === '' || hash === '#' || hash === '#/';
    if (isAuthenticated && isEmptyHash) {
      window.location.hash = '#/predictive';
    }
  }, [isAuthenticated]);
  
  if (isAuthenticated && (window?.location?.hash === '' || window?.location?.hash === '#/' || window?.location?.hash === '#')) return null;

  const handleSignInClick = () => {
    window.location.hash = '#/signin';
  };

  return (
    <section className="hero hero-bg">
      <div className="hero-content">
        <div className="hero-right">
          <h1>Welcome to NeuroAI</h1>
          <p>
            Empowering healthcare with emotion-aware intelligence. NeuroAI integrates predictive modeling, EEG-based emotion recognition, and cognitive monitoring to deliver real-time, personalized insights.
          </p>
          <div className="signin-wrapper">
            <button
              className="signin-btn"
              onClick={handleSignInClick}
              aria-label="Sign in to your account"
            >
              Sign In
            </button>
          </div>
        </div>
      </div>

      {isSignInOpen && (
        <NeurologistSignIn 
          onSignIn={() => setIsSignInOpen(false)} 
          onClose={() => setIsSignInOpen(false)} 
        />
      )}
    </section>
  );
};

export default Hero;