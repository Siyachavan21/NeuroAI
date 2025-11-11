import React from 'react';
import './SignInNotification.css';

const SignInNotification = ({ isVisible, onClose, onSignIn }) => {
  if (!isVisible) return null;

  return (
    <div className="signin-notification-overlay" onClick={onClose}>
      <div className="signin-notification" onClick={(e) => e.stopPropagation()}>
        <div className="notification-header">
          <h3>🔒 Access Restricted</h3>
          <button 
            className="close-notification-btn" 
            onClick={onClose}
            aria-label="Close notification"
          >
            ×
          </button>
        </div>
        
        <div className="notification-content">
          <div className="notification-icon">⚠️</div>
          <h4>Sign In Required</h4>
          <p>
            You need to sign in to access this feature. NeuroAI is restricted to qualified neurologists only.
          </p>
          <p>
            Please sign in with your medical credentials to continue.
          </p>
        </div>
        
        <div className="notification-actions">
          <button 
            type="button" 
            className="cancel-notification-btn" 
            onClick={onClose}
          >
            Cancel
          </button>
          <button 
            type="button" 
            className="signin-notification-btn"
            onClick={onSignIn}
          >
            Sign In
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignInNotification;
