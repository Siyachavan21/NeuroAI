import React from 'react';
import './ModelLayout.css';
import { useAuth } from '../context/AuthContext';

const ModelLayout = ({ children, currentModel, onModelSwitch }) => {
  const { user, isAuthenticated, signOut } = useAuth();

  const availableModels = [
    { id: 'predictive-treatment', name: 'Predictive Treatment Response', icon: '🧠' },
    { id: 'emotion-recognition', name: 'Emotion Recognition Using EEG', icon: '📊' },
    { id: 'cognitive-monitoring', name: 'Cognitive Monitoring System', icon: '🔍' }
  ];

  const handleModelSwitch = (modelId) => {
    if (onModelSwitch) {
      onModelSwitch(modelId);
    }
  };

  const handleSignOut = () => {
    signOut();
    window.location.hash = '#/signin';
  };

  if (!isAuthenticated) {
    if (window && window.location) {
      window.location.hash = '#/signin';
    }
    return null;
  }

  return (
    <div className="model-layout">
      <div className="model-sidebar">
        <div className="sidebar-header">
          <h2>NeuroAI Dashboard</h2>
        </div>
        
        {user && (
          <div className="user-section">
            <div className="user-avatar">
              <span>👨‍⚕️</span>
            </div>
            <div className="user-info">
              <h3>Dr. {user.lastName}</h3>
              <p>{user.email}</p>
              <p className="user-role">{user.role}</p>
              {user.specialization && (
                <p className="user-specialization">{user.specialization}</p>
              )}
            </div>
            <button className="signout-btn" onClick={handleSignOut}>
              Sign Out
            </button>
          </div>
        )}

        <div className="model-navigation">
          <h3>Available Models</h3>
          <div className="model-list">
            {availableModels.map((model) => (
              <button
                key={model.id}
                className={`model-nav-item ${currentModel === model.id ? 'active' : ''}`}
                onClick={() => handleModelSwitch(model.id)}
              >
                <span className="model-icon">{model.icon}</span>
                <span className="model-name">{model.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="current-model-info">
          <h3>Current Model</h3>
          {currentModel && (
            <div className="current-model">
              <span className="model-icon">
                {availableModels.find(m => m.id === currentModel)?.icon || '📋'}
              </span>
              <span className="model-name">
                {availableModels.find(m => m.id === currentModel)?.name || 'Unknown Model'}
              </span>
            </div>
          )}
        </div>
      </div>
      
      <div className="model-main-content">
        {children}
      </div>
    </div>
  );
};

export default ModelLayout;
