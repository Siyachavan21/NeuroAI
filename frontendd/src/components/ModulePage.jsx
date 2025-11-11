import React from 'react';
import { useAuth } from '../context/AuthContext';
import './ModulePage.css';

const ModulePage = () => {
  const { user, signOut } = useAuth();

  const handleSignOut = () => {
    signOut();
  };

  return (
    <div className="module-page">
      <header className="module-header">
        <div className="header-content">
          <h1>NeuroAI Dashboard</h1>
          <div className="user-info">
            <span>Welcome, Dr. {user?.lastName}</span>
            <button onClick={handleSignOut} className="signout-btn">
              Sign Out
            </button>
          </div>
        </div>
      </header>

      <main className="module-content">
        <div className="welcome-section">
          <h2>Welcome to Your Neurology Dashboard</h2>
          <p>Access your patient data, EEG readings, and AI-powered insights.</p>
        </div>

        <div className="modules-grid">
          <div className="module-card">
            <div className="module-icon">🧠</div>
            <h3>EEG Analysis</h3>
            <p>Review and analyze patient EEG data with AI assistance</p>
            <button className="module-btn">Access Module</button>
          </div>

          <div className="module-card">
            <div className="module-icon">📊</div>
            <h3>Patient Records</h3>
            <p>Manage patient information and medical history</p>
            <button className="module-btn">Access Module</button>
          </div>

          <div className="module-card">
            <div className="module-icon">🔍</div>
            <h3>AI Diagnostics</h3>
            <p>Get AI-powered diagnostic suggestions and insights</p>
            <button className="module-btn">Access Module</button>
          </div>

          <div className="module-card">
            <div className="module-icon">📈</div>
            <h3>Progress Tracking</h3>
            <p>Monitor patient progress and treatment outcomes</p>
            <button className="module-btn">Access Module</button>
          </div>

          <div className="module-card">
            <div className="module-icon">⚡</div>
            <h3>Real-time Monitoring</h3>
            <p>Live patient monitoring and alert systems</p>
            <button className="module-btn">Access Module</button>
          </div>

          <div className="module-card">
            <div className="module-icon">📋</div>
            <h3>Reports & Analytics</h3>
            <p>Generate comprehensive reports and analytics</p>
            <button className="module-btn">Access Module</button>
          </div>
        </div>

        <div className="quick-stats">
          <h3>Quick Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-number">24</span>
              <span className="stat-label">Active Patients</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">156</span>
              <span className="stat-label">Total Sessions</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">89%</span>
              <span className="stat-label">Accuracy Rate</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">12</span>
              <span className="stat-label">New Alerts</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ModulePage;
