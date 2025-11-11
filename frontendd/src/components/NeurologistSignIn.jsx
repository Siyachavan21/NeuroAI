import React, { useState, useEffect } from 'react';
import SignUp from './SignUp';
import { useAuth } from '../context/AuthContext.jsx';
import './NeurologistSignIn.css';

const STORAGE_PRIMARY_KEY = 'neuroai_user';
const STORAGE_LEGACY_KEY = 'neuroAIUser';

const NeurologistSignIn = ({ onSignIn, onClose }) => {
  const { isAuthenticated, signIn } = useAuth();

  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSignUp, setShowSignUp] = useState(false);

  // If already authenticated via context, close immediately
  useEffect(() => {
    if (isAuthenticated && onClose) onClose();
  }, [isAuthenticated, onClose]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    if (errors[name]) setErrors(prev => ({ ...prev, [name]: '' }));
  };

  const validateForm = () => {
    const next = {};
    if (!formData.email.trim()) next.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) next.email = 'Please enter a valid email address';
    if (!formData.password) next.password = 'Password is required';
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsSubmitting(true);
    try {
      // Read user record from either primary or legacy storage
      const raw = localStorage.getItem(STORAGE_PRIMARY_KEY) || localStorage.getItem(STORAGE_LEGACY_KEY);
      if (!raw) {
        setErrors({ general: 'No account found. Please sign up first.' });
        setIsSubmitting(false);
        return;
      }
      const userData = JSON.parse(raw);

      if (userData.email === formData.email && userData.password === formData.password) {
        const authedUser = { ...userData, isAuthenticated: true };
        // Persist to both keys for backward compatibility
        localStorage.setItem(STORAGE_PRIMARY_KEY, JSON.stringify(authedUser));
        localStorage.setItem(STORAGE_LEGACY_KEY, JSON.stringify(authedUser));
        // Update global auth context
        signIn(authedUser);
        if (onSignIn) onSignIn(authedUser);
        if (onClose) onClose();
      } else {
        setErrors({ general: 'Invalid email or password. Please try again.' });
      }
    } catch (err) {
      console.error('Sign in error:', err);
      setErrors({ general: 'An error occurred while signing in. Please try again.' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSignUp = (userData) => {
    // Persist to both keys and update global auth
    const authedUser = { ...userData, isAuthenticated: true };
    localStorage.setItem(STORAGE_PRIMARY_KEY, JSON.stringify(authedUser));
    localStorage.setItem(STORAGE_LEGACY_KEY, JSON.stringify(authedUser));
    signIn(authedUser);
    setShowSignUp(false);
    if (onClose) onClose();
  };

  if (showSignUp) {
    return (
      <SignUp 
        onSignUp={handleSignUp}
        onSwitchToSignIn={() => setShowSignUp(false)}
        onClose={() => setShowSignUp(false)}
      />
    );
  }

  return (
    <div className="neurologist-signin-overlay" onClick={onClose}>
      <div className="neurologist-signin-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Sign In</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        {errors.general && (
          <div className="error-message general-error">{errors.general}</div>
        )}

        <form onSubmit={handleSubmit} className="neurologist-form" noValidate>
          <div className="form-section">
            <h3>Credentials</h3>
            <div className="form-group">
              <label htmlFor="email">Email Address *</label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="Enter your email address"
                aria-invalid={Boolean(errors.email)}
                className={errors.email ? 'error' : ''}
              />
              {errors.email && <span className="error-text">{errors.email}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="password">Password *</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Enter your password"
                aria-invalid={Boolean(errors.password)}
                className={errors.password ? 'error' : ''}
              />
              {errors.password && <span className="error-text">{errors.password}</span>}
            </div>
          </div>

          <div className="form-actions">
            <button
              type="button"
              className="cancel-btn"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="submit-btn"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Signing In...' : 'Sign In'}
            </button>
          </div>
        </form>

        <div className="form-footer">
          <p>
            Don't have an account?{' '}
            <button
              type="button"
              className="switch-to-signup-btn"
              onClick={() => setShowSignUp(true)}
              disabled={isSubmitting}
            >
              Sign Up
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default NeurologistSignIn;
