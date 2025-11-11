import React, { useState } from 'react';
import './SignUp.css';

const SPECIALIZATIONS = [
  'General Neurology',
  'Epilepsy',
  'Stroke/Neurovascular',
  'Movement Disorders',
  'Neurocritical Care',
  'Headache/Neurology',
  'Multiple Sclerosis',
  'Neuromuscular',
  'Pediatric Neurology',
  'Neuroimmunology',
  'Behavioral/Memory',
  'Other'
];

const SignUp = ({ onSignUp, onSwitchToSignIn, onClose }) => {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'neurologist',
    specialization: '',
    licenseNumber: '',
    attested: false
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [pendingUser, setPendingUser] = useState(null);

  // Existing account (login) handling
  const [existingAccount, setExistingAccount] = useState(false);
  const [existingForm, setExistingForm] = useState({ email: '', password: '' });
  const [existingErrors, setExistingErrors] = useState({});
  const [existingSubmitting, setExistingSubmitting] = useState(false);

  // password show/hide toggle
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [showExistingPassword, setShowExistingPassword] = useState(false);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const handleExistingInputChange = (e) => {
    const { name, value } = e.target;
    setExistingForm((prev) => ({ ...prev, [name]: value }));
    if (existingErrors[name]) {
      setExistingErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const handleExistingSubmit = async (e) => {
    e.preventDefault();
    const next = {};
    if (!existingForm.email.trim()) next.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(existingForm.email))
      next.email = 'Please enter a valid email address';
    if (!existingForm.password) next.password = 'Password is required';
    setExistingErrors(next);
    if (Object.keys(next).length) return;

    setExistingSubmitting(true);
    try {
      const raw =
        localStorage.getItem('neuroai_user') ||
        localStorage.getItem('neuroAIUser');
      if (!raw) {
        setExistingErrors({
          general: 'No account found. Please create an account first.',
        });
        setExistingSubmitting(false);
        return;
      }
      const stored = JSON.parse(raw);
      if (
        stored.email === existingForm.email &&
        stored.password === existingForm.password
      ) {
        const authedUser = { ...stored, isAuthenticated: true };
        localStorage.setItem('neuroai_user', JSON.stringify(authedUser));
        localStorage.setItem('neuroAIUser', JSON.stringify(authedUser));
        if (onSignUp) onSignUp(authedUser);
      } else {
        setExistingErrors({
          general: 'Invalid email or password. Please try again.',
        });
      }
    } catch (err) {
      console.error('Existing account sign-in error:', err);
      setExistingErrors({ general: 'An error occurred. Please try again.' });
    } finally {
      setExistingSubmitting(false);
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
    if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
    if (!formData.email.trim()) newErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email))
      newErrors.email = 'Please enter a valid email address';
    if (!formData.password) newErrors.password = 'Password is required';
    else if (formData.password.length < 8)
      newErrors.password = 'Password must be at least 8 characters long';
    if (!formData.confirmPassword)
      newErrors.confirmPassword = 'Please confirm your password';
    else if (formData.password !== formData.confirmPassword)
      newErrors.confirmPassword = 'Passwords do not match';
    if (!formData.specialization.trim())
      newErrors.specialization = 'Specialization is required';
    if (!formData.licenseNumber.trim())
      newErrors.licenseNumber = 'License number is required';
    if (!formData.attested)
      newErrors.attested = 'You must confirm you are a licensed neurologist';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    // Prevent duplicate account
    try {
      const rawExisting =
        localStorage.getItem('neuroai_user') ||
        localStorage.getItem('neuroAIUser');
      if (rawExisting) {
        const existing = JSON.parse(rawExisting);
        if (existing.email === formData.email) {
          setErrors({
            general:
              'An account with this email already exists. Please sign in instead.',
          });
          return;
        }
      }
    } catch (err) {
      // ignore
    }

    setIsSubmitting(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 800));

      const userData = {
        id: Date.now().toString(),
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        password: formData.password,
        role: 'neurologist',
        specialization: formData.specialization,
        licenseNumber: formData.licenseNumber,
        isAuthenticated: true,
      };

      localStorage.setItem('neuroai_user', JSON.stringify(userData));
      localStorage.setItem('neuroAIUser', JSON.stringify(userData));

      setPendingUser(userData);
      setSuccessMessage('Account created successfully! You can now access the module pages.');
    } catch (error) {
      console.error('Sign up error:', error);
      setErrors({
        general: 'An error occurred while creating your account. Please try again.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleContinueAfterSuccess = () => {
    if (pendingUser && onSignUp) {
      onSignUp(pendingUser);
    }
  };

  return (
    <div className="signup-modal-overlay" onClick={onClose}>
      <div className="signup-modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="close-signup-btn" onClick={onClose}>
          ×
        </button>

        <div className="signup-container">
          <div className="signup-header">
            <h2>Neurologist Registration</h2>
            <p>Only licensed neurologists are permitted to create an account</p>
          </div>

          {errors.general && (
            <div className="error-message general-error">{errors.general}</div>
          )}

          {successMessage ? (
            <div className="success-message" role="status" aria-live="polite">
              <p>{successMessage}</p>
              <button
                type="button"
                className="continue-btn"
                onClick={handleContinueAfterSuccess}
              >
                Continue
              </button>
            </div>
          ) : (
            <>
              <div style={{ marginBottom: 12 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={existingAccount}
                    onChange={() => {
                      setExistingAccount((prev) => !prev);
                      setExistingErrors({});
                    }}
                  />
                  <span>Existing account (Sign in with email & password)</span>
                </label>
              </div>

              {existingAccount ? (
                <form
                  onSubmit={handleExistingSubmit}
                  className="signup-form"
                  noValidate
                >
                  {existingErrors.general && (
                    <div className="error-message general-error">
                      {existingErrors.general}
                    </div>
                  )}
                  <div className="form-group">
                    <label htmlFor="existingEmail">Email Address *</label>
                    <input
                      type="email"
                      id="existingEmail"
                      name="email"
                      value={existingForm.email}
                      onChange={handleExistingInputChange}
                      placeholder="Enter email address"
                      className={existingErrors.email ? 'error' : ''}
                    />
                    {existingErrors.email && (
                      <span className="error-text">{existingErrors.email}</span>
                    )}
                  </div>

                  <div className="form-group">
                    <label htmlFor="existingPassword">Password *</label>
                    <div className="input-with-icon">
                      <input
                        type={showExistingPassword ? 'text' : 'password'}
                        id="existingPassword"
                        name="password"
                        value={existingForm.password}
                        onChange={handleExistingInputChange}
                        placeholder="Enter password"
                        className={existingErrors.password ? 'error' : ''}
                      />
                      <button
                        type="button"
                        className="icon-inside-input"
                        onClick={() => setShowExistingPassword((p) => !p)}
                        aria-label={showExistingPassword ? 'Hide password' : 'Show password'}
                      >
                        {showExistingPassword ? (
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path
                              d="M2 2l20 20M10.73 5.08A9.77 9.77 0 0 1 12 5c5 0 9.27 3.11 11 7.5a11.48 
                                  11.48 0 0 1-4.22 4.87M9.9 9.9a3 3 0 0 1 4.24 4.24M6.12 
                                  6.12A11.45 11.45 0 0 0 1 12a11.48 11.48 0 0 0 6.93 
                                  6.36"
                            />
                          </svg>
                        ) : (
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path
                              d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"
                            />
                            <circle cx="12" cy="12" r="3" />
                          </svg>
                        )}
                      </button>
                    </div>
                    <p className="forgot-note">
                      Forgot your password? <br />
                      <span
                        className="create-new-link"
                        onClick={() => setExistingAccount(false)}
                      >
                        Click here to create a new account.
                      </span>
                    </p>

                  </div>

                  <div className="form-actions">
                    <button
                      type="button"
                      className="cancel-btn"
                      onClick={onClose}
                      disabled={existingSubmitting}
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="signup-btn"
                      disabled={existingSubmitting}
                    >
                      {existingSubmitting ? 'Signing In...' : 'Sign In'}
                    </button>
                  </div>
                </form>
              ) : (
                <form onSubmit={handleSubmit} className="signup-form" noValidate>
                  <div className="form-row">
                    <div className="form-group">
                      <label htmlFor="firstName">First Name *</label>
                      <input
                        type="text"
                        id="firstName"
                        name="firstName"
                        value={formData.firstName}
                        onChange={handleInputChange}
                        placeholder="Enter first name"
                        className={errors.firstName ? 'error' : ''}
                      />
                      {errors.firstName && (
                        <span className="error-text">{errors.firstName}</span>
                      )}
                    </div>

                    <div className="form-group">
                      <label htmlFor="lastName">Last Name *</label>
                      <input
                        type="text"
                        id="lastName"
                        name="lastName"
                        value={formData.lastName}
                        onChange={handleInputChange}
                        placeholder="Enter last name"
                        className={errors.lastName ? 'error' : ''}
                      />
                      {errors.lastName && (
                        <span className="error-text">{errors.lastName}</span>
                      )}
                    </div>
                  </div>
                  <div className="form-group">
                    <label htmlFor="email">Email Address *</label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      placeholder="Enter email address"
                      className={errors.email ? 'error' : ''}
                    />
                    {errors.email && (
                      <span className="error-text">{errors.email}</span>
                    )}
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label htmlFor="password">Password *</label>
                      <div className="input-with-icon">
                        <input
                          type={showPassword ? 'text' : 'password'}
                          id="password"
                          name="password"
                          value={formData.password}
                          onChange={handleInputChange}
                          placeholder="Enter password"
                          className={errors.password ? 'error' : ''}
                        />
                        <button
                          type="button"
                          className="icon-inside-input"
                          onClick={() => setShowPassword((p) => !p)}
                          aria-label={showPassword ? 'Hide password' : 'Show password'}
                        >
                          {showPassword ? (
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                              <path
                                d="M2 2l20 20M10.73 5.08A9.77 9.77 0 0 1 12 5c5 0 9.27 3.11 11 7.5a11.48 
                                  11.48 0 0 1-4.22 4.87M9.9 9.9a3 3 0 0 1 4.24 4.24M6.12 
                                  6.12A11.45 11.45 0 0 0 1 12a11.48 11.48 0 0 0 6.93 
                                  6.36"
                              />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                              <path
                                d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"
                              />
                              <circle cx="12" cy="12" r="3" />
                            </svg>
                          )}
                        </button>
                      </div>
                      {errors.password && (
                        <span className="error-text">{errors.password}</span>
                      )}
                    </div>

                    <div className="form-group">
                      <label htmlFor="confirmPassword">Confirm Password *</label>
                      <div className="input-with-icon">
                        <input
                          type={showConfirmPassword ? 'text' : 'password'}
                          id="confirmPassword"
                          name="confirmPassword"
                          value={formData.confirmPassword}
                          onChange={handleInputChange}
                          placeholder="Confirm password"
                          className={errors.confirmPassword ? 'error' : ''}
                        />
                        <button
                          type="button"
                          className="icon-inside-input"
                          onClick={() => setShowConfirmPassword((p) => !p)}
                          aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                        >
                          {showConfirmPassword ? (
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                              <path
                                d="M2 2l20 20M10.73 5.08A9.77 9.77 0 0 1 12 5c5 0 9.27 3.11 11 7.5a11.48 
                                  11.48 0 0 1-4.22 4.87M9.9 9.9a3 3 0 0 1 4.24 4.24M6.12 
                                  6.12A11.45 11.45 0 0 0 1 12a11.48 11.48 0 0 0 6.93 
                                  6.36"
                              />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                              <path
                                d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"
                              />
                              <circle cx="12" cy="12" r="3" />
                            </svg>
                          )}
                        </button>
                      </div>
                      {errors.confirmPassword && (
                        <span className="error-text">
                          {errors.confirmPassword}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label>Role</label>
                      <input type="text" value="Neurologist" readOnly />
                    </div>

                    <div className="form-group">
                      <label htmlFor="specialization">Specialization *</label>
                      <select
                        id="specialization"
                        name="specialization"
                        value={formData.specialization}
                        onChange={handleInputChange}
                        className={errors.specialization ? 'error' : ''}
                      >
                        <option value="">Select specialization</option>
                        {SPECIALIZATIONS.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </select>
                      {errors.specialization && (
                        <span className="error-text">
                          {errors.specialization}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="form-group">
                    <label htmlFor="licenseNumber">License/ID Number *</label>
                    <input
                      type="text"
                      id="licenseNumber"
                      name="licenseNumber"
                      value={formData.licenseNumber}
                      onChange={handleInputChange}
                      placeholder="Enter license or ID number"
                      className={errors.licenseNumber ? 'error' : ''}
                    />
                    {errors.licenseNumber && (
                      <span className="error-text">
                        {errors.licenseNumber}
                      </span>
                    )}
                  </div>

                  <div className="form-group">
                    <label htmlFor="attested">
                      <input
                        type="checkbox"
                        id="attested"
                        name="attested"
                        checked={formData.attested}
                        onChange={handleInputChange}
                      />
                      &nbsp;I confirm that I am a licensed neurologist and authorized to
                      access this system *
                    </label>
                    {errors.attested && (
                      <span className="error-text">{errors.attested}</span>
                    )}
                  </div>

                  <div className="form-actions">
                    <button
                      type="submit"
                      className="signup-btn"
                      disabled={isSubmitting}
                    >
                      {isSubmitting ? 'Creating Account...' : 'Create Account'}
                    </button>
                  </div>
                </form>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SignUp;
