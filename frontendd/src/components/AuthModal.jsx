import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext.jsx';

const AuthModal = () => {
  const { isModalOpen, closeAuthModal, signIn } = useAuth();
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  if (!isModalOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(email)) {
      setError('Please enter a valid email');
      return;
    }
    signIn(email);
  };

  return (
    <div className="auth-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <div className="auth-modal">
        <h3 id="auth-title">Sign In</h3>
        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          <label htmlFor="auth-email">Email</label>
          <input
            id="auth-email"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={Boolean(error)}
          />
          {error && <span className="error-text">{error}</span>}
          <div className="auth-actions">
            <button type="button" onClick={closeAuthModal}>Cancel</button>
            <button type="submit">Sign In</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AuthModal;


