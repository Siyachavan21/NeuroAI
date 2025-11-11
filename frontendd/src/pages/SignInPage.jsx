import React from 'react';
import { useAuth } from '../context/AuthContext';
import SignUp from '../components/SignUp';

const SignInPage = () => {
  const { isAuthenticated, signIn } = useAuth();

  if (isAuthenticated) {
    if (window && window.location) {
      window.location.hash = '';
    }
    return null;
  }

  const handleSignedUp = (userData) => {
    signIn(userData);
    if (window && window.location) window.location.hash = '';
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f4f7ff' }}>
      <div style={{ width: '100%', maxWidth: 680 }}>
        <SignUp onSignUp={handleSignedUp} onClose={() => { if (window && window.location) window.location.hash = ''; }} />
      </div>
    </div>
  );
};

export default SignInPage;
