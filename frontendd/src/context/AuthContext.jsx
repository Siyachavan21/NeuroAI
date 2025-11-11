import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('neuroai_user');
      if (saved) setUser(JSON.parse(saved));
    } catch (_) {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      if (user) localStorage.setItem('neuroai_user', JSON.stringify(user));
      else localStorage.removeItem('neuroai_user');
    } catch (_) {
      // ignore
    }
  }, [user]);

  const signIn = useCallback((userData) => {
    setUser(userData);
    setIsModalOpen(false);
  }, []);

  const signOut = useCallback(() => {
    setUser(null);
  }, []);

  const openAuthModal = useCallback(() => setIsModalOpen(true), []);
  const closeAuthModal = useCallback(() => setIsModalOpen(false), []);

  const value = useMemo(() => ({
    user,
    isAuthenticated: Boolean(user),
    signIn,
    signOut,
    isModalOpen,
    openAuthModal,
    closeAuthModal,
  }), [user, signIn, signOut, isModalOpen, openAuthModal, closeAuthModal]);

  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
};


