import React from 'react';
import './Navbar.css';
import logoImg from '../assets/brain-logo.png';

const Navbar = () => {
  const handleScrollToSection = (sectionId) => {
    // If not on home route, go to home first, then scroll
    if (window.location.hash !== '#/home' && window.location.hash !== '') {
      window.location.hash = '#/home';
      setTimeout(() => {
        const section = document.getElementById(sectionId);
        if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 400); // small delay to wait for home to render
    } else {
      // Already on home page, scroll directly
      const section = document.getElementById(sectionId);
      if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <header className="navbar">
      <div className="logo" onClick={() => (window.location.hash = '#/home')} style={{ cursor: 'pointer' }}>
        <img src={logoImg} alt="NeuroAI Logo" className="logo-img" />
        <span className="logo-text">NeuroAI</span>
      </div>

      <nav className="nav-links">
        <a href="#about" onClick={(e) => { e.preventDefault(); handleScrollToSection('about'); }}>About Us</a>
        <a href="#contact" onClick={(e) => { e.preventDefault(); handleScrollToSection('contact'); }}>Contact Us</a>
        <a href="#/exercise">Exercise</a>
      </nav>
    </header>
  );
};

export default Navbar;
