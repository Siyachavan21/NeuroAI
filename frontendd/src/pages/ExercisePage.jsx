import React, { useState } from 'react';
import './ExercisePage.css';
import RelaxImg from '../assets/Relax.png';
import FocusImg from '../assets/Focus.png';
import StressetImg from '../assets/Stresset.png';

const sections = [
  { 
    id: 'relax', 
    title: 'Relax', 
    description: 'Unwind your mind with gentle activities designed to reduce anxiety and promote calmness.', 
    accent: '#FFFFFF', 
    image: RelaxImg,
    benefits: 'Ideal for managing sensory overload, reducing anxiety, and promoting emotional regulation for individuals with ASD.'
  },
  { 
    id: 'focus', 
    title: 'Focus', 
    description: 'Enhance attention and concentration with engaging focus games that support cognitive development.', 
    accent: '#FFFFFF', 
    image: FocusImg,
    benefits: 'Helps improve sustained attention, reduce distractibility, and enhance task completion skills.'
  },
  { 
    id: 'stressed', 
    title: 'Stressed', 
    description: 'Release tension and manage stress through interactive activities that support emotional well-being.', 
    accent: '#FFFFFF', 
    image: StressetImg,
    benefits: 'Supports stress management, emotional regulation, and provides healthy outlets for overwhelming feelings.'
  },
];

export default function ExercisePage() {
  const [activeId, setActiveId] = useState('relax');
  const [launching, setLaunching] = useState(false);

  const stateMap = {
    relax: 'relax',
    focus: 'focus',
    stressed: 'stress',
  };

  const startGame = async (sectionId, index) => {
    if (launching) return;
    setLaunching(true);
    
    try {
      const mapped = stateMap[sectionId];
      const res = await fetch('http://localhost:5000/api/start-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: mapped, index }),
      });
      
      const data = await res.json();
      
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to start game');
      }
      
      console.log('Game sequence started successfully');
      
    } catch (e) {
      console.error('Error starting game:', e);
      if (e.message.includes('already running')) {
        alert('A game is already running. Please complete it first.');
      } else {
        alert('Could not start the game. Please ensure the backend server is running on port 5000.');
      }
    } finally {
      setLaunching(false);
    }
  };

  const gameTitles = {
    relax: ['Drawing Canvas', 'Zen Box'],
    focus: ['Target Practice', 'Typing Speed Test'],
    stressed: ['Zen Box', 'Drawing Canvas'],
  };

  const goHome = () => {
    window.location.hash = '#/home';
  };

  return (
    <div className="exercise-container">
      <button className="home-button" onClick={goHome}>
        ← Back to Home
      </button>
      
      {/* Removed help image/info section */}

      <div className="exercise-header">
        <h1 className="exercise-title">Mindful Exercises for ASD Support</h1>
      </div>

      <div className="exercise-tabs" role="tablist" aria-label="Exercise categories">
        {sections.map((s) => (
          <button
            key={s.id}
            role="tab"
            aria-selected={activeId === s.id}
            aria-controls={`panel-${s.id}`}
            id={`tab-${s.id}`}
            className={`exercise-tab ${activeId === s.id ? 'active' : ''}`}
            style={{ '--accent': s.accent }}
            onClick={() => setActiveId(s.id)}
          >
            <span className="tab-glow" />
            {s.title}
          </button>
        ))}
      </div>

      <div className="panels">
        {sections.map((s) => (
          <section
            key={s.id}
            id={`panel-${s.id}`}
            role="tabpanel"
            aria-labelledby={`tab-${s.id}`}
            hidden={activeId !== s.id}
            className={`panel ${activeId === s.id ? 'show' : 'hide'}`}
            style={{ '--accent': s.accent }}
          >
            <header className="panel-header">
              <div className="panel-header-content">
                <div className="panel-image-wrapper">
                  <img src={s.image} alt={s.title} className="panel-section-image" />
                </div>
                <div className="panel-header-text">
                  <h2>{s.title}</h2>
                  <p>{s.description}</p>
                  {s.benefits && (
                    <div className="section-benefits">
                      <span className="benefits-text">{s.benefits}</span>
                    </div>
                  )}
                </div>
              </div>
            </header>

            <div className="games-grid">
              <div
                className="game-card"
                role="button"
                tabIndex={0}
                onClick={() => startGame(s.id, 0)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') startGame(s.id, 0); }}
                aria-label={`${s.title} game 1`}
                style={{ opacity: launching ? 0.6 : 1, cursor: launching ? 'not-allowed' : 'pointer' }}
              >
                <div className="card-accent" />
                <h3>{gameTitles[s.id][0]}</h3>
                <p>{launching ? 'Starting...' : 'Click to launch this game.'}</p>
                <div className="breath-visual" aria-hidden="true">
                  <div className="pulse" />
                </div>
              </div>
              <div
                className="game-card"
                role="button"
                tabIndex={0}
                onClick={() => startGame(s.id, 1)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') startGame(s.id, 1); }}
                aria-label={`${s.title} game 2`}
                style={{ opacity: launching ? 0.6 : 1, cursor: launching ? 'not-allowed' : 'pointer' }}
              >
                <div className="card-accent" />
                <h3>{gameTitles[s.id][1]}</h3>
                <p>{launching ? 'Starting...' : 'Click to launch this game.'}</p>
                <div className="focus-visual" aria-hidden="true">
                  <div className="track">
                    <div className="dot" />
                    <div className="marker" />
                  </div>
                </div>
              </div>
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
