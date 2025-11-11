import React from 'react';
import './AboutUs.css';
import About1 from '../assets/About1.png';
import About2 from '../assets/About2.png';

const AboutUs = () => (
  <section id="about" className="about-section">
    <div className="about-container">
      <h2 className="about-title">About NeuroAI</h2>

      <div className="image-row">
        <div className="hover-card">
          <img
            src={About1}
            alt="Neuroscience Legacy"
            className="hover-img"
            loading="lazy"
          />
          <div className="hover-overlay">
            <div className="hover-scroll">
              <div className="hover-text">
                <h3>Neuroscience Legacy</h3>
                <p>
                  NeuroAI's foundation is rooted in the pioneering work of Hans Berger, who first recorded electrical activity from the human brain in 1924. His invention of EEG opened a gateway to understanding neural rhythms, emotional states, and cognitive patterns.
                </p>
                <p>
                  NeuroAI honors this legacy by transforming raw EEG signals into meaningful visualizations. We don't just display data — we narrate the brain's story, tracing emotional arcs and cognitive shifts with clarity and reverence for the science that came before us.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="hover-card">
          <img
            src={About2}
            alt="NeuroAI Today"
            className="hover-img small"
            loading="lazy"
          />
          <div className="hover-overlay">
            <div className="hover-text no-scroll">
              <h3>NeuroAI Today</h3>
              <p>
                Today, NeuroAI is a full-stack intelligence dashboard that blends predictive modeling, emotion recognition, and cognitive monitoring into a cinematic interface. Built with React.js, Docker, and cloud-native architecture, it's designed for scalability, precision, and visual polish.
              </p>
              <p>
                But beyond the tech, our mission is human: to help people grow better. We empower clinicians and researchers with tools that are not only smart, but emotionally aware — bridging the gap between data and empathy, science and storytelling.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

export default AboutUs;