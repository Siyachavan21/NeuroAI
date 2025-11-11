import React from 'react';
import { useAuth } from '../context/AuthContext';
import './InfoCard.css';

const InfoCard = ({ title, description, features, onGenerate, backgroundImage }) => {
  const { isAuthenticated } = useAuth();

  const handleGenerate = () => {
    if (onGenerate) {
      onGenerate();
    }
  };

  const cardStyle = backgroundImage 
    ? { 
        backgroundImage: `url("${backgroundImage}")`,
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        backgroundSize: 'cover'
      }
    : {};

  return (
    <>
      <div className="info-card" style={cardStyle}>
        <div className="card-header">
          <h3>{title}</h3>
        </div>
        <div className="card-body">
          <p>{description}</p>
        </div>
        <div className="card-actions">
          <button 
            className="generate-btn"
            onClick={handleGenerate}
          >
            {isAuthenticated ? 'Access Module' : 'Login to Generate'}
          </button>
        </div>
      </div>
    </>
  );
};

export default InfoCard;
