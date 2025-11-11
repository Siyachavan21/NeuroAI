import React, { useState, useEffect } from 'react';
import './PatientForm.css';

const PatientForm = ({ onSubmit, modelName }) => {
  const [formData, setFormData] = useState({
    name: '',
    age: '',
    gender: '',
    dateOfBirth: '',
    pastMedicalReports: '',
    eegImage: null,
    medicalFiles: []
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedModels, setSelectedModels] = useState([]);
  const [showHeatmapPrompt, setShowHeatmapPrompt] = useState(false);
  const [isConvertingHeatmap, setIsConvertingHeatmap] = useState(false);
  const [heatmapPreviewUrl, setHeatmapPreviewUrl] = useState('');

  useEffect(() => {
    if (Array.isArray(modelName)) setSelectedModels(modelName);
    else if (typeof modelName === 'string' && modelName.trim().length > 0)
      setSelectedModels([modelName]);
  }, [modelName]);

  const toggleModel = (label) => {
    setSelectedModels((prev) =>
      prev.includes(label) ? prev.filter((m) => m !== label) : [...prev, label]
    );
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));

    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: ''
      }));
    }

    if (name === 'age' && value) {
      const currentYear = new Date().getFullYear();
      const birthYear = currentYear - parseInt(value, 10);

      const dob = new Date(birthYear, 0, 1); // January 1st of calculated year

      setFormData((prev) => ({
        ...prev,
        dateOfBirth: dob.toISOString().split('T')[0] // format as yyyy-mm-dd
      }));
    }
  };


  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setErrors((prev) => ({
          ...prev,
          eegImage: 'Please select an image file (JPG, PNG, GIF, etc.)'
        }));
        return;
      }

      if (file.size > 5 * 1024 * 1024) {
        setErrors((prev) => ({
          ...prev,
          eegImage: 'File size must be less than 5MB'
        }));
        return;
      }

      setFormData((prev) => ({
        ...prev,
        eegImage: file
      }));

      if (errors.eegImage) {
        setErrors((prev) => ({
          ...prev,
          eegImage: ''
        }));
      }

      setShowHeatmapPrompt(true);
    }
  };

  const handleCancelHeatmap = () => {
    setShowHeatmapPrompt(false);
  };

  const handleConvertToHeatmap = async () => {
    if (!formData.eegImage) {
      return;
    }

    setIsConvertingHeatmap(true);
    try {
      const fd = new FormData();
      fd.append('eegImage', formData.eegImage);

      const resp = await fetch('http://localhost:5000/api/convert-heatmap', {
        method: 'POST',
        body: fd
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || 'Heatmap conversion failed');
      }

      const data = await resp.json();

      if (data && data.heatmap_url) {
        setHeatmapPreviewUrl(`http://localhost:5000${data.heatmap_url}`);
      }

      alert('Heatmap generated successfully. You can continue to submit the form.');
    } catch (err) {
      alert(`Failed to convert to heatmap: ${err.message}`);
    } finally {
      setIsConvertingHeatmap(false);
      setShowHeatmapPrompt(false);
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Patient name is required';
    }

    if (!formData.age) {
      newErrors.age = 'Age is required';
    } else if (formData.age < 0 || formData.age > 150) {
      newErrors.age = 'Age must be between 0 and 150';
    }

    if (!formData.gender) {
      newErrors.gender = 'Gender is required';
    }

    if (!formData.dateOfBirth) {
      newErrors.dateOfBirth = 'Date of birth is required';
    }

    if (!formData.eegImage) {
      newErrors.eegImage = 'EEG image is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      const submitData = new FormData();
      Object.keys(formData).forEach((key) => {
        if (key === 'eegImage' && formData[key]) {
          submitData.append(key, formData[key]);
        } else if (key === 'medicalFiles' && formData[key]) {
          formData[key].forEach((file) => submitData.append('medicalFiles', file));
        } else {
          submitData.append(key, formData[key]);
        }
      });

      const modelsToSend =
        selectedModels.length > 0
          ? selectedModels
          : Array.isArray(modelName)
            ? modelName
            : [modelName];
      submitData.append('selectedModels', JSON.stringify(modelsToSend));

      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: submitData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process prediction');
      }

      const result = await response.json();

      if (result.success) {
        let message = 'Prediction completed successfully!\n\n';
        if (result.results) {
          if (result.results.therapy && result.results.therapy.success) {
            const r = result.results.therapy;
            message += `Treatment Response: ${r.label} (${(r.confidence * 100).toFixed(2)}%)\n`;
          }
          if (result.results.emotion && result.results.emotion.success) {
            const e = result.results.emotion;
            message += `Emotion: ${e.label} (${(e.confidence * 100).toFixed(2)}%)\n`;
          }
          if (result.results.cognitive && result.results.cognitive.success) {
            const c = result.results.cognitive;
            message += `Cognitive: ${c.label} (${(c.confidence * 100).toFixed(2)}%)\n`;
          }
        } else if (result.label) {
          message += `${result.label} (${(result.confidence * 100).toFixed(2)}%)\n`;
        }
        message += '\nPDF report is being generated...';
        alert(message);

        if (result.pdf_report) {
          const pdfUrl = `http://localhost:5000${result.pdf_report}`;
          const link = document.createElement('a');
          link.href = pdfUrl;
          link.download = `prediction_report_${formData.name.replace(/\s+/g, '_')}.pdf`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }

        if (onSubmit) {
          await onSubmit(submitData, formData, result);
        }

        // Reset form after successful submission
        setFormData({
          name: '',
          age: '',
          gender: '',
          dateOfBirth: '',
          pastMedicalReports: '',
          eegImage: null,
          medicalFiles: []
        });
        setHeatmapPreviewUrl('');
      } else {
        throw new Error(result.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('Form submission error:', error);
      setErrors({ general: `An error occurred: ${error.message}` });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setFormData({
      name: '',
      age: '',
      gender: '',
      dateOfBirth: '',
      pastMedicalReports: '',
      eegImage: null,
      medicalFiles: []
    });
    setErrors({});
    setHeatmapPreviewUrl(''); // 🟩 Added: also clears heatmap image preview
  };

  return (
    <div className="patient-form-container">
      <div className="form-header">
        <h2>Patient Information Form</h2>
        <p>Model: {Array.isArray(modelName) ? modelName.join(', ') : modelName}</p>
      </div>

      {errors.general && (
        <div className="error-message general-error">{errors.general}</div>
      )}

      <form onSubmit={handleSubmit} className="patient-form" noValidate>
        <div className="form-section">
          <h3>Select Models</h3>
          <div className="form-row">
            <label style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <span>
                <input
                  type="checkbox"
                  checked={selectedModels.includes('Predictive Treatment Response')}
                  onChange={() => toggleModel('Predictive Treatment Response')}
                />{' '}
                Predictive Treatment Response
              </span>
              <span>
                <input
                  type="checkbox"
                  checked={selectedModels.includes('Emotion Recognition Using EEG')}
                  onChange={() => toggleModel('Emotion Recognition Using EEG')}
                />{' '}
                Emotion Recognition Using EEG
              </span>
              <span>
                <input
                  type="checkbox"
                  checked={selectedModels.includes('Cognitive Monitoring System')}
                  onChange={() => toggleModel('Cognitive Monitoring System')}
                />{' '}
                Cognitive Monitoring System
              </span>
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3>Basic Information</h3>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="name">Full Name *</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="Enter patient's full name"
                className={errors.name ? 'error' : ''}
              />
              {errors.name && <span className="error-text">{errors.name}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="age">Age *</label>
              <input
                type="number"
                id="age"
                name="age"
                value={formData.age}
                onChange={handleInputChange}
                placeholder="Enter age"
                min="0"
                max="150"
                className={errors.age ? 'error' : ''}
              />
              {errors.age && <span className="error-text">{errors.age}</span>}
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="gender">Gender *</label>
              <select
                id="gender"
                name="gender"
                value={formData.gender}
                onChange={handleInputChange}
                className={errors.gender ? 'error' : ''}
              >
                <option value="">Select gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
                <option value="prefer-not-to-say">Prefer not to say</option>
              </select>
              {errors.gender && <span className="error-text">{errors.gender}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="dateOfBirth">Date of Birth *</label>
              <input
                type="date"
                id="dateOfBirth"
                name="dateOfBirth"
                value={formData.dateOfBirth}
                onChange={handleInputChange}
                className={errors.dateOfBirth ? 'error' : ''}
              />
              {errors.dateOfBirth && (
                <span className="error-text">{errors.dateOfBirth}</span>
              )}
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>EEG Data Upload</h3>

          <div className="form-group">
            <label htmlFor="eegImage">EEG Signal Image *</label>
            <div className="file-upload-area">
              <input
                type="file"
                id="eegImage"
                name="eegImage"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
              />
              <div className="upload-placeholder">
                <span className="upload-icon">📁</span>
                <p>Click to select EEG image file</p>
                <small>
                  Supported formats: JPG, PNG, GIF, BMP, TIFF (Max size: 5MB)
                </small>
              </div>
            </div>
            {errors.eegImage && (
              <span className="error-text">{errors.eegImage}</span>
            )}

            {formData.eegImage && (
              <div className="file-preview">
                <p>Selected file: {formData.eegImage.name}</p>
                <p>Size: {(formData.eegImage.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            )}

            {heatmapPreviewUrl && (
              <div className="file-preview" style={{ marginTop: 12 }}>
                <p>Heatmap preview:</p>
                <img
                  src={heatmapPreviewUrl}
                  alt="EEG Heatmap"
                  style={{ maxWidth: '100%', borderRadius: 6 }}
                />
              </div>
            )}
          </div>
        </div>

        {showHeatmapPrompt && (
          <div
            className="modal-overlay"
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000
            }}
          >
            <div
              className="modal-content"
              style={{
                background: '#fff',
                padding: 20,
                borderRadius: 8,
                width: 'min(420px, 92%)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.2)'
              }}
            >
              <h3 style={{ marginTop: 0 }}>Convert to heatmap?</h3>
              <p>
                Do you want to convert the selected EEG image to a heatmap using the
                model?
              </p>
              <div
                style={{
                  display: 'flex',
                  gap: 12,
                  justifyContent: 'flex-end',
                  marginTop: 16
                }}
              >
                <button
                  type="button"
                  onClick={handleCancelHeatmap}
                  className="reset-btn"
                  disabled={isConvertingHeatmap}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConvertToHeatmap}
                  className="submit-btn"
                  disabled={isConvertingHeatmap}
                >
                  {isConvertingHeatmap ? 'Converting...' : 'Convert to heatmap'}
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="form-actions">
          <button
            type="button"
            onClick={handleReset}
            className="reset-btn"
            disabled={isSubmitting}
          >
            Reset Form
          </button>
          <button
            type="submit"
            className="submit-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Submitting...' : 'Submit Patient Data'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PatientForm;
