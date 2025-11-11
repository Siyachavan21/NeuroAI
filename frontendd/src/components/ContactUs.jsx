import React, { useState } from "react";
import "./ContactUs.css";

const ContactUs = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const [errors, setErrors] = useState({});
  const [success, setSuccess] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setErrors({ ...errors, [e.target.name]: "" });
    setSuccess("");
  };

  const handleClear = () => {
    setFormData({ name: "", email: "", message: "" });
    setErrors({});
    setSuccess("");
  };

  const validateForm = () => {
    const newErrors = {};
    const nameRegex = /^[A-Za-z\s]+$/;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!formData.name || !nameRegex.test(formData.name)) {
      newErrors.name = "Please enter a valid name (letters only).";
    }
    if (!formData.email || !emailRegex.test(formData.email)) {
      newErrors.email = "Please enter a valid email address.";
    }
    if (!formData.message || formData.message.length < 10) {
      newErrors.message = "Message must be at least 10 characters.";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    // Prepare data for FormSubmit
    const data = new FormData();
    data.append("name", formData.name);
    data.append("email", formData.email);
    data.append("message", formData.message);
    data.append("_captcha", "false");
    data.append("_subject", "New message from NeuroAI contact form");

    try {
      const response = await fetch(
        "https://formsubmit.co/ajax/neuroai36@gmail.com",
        {
          method: "POST",
          body: data,
        }
      );

      if (response.ok) {
        alert("Mail sent successfully!");
        setFormData({ name: "", email: "", message: "" });
      } else {
        alert("Failed to send mail. Please try again.");
      }
    } catch (error) {
      alert("Error sending mail. Please check your connection.");
      console.error(error);
    }
  };

  return (
    <section id="contact" className="contact-section">
      <div className="contact-container">
        <h2 className="contact-title">Get in Touch</h2>
        <p className="contact-subtitle">
          Whether you're a researcher, clinician, or curious mind — we'd love
          to hear from you.
        </p>

        <div className="contact-grid">
          <form className="contact-form card" onSubmit={handleSubmit} noValidate>
            <div className="field-row">
              <div className="field">
                <label htmlFor="name">Your Name</label>
                <input
                  id="name"
                  name="name"
                  type="text"
                  placeholder="Enter your name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
                {errors.name && <p className="error-text">{errors.name}</p>}
              </div>

              <div className="field">
                <label htmlFor="email">Your Email</label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="Enter your email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
                {errors.email && <p className="error-text">{errors.email}</p>}
              </div>
            </div>

            <label htmlFor="message">Your Message</label>
            <textarea
              id="message"
              name="message"
              rows="6"
              placeholder="Type your message..."
              value={formData.message}
              onChange={handleChange}
              required
            />
            {errors.message && <p className="error-text">{errors.message}</p>}

            <div className="form-actions">
              <button
                type="button"
                className="secondary-btn"
                onClick={handleClear}
              >
                Clear
              </button>
              <button type="submit">Send Message</button>
            </div>

            {success && <p className="success-text">{success}</p>}
          </form>

          <aside
            className="contact-info"
            aria-labelledby="contact-info-title"
          >
            <h3 id="contact-info-title" className="contact-info-title">
              Contact Information
            </h3>
            <p className="contact-info-line">Email: neuroai36@gmail.com</p>
          </aside>
        </div>
      </div>
    </section>
  );
};

export default ContactUs;
