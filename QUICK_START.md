# Quick Start - Contact Form Email Setup

## What Was Done

✅ **Backend:** Added `/api/send-contact-email` endpoint in `backend/app.py`
✅ **Frontend:** Updated `ContactUs.jsx` to call the backend API
✅ **Email:** Configured to send to **neuroai36@gmail.com** (FREE)

## Setup in 3 Steps

### 1. Get Gmail App Password

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable **2-Step Verification** (if not already enabled)
3. Go to **App passwords**
4. Generate password for "Mail" → "Other (NeuroAI Contact Form)"
5. Copy the 16-character password

### 2. Set Environment Variables

**Windows PowerShell:**
```powershell
$env:GMAIL_USER = "neuroai36@gmail.com"
$env:GMAIL_APP_PASSWORD = "your-16-char-password"
```

**Windows CMD:**
```cmd
set GMAIL_USER=neuroai36@gmail.com
set GMAIL_APP_PASSWORD=your-16-char-password
```

### 3. Start Servers

```bash
# Backend
cd backend
python app.py

# Frontend (new terminal)
cd frontendd
npm start
```

## Test It

1. Open website → Contact form
2. Fill and click "Send"
3. Check **neuroai36@gmail.com** inbox

## Files Modified

- `backend/app.py` - Added email endpoint
- `frontendd/src/components/ContactUs.jsx` - Updated to use backend API

## Documentation

- Full setup: `backend/EMAIL_SETUP_GUIDE.md`
- No cost - Gmail SMTP is FREE (500 emails/day limit)
