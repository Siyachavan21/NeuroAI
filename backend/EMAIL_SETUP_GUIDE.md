# Gmail SMTP Email Setup Guide (FREE)

This guide will help you configure Gmail to send emails from your contact form to **neuroai36@gmail.com** using Gmail's free SMTP service.

## Prerequisites
- Gmail account: **neuroai36@gmail.com**
- Access to the Gmail account settings

## Step 1: Enable 2-Factor Authentication

1. Go to your Google Account: [https://myaccount.google.com/](https://myaccount.google.com/)
2. Sign in with **neuroai36@gmail.com**
3. Navigate to **Security** in the left sidebar
4. Under "How you sign in to Google", click **2-Step Verification**
5. Follow the prompts to enable 2FA (required for App Passwords)

## Step 2: Generate Gmail App Password

1. After enabling 2FA, go back to **Security** settings
2. Under "How you sign in to Google", click **App passwords**
   - If you don't see this option, make sure 2FA is enabled
3. In the "Select app" dropdown, choose **Mail**
4. In the "Select device" dropdown, choose **Other (Custom name)**
5. Enter a name like: **NeuroAI Contact Form**
6. Click **Generate**
7. **Copy the 16-character password** (it will look like: `abcd efgh ijkl mnop`)
   - **IMPORTANT:** Save this password securely - you won't be able to see it again!

## Step 3: Set Environment Variables

### Option A: Windows (PowerShell) - Temporary
```powershell
$env:GMAIL_USER = "neuroai36@gmail.com"
$env:GMAIL_APP_PASSWORD = "your-16-char-app-password"
```

### Option B: Windows (Command Prompt) - Temporary
```cmd
set GMAIL_USER=neuroai36@gmail.com
set GMAIL_APP_PASSWORD=your-16-char-app-password
```

### Option C: Create .env file (Recommended for Development)

1. Create a file named `.env` in the `backend` folder
2. Add these lines (replace with your actual app password):

```env
GMAIL_USER=neuroai36@gmail.com
GMAIL_APP_PASSWORD=abcdefghijklmnop
```

3. Install python-dotenv:
```bash
pip install python-dotenv
```

4. Add this to the top of `app.py` (after imports):
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option D: Windows System Environment Variables (Permanent)

1. Press `Win + X` and select **System**
2. Click **Advanced system settings**
3. Click **Environment Variables**
4. Under "User variables", click **New**
5. Add two variables:
   - Variable name: `GMAIL_USER`
     Value: `neuroai36@gmail.com`
   - Variable name: `GMAIL_APP_PASSWORD`
     Value: `your-16-char-app-password`
6. Click **OK** and restart your terminal/IDE

## Step 4: Test the Setup

1. Make sure both backend and frontend are running:
   ```bash
   # Terminal 1 - Backend
   cd backend
   python app.py
   
   # Terminal 2 - Frontend
   cd frontendd
   npm start
   ```

2. Open the website in your browser
3. Fill out the contact form
4. Click **Send**
5. Check **neuroai36@gmail.com** inbox for the email

## Troubleshooting

### "Email service not configured" error
- Make sure environment variables are set correctly
- Restart your terminal/IDE after setting environment variables
- Verify the variable names are exactly: `GMAIL_USER` and `GMAIL_APP_PASSWORD`

### "Email authentication failed" error
- Double-check your App Password (16 characters, no spaces)
- Make sure 2FA is enabled on the Gmail account
- Try generating a new App Password
- Ensure you're using the App Password, NOT your regular Gmail password

### Email not received
- Check spam/junk folder
- Verify the backend console shows "✅ Contact email sent successfully"
- Check Gmail account settings to ensure SMTP is not blocked
- Try sending a test email from Gmail to verify the account is working

### "Less secure app access" message
- This is outdated - use App Passwords instead (requires 2FA)
- App Passwords are the secure, recommended method

## Security Notes

✅ **Safe Practices:**
- App Passwords are designed for this use case
- Store App Password in environment variables, never in code
- Add `.env` to `.gitignore` to avoid committing secrets
- App Passwords can be revoked anytime from Google Account settings

❌ **Never:**
- Hardcode passwords in source code
- Commit `.env` files to version control
- Share your App Password publicly
- Use your main Gmail password for SMTP

## Cost

- **100% FREE** - Gmail SMTP is free for reasonable usage
- Sending limits: ~500 emails per day (more than enough for a contact form)
- No credit card required

## Alternative: Using a Different Gmail Account

If you want to send FROM a different Gmail account but still receive at neuroai36@gmail.com:

1. Create App Password for the sender account
2. Update environment variables:
   ```env
   GMAIL_USER=sender@gmail.com
   GMAIL_APP_PASSWORD=sender-app-password
   ```
3. The recipient (`neuroai36@gmail.com`) is hardcoded in the backend

---

**Need help?** 
- [Google App Passwords Help](https://support.google.com/accounts/answer/185833)
- [Gmail SMTP Settings](https://support.google.com/mail/answer/7126229)
