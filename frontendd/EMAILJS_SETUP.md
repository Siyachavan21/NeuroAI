# EmailJS Setup Guide for Contact Form

This guide will help you set up EmailJS to send emails from the contact form to **neuroai36@gmail.com** for FREE (up to 200 emails/month).

## Step 1: Create an EmailJS Account

1. Go to [https://www.emailjs.com/](https://www.emailjs.com/)
2. Click **Sign Up** and create a free account
3. Verify your email address

## Step 2: Add Email Service

1. After logging in, go to **Email Services** in the dashboard
2. Click **Add New Service**
3. Choose **Gmail** (recommended for neuroai36@gmail.com)
4. Click **Connect Account** and sign in with **neuroai36@gmail.com**
5. Give your service a name (e.g., "NeuroAI Contact")
6. Click **Create Service**
7. **Copy the Service ID** (you'll need this later)

## Step 3: Create Email Template

1. Go to **Email Templates** in the dashboard
2. Click **Create New Template**
3. Use this template content:

**Subject:**
```
New Contact Form Message from {{from_name}}
```

**Body:**
```
You have received a new message from your website contact form.

Name: {{from_name}}
Email: {{from_email}}

Message:
{{message}}

---
This email was sent from your NeuroAI website contact form.
```

4. Click **Save**
5. **Copy the Template ID** (you'll need this later)

## Step 4: Get Your Public Key

1. Go to **Account** → **General** in the dashboard
2. Find your **Public Key** (it looks like: `xYz123AbC456DeF789`)
3. **Copy the Public Key**

## Step 5: Update the Code

Open `src/components/ContactUs.jsx` and replace these placeholders on lines 56-58:

```javascript
const serviceId = 'YOUR_SERVICE_ID'; // Replace with Service ID from Step 2
const templateId = 'YOUR_TEMPLATE_ID'; // Replace with Template ID from Step 3
const publicKey = 'YOUR_PUBLIC_KEY'; // Replace with Public Key from Step 4
```

Example:
```javascript
const serviceId = 'service_abc1234';
const templateId = 'template_xyz5678';
const publicKey = 'xYz123AbC456DeF789';
```

## Step 6: Test the Form

1. Start your React application
2. Fill out the contact form
3. Click **Send**
4. Check **neuroai36@gmail.com** inbox for the email

## Troubleshooting

### Email not received?
- Check spam/junk folder
- Verify all IDs are correct in the code
- Check EmailJS dashboard for delivery status
- Ensure you haven't exceeded the 200 emails/month limit

### Error in console?
- Make sure `@emailjs/browser` package is installed
- Verify your Public Key is correct
- Check that Gmail service is properly connected

## Free Tier Limits

- **200 emails per month** (free forever)
- No credit card required
- Upgrade available if you need more

## Security Note

The Public Key is safe to use in frontend code - it's designed for client-side use. Your Service ID and Template ID are also safe to expose.

---

**Need help?** Visit [EmailJS Documentation](https://www.emailjs.com/docs/)
