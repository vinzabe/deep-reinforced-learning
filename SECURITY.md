# Security Policy

## 🔒 Protecting Your API Keys and Credentials

This project requires sensitive credentials (API keys, account IDs, tokens). Follow these guidelines to keep them secure.

## ✅ Best Practices

### 1. Never Commit Secrets to Git

**DO:**
- ✅ Use `.env` file for local development
- ✅ Use environment variables in production
- ✅ Keep `.env` in `.gitignore` (already configured)
- ✅ Use `.env.example` as a template (safe to commit)

**DON'T:**
- ❌ Hardcode API keys in Python files
- ❌ Commit `.env` file to git
- ❌ Share API keys in screenshots or logs
- ❌ Use production keys in public repositories

### 2. Environment Variables Setup

```bash
# Copy the example file
cp .env.example .env

# Edit with your real credentials (never commit this file)
nano .env
```

Your `.env` file should look like:
```bash
METAAPI_TOKEN=eyJhbGc...  # Your real token
METAAPI_ACCOUNT_ID=4e8fb4f8...  # Your real account ID
```

### 3. Required Credentials

| Service | Variable | Where to Get |
|---------|----------|--------------|
| **MetaAPI** | `METAAPI_TOKEN` | [metaapi.cloud](https://metaapi.cloud/) |
| **MetaAPI** | `METAAPI_ACCOUNT_ID` | MetaAPI dashboard after connecting MT5 |
| **NewsAPI** (optional) | `NEWS_API_KEY` | [newsapi.org](https://newsapi.org/) |

### 4. Production Deployment

When deploying to a server or cloud platform:

**Option A: VPS/Cloud Server**
```bash
# Set environment variables in shell
export METAAPI_TOKEN="your_token"
export METAAPI_ACCOUNT_ID="your_account_id"

# Or use a .env file (make sure it's not in git)
```

**Option B: Docker**
```bash
# Use --env-file flag
docker run --env-file .env your_trading_bot
```

**Option C: Cloud Platforms (Heroku, Railway, etc.)**
- Use their web dashboard to set environment variables
- Never put secrets in code or config files

### 5. Rotating Credentials

**If your API key is compromised:**
1. Immediately revoke it in MetaAPI dashboard
2. Generate a new token
3. Update your `.env` file
4. Restart the trading bot
5. Check for any unauthorized trades

### 6. File Permissions

On Linux/Mac, restrict `.env` file access:
```bash
chmod 600 .env  # Only you can read/write
```

## 🚨 What to Do If You Accidentally Commit Secrets

### If you haven't pushed yet:
```bash
# Remove the file from git but keep it locally
git rm --cached .env

# Amend the commit
git commit --amend

# Verify it's gone
git log --patch
```

### If you already pushed to GitHub:
1. **Immediately revoke the exposed credentials** (MetaAPI dashboard)
2. Remove the secret from git history:
```bash
# Use git-filter-repo (recommended) or BFG Repo Cleaner
git filter-repo --path .env --invert-paths

# Force push
git push origin --force --all
```
3. Generate new credentials
4. **Important:** The old credentials are now public - they must be revoked!

## 🛡️ Additional Security Measures

### Trading Account Security
- Use a **demo account** for testing (no real money at risk)
- Start with **minimum position sizes** when going live
- Set **maximum loss limits** in the code
- Enable **2FA** on your MetaAPI account
- Use **separate API keys** for testing vs production

### Code Security
- Review code before running (especially from unknown sources)
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- Monitor your trading bot logs for suspicious activity
- Use **read-only API keys** when possible (for monitoring only)

### Network Security
- Use VPN when accessing trading accounts from public WiFi
- Keep your server/VPS firewall configured
- Use SSH keys instead of passwords for server access
- Monitor server access logs

## 📞 Reporting Security Issues

If you discover a security vulnerability in this project:

**DO NOT** open a public GitHub issue.

Instead:
1. Email: jebariayman8@gmail.com
2. Include: Description, impact, steps to reproduce
3. Allow 48 hours for response

We will:
- Acknowledge receipt within 48 hours
- Provide a fix timeline
- Credit you in the security advisory (if you wish)

## 🔍 Security Checklist

Before running in production:

- [ ] `.env` file created with real credentials
- [ ] `.env` is in `.gitignore`
- [ ] No API keys in Python code
- [ ] Tested on demo account first
- [ ] Maximum loss limits configured
- [ ] 2FA enabled on broker accounts
- [ ] Server firewall configured (if applicable)
- [ ] Monitoring/alerting set up
- [ ] Backup strategy in place

---

**Remember:** Security is not a one-time setup. Regularly review your credentials, monitor for suspicious activity, and keep software updated.
