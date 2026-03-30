# 📤 Upload to GitHub – Quick Guide

**Target:** https://github.com/STS1983/ohlcv-cache

## Option 1: Using GitHub CLI (Recommended)

### Install gh (if not installed)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gh

# Or download from: https://github.com/cli/cli/releases
```

### Authenticate
```bash
gh auth login
# Follow prompts: GitHub.com → HTTPS → Login with browser
```

### Create & Push Repository
```bash
cd /home/nodeadmin/.openclaw/workspace/ohlcv_cache

# Initialize git (if not done)
git init
git add .
git commit -m "Initial commit: OHLCV Cache v1.0"

# Create repo and push
gh repo create ohlcv-cache --public --source=. --push
```

**Done!** 🎉 Your repo is live at: `https://github.com/STS1983/ohlcv-cache`

---

## Option 2: Using Git + GitHub Web UI

### Step 1: Initialize Git
```bash
cd /home/nodeadmin/.openclaw/workspace/ohlcv_cache
git init
git add .
git commit -m "Initial commit: OHLCV Cache v1.0"
```

### Step 2: Create GitHub Repo
1. Go to https://github.com/new
2. Repository name: `ohlcv-cache`
3. Description: "High-performance OHLCV data compression for trading"
4. Public ✅
5. **Don't** initialize with README (we already have one)
6. Click "Create repository"

### Step 3: Push to GitHub
```bash
git remote add origin https://github.com/STS1983/ohlcv-cache.git
git branch -M main
git push -u origin main
```

---

## Option 3: One-Liner (if gh is available)
```bash
cd /home/nodeadmin/.openclaw/workspace/ohlcv_cache && git init && git add . && git commit -m "Initial commit" && gh repo create ohlcv-cache --public --source=. --push
```

---

## After Upload

### Add Topics (optional)
On GitHub repo page, add topics:
- `trading`
- `compression`
- `backtesting`
- `cryptocurrency`
- `python`
- `quantitative-finance`

### Enable Issues & Discussions
- Settings → Features → Enable Issues
- Settings → Features → Enable Discussions (optional)

### Add PyPI Badge (after publishing)
Update README.md with:
```markdown
[![PyPI](https://img.shields.io/pypi/v/ohlcv-cache.svg)](https://pypi.org/project/ohlcv-cache/)
```

---

## Publish to PyPI (Optional)

### Install twine
```bash
pip install twine
```

### Build package
```bash
cd /home/nodeadmin/.openclaw/workspace/ohlcv_cache
python3 -m build
```

### Upload to PyPI
```bash
twine upload dist/*
# Enter your PyPI API token when prompted
```

### Test installation
```bash
pip install ohlcv-cache
```

---

## Checklist

- [ ] All files in `ohlcv_cache/` directory
- [ ] README.md with badges
- [ ] LICENSE (MIT)
- [ ] .gitignore (Python)
- [ ] pyproject.toml (package config)
- [ ] Tests passing (14/14)
- [ ] Demo working
- [ ] Repository created on GitHub
- [ ] Initial commit pushed
- [ ] Topics added
- [ ] (Optional) PyPI published

---

**Need help?** Open an issue on GitHub!
