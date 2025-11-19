# How to Push to Your Branch

## 1. Switch to your branch
```bash
git checkout <mybranch>
```

## 2. Check branch status
```bash
git status
```

## 3. Add and commit changes
```bash
git add .
git commit -m "Your commit message"
```

## 4. Push to the remote
```bash
git push
```

---

# How to Merge Your Branch

## Option 1 — Using GitHub (recommended)
- Create a Pull Request (PR)
- Review and merge on GitHub

## Option 2 — Using the CLI
```bash
git checkout main
git merge <mybranch>
git push origin main
```