
# Rebase feature branch onto main and resolve conflicts (safe, step-by-step)

1. Update main
```bash
git checkout main
git pull origin main
```

2. Rebase your feature branch onto main
```bash
git checkout feature-branch
git rebase main
# (Optional) Interactive rebase to clean commits:
git rebase -i main
```

3. Resolve conflicts (if any)
```bash
# See conflicted files
git status

# Open each conflicted file, fix conflicts, then stage them:
git add <file1> <file2> ...

# Continue the rebase
git rebase --continue

# If you want to skip the current patch:
git rebase --skip

# To abort and return to the pre-rebase state:
git rebase --abort
```
Tips: use git mergetool if helpful, and run tests/lint before continuing.

4. Push the rebased branch
```bash
# If you previously pushed this branch, force with lease:
git push --force-with-lease origin feature-branch
```
Note: --force-with-lease is safer than --force because it protects against overwriting upstream updates.

5. Merge into main and push
```bash
git checkout main
git pull origin main         # ensure main is up to date
git merge feature-branch     # usually fast-forward after a successful rebase
git push origin main
```

Notes : If the branches had diverged significantly, you will have to resolve conflict for each commit during the rebase. Take your time to ensure correctness at each step.
