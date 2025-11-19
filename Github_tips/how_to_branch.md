# Git — Branch Management Cheat Sheet

## Create a new local branch
```bash
git checkout -b <branch_name>
```

## Push the branch to the remote (and set upstream)
```bash
git push -u origin <branch_name>
```

## List remote branches
```bash
git branch -r
```

## List local branches
```bash
git branch
```

## Delete a branch
- **Local**  
  ```bash
  git branch -d <branch_name>
  ```
- **Remote**  
  ```bash
  git push origin --delete <branch_name>
  ```

## Reset a branch and discard all local changes  
*(Force sync with the remote version — irreversible)*

```bash
git checkout <branch_name>
git fetch origin
git reset --hard origin/<target_branch>
```
