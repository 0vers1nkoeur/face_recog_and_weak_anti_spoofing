# To create locally

```git checkout -b name_of_branch```

# To synchronise with the distant branch (remote)

```git push -u origin name_of_branch```

# To see the distant branches

```git branch -r```

# To see the local branches

```git branch```

# To delete a branch

- Local : ```git branch -d {mybranch}```
- Distant : ```git push origin --delete {mybranch}```

# To pull the branch without taking into account the actual modifications (ERASE ALL THE LOCAL MODIFICATIONS)

- ```git checkout {mybranch}```
- ```git fetch origin```
- ```git reset --hard origin/{aimedbranch}```
