# How to create a virtual environnement ?

A **virtual environment** is an isolated workspace for Python projects.  
It allows you to manage dependencies separately for each project — avoiding version conflicts.

---

# Pre-checks

- Install the good version of pyhton on your system computer, ex: ```brew install python@X.YY``` for Mac, apt for Linux and npm for Windows (look for spec on each OS)
- Version of python : ```python3 --version``` => KEEP THE BIN PATH
- BEING ON THE RIGHT FOLDER WHERE WE WORK TO NOT SPREAD ENV EVERYWHERE

# Recipes

1 Create a virtual env
    1.1 Last version : ```path/to/bin/python3 -m venv {myenv}```
    1.2 For a certain version : ```path/to/bin/pythonX.YY venv {myenv}```
2 Update the last version of pip of the aimed version : ```pip install --upgrade pip```
3 Activation of the virtual environnement
    3.1 Verify if the good version has been installed on the virtualised env : ```which pythonX.YY``` 
    3.2 Mac/Linux : ```source {myenv}/bin/activate```
    3.3 Windows : ```{myenv}\Scripts\Activate.ps1```
4 Install dependencies
    4.1 Install a certain dependency : ```pip install {package}```
    4.2 Install using requirements.txt : ```pip install -r requirements.txt```
5 Deactivation of the virtual environnement : ```deactivate```

---

# Problems ? Solutions !

1 Deactivate
2 Remove the virtual environnement : ```rm -rf myenv```
3 Reinstall the virtual environnement
