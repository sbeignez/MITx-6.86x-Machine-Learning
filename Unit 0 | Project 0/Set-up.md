# Set-up

## Virtual environement

[Installing packages using pip and virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
> If you are using Python 3.3 or newer, the venv module is the preferred way to create and manage virtual environments.

### Python version

Python 3.6

### Creating a virtual environment

`python3 -m venv mitx`

### Activating a virtual environment

`source mitx/bin/activate`

Check: `which python`


## packages

* NumPy
* matplotlib
* scikit-learn
* SciPy
* tqdm
* PyTorch

python3 -m pip

```python
pip3 install numpy
pip3 install matplotlib
pip3 install scipy
pip3 install tqdm
pip3 install scikit-learn
```

* Create the environement
* Activate the environment
* Create a txt file called "requirements.txt"
* Write down all the required packages, on package on each line
* Save the txt file
* Type the following code in the command console: pip install -r requirements.txt


`python3 -m pip install -r requirements.txt`

`python3 -m pip freeze`