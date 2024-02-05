import os
from setuptools import setup

def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()

# Read the Requirements file
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line]

setup(
    name='saged',
    version='0.1.0',
    author='Mohamed Abdelaal, Daniel Staedler, Tim Ktitarev',
    author_email='mohamed.abdelaal@softwareag.com',
    description='SAGED: Meta Learning-based Error Detection for Structured Data',
    long_description=read('README.md'),
    packages=['saged', 'baseline'],
    install_requires=requirements,
    classifiers=['Development Status :: 3 - Alpha'],
)
 