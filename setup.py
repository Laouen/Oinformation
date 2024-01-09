from setuptools import setup, find_packages

setup(
    name='Oinfo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'networkx'
    ]   
)