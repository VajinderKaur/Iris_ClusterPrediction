from setuptools import setup, find_packages

setup(
    name='Iris_ClusterPrediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'seaborn',
        'scikit-learn',
        'joblib'
    ],
)
