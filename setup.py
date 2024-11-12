from setuptools import setup, find_packages

setup(
    name="sentiment_analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "numpy",
        "tqdm"
    ],
)
