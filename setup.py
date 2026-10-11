from setuptools import setup, find_packages

setup(
    name="ecg_xai",
    version="1.0.0",
    description="Interpretable ECG Signal Analysis using Feature-Based Machine Learning and SHAP Explainability",
    author="Kemsaram Omkar",
    author_email="kemsaramomkar30@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "shap>=0.44.0",
    ],
)
