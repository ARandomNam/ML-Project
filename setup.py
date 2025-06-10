"""
Resume Match Score Predictor - Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resume-match-predictor",
    version="1.0.0",
    author="Resume Matcher Team",
    author_email="team@resumematcher.com",
    description="A lightweight NLP-powered tool for resume-job matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/resume-match-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "resume-matcher=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["data/sample/*"],
    },
) 