#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# Load the README as long_description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="panorai",  # Adjust if you prefer another package name
    version="0.1.0",  # Update or automate versioning as needed
    author="RLSGarcia",
    author_email="rlsgarcia@icloud.com",
    description="A Python package for panoramic image projection and blending using Gnomonic (and other) projections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobinsonGarcia/PanorAi",  
    packages=find_packages(exclude=["tests*", "docs*"]),  # Adjust as needed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or another license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-image",
        "scipy",
        "joblib",
        "pydantic>=2.0.0",
        # Add or remove dependencies based on your project's requirements
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            # any additional dev/test tools
        ]
    },
    entry_points={
        # Example: If you provide console scripts, declare them here.
        # "console_scripts": [
        #     "panorai-cli=panorai.cli:main",
        # ],
    },
    include_package_data=True,  # If your package includes non-Python files
    license="MIT",  # Or whichever license you use
)