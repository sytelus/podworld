# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="podworld",
    version="0.2.0",
    author="Shital Shah",
    author_email="shitals@microsoft.com",
    description="2D partially observable dynamic world for RL experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/podworld",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=[
          'pymunk', 'pygame', 'numpy', 'gym', 
          'tensorwatch' # optional, only if test code/baselines are used
    ]
)