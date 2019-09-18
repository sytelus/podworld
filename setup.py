# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boxworld",
    version="0.0.1",
    author="Shital Shah",
    author_email="shitals@microsoft.com",
    description="2D dynamic world for RL experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/boxworld",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=[
          'pymunk', 'pygame', 'numpy', 'Shapely', 'matplotlib', 'Pillow', 'opencv-python'
    ]
)