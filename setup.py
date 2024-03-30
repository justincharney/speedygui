from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="speedygui",
    version="0.1.3",
    author="Justin Charney",
    author_email="justin.charney@gmail.com",
    description="A simple template for creating GUI PyTorch medical image segmentation applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "toga",
        "torch",
        "pillow",
        "torchvision",
        "scikit-image"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)