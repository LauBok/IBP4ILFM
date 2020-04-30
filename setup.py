import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IBP4ILFM", 
    version="0.1a",
    author="Bo Liu & Linlin Li",
    author_email="bl226@duke.edu",
    description="Bayesian Infinite Latent Feature Models and IBP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LauBok/IBP4ILFM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT",
    ],
    python_requires='>=3.6',
)