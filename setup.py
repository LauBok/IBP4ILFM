import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Python_IBP", 
    version="0.0.1",
    author="Bo Liu & Linlin Li",
    author_email="bl226@duke.edu",
    description="Bayesian Infinite Latent Feature Models and IBP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LauBok/Infinite-Latent-Feature-Models-and-the-Indian-Buffet-Process",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ??",
        "Operating System :: ??",
    ],
    python_requires='>=3.6',
)