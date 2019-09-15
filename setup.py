import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sway__utils",
    version="0.0.1",
    author="Sean Maudsley-Barton",
    author_email="s.maudsley-barton@mmu.ac.uk",
    description="This package calculates common sway metrics from raw time-serise data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
