from setuptools import setup, find_packages

setup(
    name="partana",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Metadata
    author="Hao Wu",
    author_email="aster0502@outlook.com",
    description="Particle analysis tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aster0502/partana",
    
    # Project dependencies
    install_requires=[
        # List your package dependencies here
        # For example:
        # "numpy>=1.20.0",
        # "pandas>=1.3.0",
    ],
    
    # Python version compatibility
    python_requires=">=3.11",
)
