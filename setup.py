from setuptools import setup, find_packages

setup(
    name="poligonos",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
        "matplotlib",
        "bokeh"
    ],
    author="Jorge Guzmán",
    author_email="jguzmanv01@uc.cl",
    description="Un paquete para la generación y visualización de polígonos en minería",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jorgeguzz/generacion_poligonos",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
