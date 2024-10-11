#! /usr/bin/env python

from setuptools import setup


# try:
# 	from pdll import __version__
# except:
# 	print("failed to load version, assuming 0.0.1")
# 	__version__ = "0.0.1dev"

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()



setup(
    name='pdll',
    version='0.3.1',
    author='Mohamed Karim Belaid',
    author_email='karim.belaid@idiada.com',
    description='Pairwise difference learning library is a scikit learn compatible library for learning from pairwise differences.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        "Tracker": "https://github.com/Karim-53/pdll/issues",
        "Source": "https://github.com/Karim-53/pdll",
    },
    license="Apache-2.0",
    # python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'setuptools',
    ],

    # packages=find_packages(),
    packages=[
        'pdll',
        # 'pdl.examples',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",    'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
