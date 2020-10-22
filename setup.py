"""
*****************************************************
*               Color Tracker
*
*              Gabor Vecsei
* Website:     https://gaborvecsei.com
* Blog:        https://gaborvecsei.com
* LinkedIn:    https://www.linkedin.com/in/gaborvecsei
* Github:      https://github.com/gaborvecsei
*
*****************************************************
"""

from codecs import open
from os import path

from setuptools import setup, find_packages

VERSION = '0.1.1'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name='color_tracker',
    version=VERSION,
    description='Easy to use color tracking package for object tracking based on colors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gaborvecsei/Color-Tracker',
    author='Gabor Vecsei',
    author_email='vecseigabor.x@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'],
    keywords='color tracker vecsei gaborvecsei color_tracker',
    install_requires=requirements,
    packages=find_packages(),
)
