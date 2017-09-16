"""
*****************************************************
*               Color Tracker
*
*              Gabor Vecsei
* Email:       vecseigabor.x@gmail.com
* Blog:        https://gaborvecsei.wordpress.com/
* LinkedIn:    https://www.linkedin.com/in/gaborvecsei
* Github:      https://github.com/gaborvecsei
*
*****************************************************
"""

from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='color_tracker',
    version='0.1.0',

    description='Track color easily',
    long_description=long_description,

    url='https://github.com/gaborvecsei',

    author='Gabor Vecsei',
    author_email='vecseigabor.x@gmail.com',

    license='MIT',

    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],

    keywords='color tracker vecsei gaborvecsei color_tracker',

    packages=find_packages(),
)
