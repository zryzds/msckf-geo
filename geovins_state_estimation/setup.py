"""
Setup script for GEOVINS state estimation package.
"""
from setuptools import setup, find_packages
import os

# Read version from __init__.py
def read_version():
    with open(os.path.join('src', '__init__.py'), 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

# Read long description from README
def read_readme():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='geovins',
    version=read_version(),
    author='GEOVINS Development Team',
    author_email='your.email@example.com',
    description='Geographic Visual-Inertial Navigation System with MSCKF',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/geovins',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Navigation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pyyaml>=5.4.0',
    ],
    extras_require={
        'cv': [
            'opencv-python>=4.5.0',
            'opencv-contrib-python>=4.5.0',
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'plotly>=5.0.0',
        ],
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.11.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
        ],
    },
    entry_points={
        'console_scripts': [
            'geovins-run=examples.demo:main',
        ],
    },
    include_package_data=True,
    package_data={
        'geovins': ['config/*.yaml'],
    },
    zip_safe=False,
)
