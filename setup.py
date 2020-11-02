import os.path as osp
import re
from setuptools import setup, find_packages


def find_version():
    with open(osp.join('foamlight', '__init__.py')) as fp:
        for line in fp:
            m = re.search(r'^__version__ = "(\d+\.\d+\.\d[a-z]*\d*)"', line, re.M)
            if m is None:
                # could be a hotfix
                m = re.search(r'^__version__ = "(\d.){3}\d"', line, re.M)
            if m is not None:
                return m.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name='foamlight',
    version=find_version(),
    author='Jun Zhu',
    author_email='zhujun981661@gmail.com',
    description='',
    long_description='',
    url='',
    packages=find_packages(),
    tests_require=['pytest'],
    install_requires=[
        'pyfoamalgo>=0.0.8',
        'foamgraph',
    ],
    extras_require={
        'karabo': [
            'extra_data',
            'karabo-bridge',
        ],
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'foamlight=foamlight.services:application',
        ],
    },
    package_data={
        'foamlight': [
            'apps/icons/*.png',
        ]
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ]
)
