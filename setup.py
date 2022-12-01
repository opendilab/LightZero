# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


setup(
    name='LightZero',
    version='0.0.1',
    description='',
    # long_description=readme,
    long_description_content_type='text/markdown',
    author='opendilab',
    author_email='',
    url='https://github.com/opendilab/LightZero',
    license='Apache License, Version 2.0',
    keywords='LightZero',
    packages=[
        # framework
        *find_packages(include=('core', "core.*")),
        # application
        *find_packages(include=('zoo'
                                'zoo.*')),
    ],
    package_data={
        package_name: ['*.yaml', '*.xml', '*cfg', '*SC2Map']
        for package_name in find_packages(include=('core.*'))
    },
    python_requires=">=3.7",
    install_requires=[
        'gym==0.25.1',  # pypy incompatible; some environmrnt only support gym==0.22.0
        'torch>=1.1.0, <=1.12.1',  # If encountering pytorch errors, you need to do something like https://github.com/opendilab/DI-engine/discussions/81
        'numpy>=1.18.0',
    ],
    entry_points={'console_scripts': ['core=core.entry.cli:cli', 'ditask=core.entry.cli_ditask:cli_ditask']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)