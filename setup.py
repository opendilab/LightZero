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
import os
import numpy as np

from setuptools import find_packages
from distutils.core import setup
from Cython.Build import cythonize  # this line should be after 'from setuptools import find_packages'

here = os.path.abspath(os.path.dirname(__file__))


def find_pyx(path=None):
    path = path or os.path.join(here, 'core')
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    print(pyx_files)
    return pyx_files


setup(
    name='LightZero',
    version='0.0.1',
    description='MCTS/MuZero Algorithm Toolkits',
    # long_description=readme,
    long_description_content_type='text/markdown',
    author='opendilab',
    author_email='opendilab@pjlab.org.cn',
    url='https://github.com/opendilab/LightZero',
    license='Apache License, Version 2.0',
    keywords='Reinforcement Learning, MCTS',
    packages=[
        # framework
        *find_packages(include=('core', "core.*")),
        # application
        *find_packages(include=('zoo'
                                'zoo.*')),
    ],
    package_data={
        package_name: ['*.yaml', '*cfg']
        for package_name in find_packages(include=('core.*'))
    },
    python_requires=">=3.7",
    install_requires=[
        'DI-engine>=0.4.4',
        'gym==0.25.1',
        'torch>=1.1.0, <=1.12.1',
        'numpy>=1.18.0',
        'kornia',
    ],
    ext_modules=cythonize(
        find_pyx(),
        language_level=3,
    ),
    include_dirs=[np.get_include()],
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
