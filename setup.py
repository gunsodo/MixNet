import setuptools
from os import path

version = {}
here = path.abspath(path.dirname(__file__))

with open( path.join(here, 'README.md'), 'r', encoding='utf-8') as fh:
    long_description = fh.read()
    
with open(path.join(here, 'min2net/version.py'), encoding='utf-8') as (
        version_file):
    exec(version_file.read(), version)

setuptools.setup(
    name='min2net',
    version=version['__version__'],
    author='INTERFACES',
    author_email='IoBT.VISTEC@gmail.com',
    description='MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://MIN2Net.github.io',
    download_url='https://github.com/IoBT-VISTEC/MIN2Net/releases',
    project_urls={
        'Bug Tracker': 'https://github.com/IoBT-VISTEC/MIN2Net/issues',
        'Documentation': 'https://MIN2Net.github.io',
        'Source Code': 'https://github.com/IoBT-VISTEC/MIN2Net',
    },
    license='Apache Software License',
    keywords=[
        'Brain-computer Interfaces'
        'BCI', 
        'Motor Imagery',
        'MI', 
        'Multi-task Learning',
        'Deep Metric Learning',
        'DML', 
        'Autoencoder',
        'AE',
        'EEG Classifier'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires = [
        # 'tensorflow-gpu==2.8.2', not support tensorflow-gpu via pip
        'tensorflow-addons==0.16.1', #'tensorflow-addons==0.9.1',
        'scikit-learn>=0.24.1',
        'wget>=3.2',
        'ray>=1.11.0'
        'pandas'
    ],
    package_data= {
        # all .csv files at any package depth
        '': ['**/*.csv']
    },
    packages=setuptools.find_packages(),
    python_requires='>=3.7, <=3.10.4',

)
