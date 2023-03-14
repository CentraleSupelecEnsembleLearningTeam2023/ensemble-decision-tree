import sys
import site
import setuptools
from setuptools import find_packages


setuptools.setup(
    name='decisiontree',
    version='0.0.1',
    description='Decision Tree for Classification and Regression',
    author='Antoine CLOUTE, Annabelle LUO, Ali NAJEM, Xioayan HONG, Karim EL HAGE',
    author_email='antoine.cloute@student-cs.fr',
    packages=['decisiontree'],
    zip_safe=False)