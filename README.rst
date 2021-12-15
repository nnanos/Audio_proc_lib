========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/Audio_proc_lib/badge/?style=flat
    :target: https://Audio_proc_lib.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/nnanos/Audio_proc_lib/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/nnanos/Audio_proc_lib/actions

.. |requires| image:: https://requires.io/github/nnanos/Audio_proc_lib/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/nnanos/Audio_proc_lib/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/nnanos/Audio_proc_lib/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/nnanos/Audio_proc_lib

.. |version| image:: https://img.shields.io/pypi/v/Audio-proc-lib.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/Audio-proc-lib

.. |wheel| image:: https://img.shields.io/pypi/wheel/Audio-proc-lib.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/Audio-proc-lib

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/Audio-proc-lib.svg
    :alt: Supported versions
    :target: https://pypi.org/project/Audio-proc-lib

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/Audio-proc-lib.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/Audio-proc-lib

.. |commits-since| image:: https://img.shields.io/github/commits-since/nnanos/Audio_proc_lib/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/nnanos/Audio_proc_lib/compare/v0.0.0...main



.. end-badges

A library that contains basic audio signal processing functionalities

* Free software: MIT license

Installation
============

::

    pip install Audio-proc-lib

You can also install the in-development version with::

    pip install https://github.com/nnanos/Audio_proc_lib/archive/main.zip


Documentation
=============


https://Audio_proc_lib.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
