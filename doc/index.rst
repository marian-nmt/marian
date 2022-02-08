Welcome to Marian's documentation!
==================================

|buildgpu| |buildcpu| |tests| |release| |license|

Marian is an efficient and self-contained Neural Machine Translation framework with an integrated
automatic differentiation engine based on dynamic computation graphs, written entirely in C++.

This is developer documentation. User documentation is available at https://marian-nmt.github.io/docs/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   structure
   graph
   operators
   layer
   factors
   api/library_index

   contributing

   doc_guide


Indices and tables
------------------

* :ref:`genindex`


.. |buildgpu| image:: https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-dev-cuda-10.2.svg?label=CUDAC%20Build
   :target: http://vali.inf.ed.ac.uk/jenkins/job/marian-dev-cuda-10.2/
   :alt: GPU build status

.. |buildcpu| image:: https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-dev-cpu.svg?label=CPU%20Build
   :target: http://vali.inf.ed.ac.uk/jenkins/job/marian-dev-cpu/
   :alt: CPU build status

.. |tests| image:: https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/marian/job/marian-regression-tests.svg?label=Tests
   :target: http://vali.inf.ed.ac.uk/jenkins/job/marian-regression-tests/
   :alt: Tests status

.. |release| image:: https://img.shields.io/github/release/marian-nmt/marian.svg?label=Release
   :target: https://github.com/marian-nmt/marian/releases
   :alt: Latest release

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: ../LICENSE.md
   :alt: License: MIT
