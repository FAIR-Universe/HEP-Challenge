Datasets module
===============


The tabular dataset is created using the particle physics simulation tools 
`Pythia 8.2 <https://pythia.org/>`_ and 
`Delphes 3.5.0 <https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/>`_. 
The proton-proton collision events are generated with a center of mass energy of 13 TeV using Pythia8. 
Subsequently, these events undergo the Delphes tool to produce simulated detector measurements. 
We used an ATLAS-like detector description to make the dataset closer to experimental data. 
The events are divided into four groups:

1. Higgs boson signal (:math:`H \rightarrow \tau \tau`)
2. :math:`Z` boson background (:math:`Z \rightarrow \tau \tau`)
3. Diboson background (:math:`VV \rightarrow \tau \tau`)
4. :math:`t\bar{t}` background (:math:`t \bar{t}`)

By default the repo has a sample dataset. To get the bigger Public dataset,  
.. code-block:: python3
   from datasets import Neurips2024_public_dataset as public_dataset
   data = public_dataset()

This code is already included in the starting kit and you can run it to get the public dataset.

.. automodule:: datasets
   :members:
   :undoc-members:
   :show-inheritance:



For more details on Data, see the :doc:`data page <../pages/data>`.
