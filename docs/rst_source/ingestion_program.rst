Ingestion Program
=================

Ingestion program is the first step in the competition pipeline. 
It is responsible for loading the data and running the participant's code on it.
To run run the ingestion yourself you can use the following command:

.. code-block:: bash

   python3 run_ingestion.py \ 
   --use-random-mus \ 
   --systematics-tes \ 
   --systematics-soft-met \ 
   --systematics-jes \ 
   --systematics-ttbar-scale \ 
   --systematics-diboson-scale \ 
   --systematics-bkg-scale \ 
   --num-pseudo-experiments 100 \ 
   --num-of-sets 1 \ 
   --parallel


.. toctree::
   :maxdepth: 1

   datasets
   derived_quantities
   ingestion
   ingestion_parallel
   systematics
   visualization
