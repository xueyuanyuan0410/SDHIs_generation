# Generating Molecules with Transformers

This repository contains the source code for two different paper demonstrating how molecules can be generated with the transformer architecture.

[Spatial Generation of Molecules with Transformers](https://ieeexplore.ieee.org/document/9533439) addresses the generation and completion molecules in a 3D atom space.

"Transformers for Molecular Graph Generation" demonstrates how transformers can be utilized for the generation of graphs.

## Setup

A conda environment can be created with

`conda env create -f environment.yaml`

## Training

To train one of the models run:

`python molegent/main.py --train`

with 

 `-config_file configs/spatial.yaml --model_type spatial`

 for the model featured in "Spatial Generation of Molecules with Transformers" or

`-config_file configs/graph.yaml --model_type graph`

for the model featured in "Transformers for Molecular Graph Generation".
