# Design and screening of succinate dehydrogenase inhibitors against agricultural fungi based on Transformer model

This repository contains the source code for how molecules can be generated with the graph-based transformer architecture(molegent).

<div align=center>
<img src="./fugure/流程图3_400dpi.tif" align=center >
</div>

The SMILES format data of 23 succinate dehydrogenase inhibitors provided by the 2020 version of the Fungicide Resistance Committee is in the data folder.

## Environment 

A conda environment can be created with
`conda env create -f environment.yaml`

## Training & generating

To pre-train one of the models run:
`python molegent/main.py --train --pretrain --config_file configs/graph.yaml --device cuda --model_type graph`
To train one of the models run:
`python molegent/main.py --train --config_file configs/graph.yaml --device cuda --model_type graph --cp_path /home/molegent/runs/best_moses_checkpoint.pt`
To train one of the models run:
`python molegent/main.py --config_file configs/graph.yaml --device cuda --model_type graph --cp_path /home/molegent/runs/best_moses_checkpoint.pt`


## Molecular docking
The code for molecular docking is in the 2_molecule_docking folder.


