# HaloGNN

## Overview

In this repo, we compare the capabilities of two models for inferring dark matter halo masses from galaxy catalogues in the IllustrisTNG and Astrid hydrodynamical simulation suites by performing autoregressive flows on each of the two models outputs:
- fully-connected network (FCN) using only summary statistics 
- a Graph Neural Network (GNN) learning point-cloud embeddings 
This repo contains data generation scripts, model definitions, training pipelines, and notebooks to aid in understanding our final analysis. The final results are documented in "blah blah."

<img src="figures/network_graphic.png" width="1000">

## Quick Start Guide

1. Download the ASTRID and TNG halo catalogues to your machine
2. Edit 'configs.yaml' and set 'catalogue_dir' to point to your catalogues
3. Run the pipeline:
```bash
python run_pipeline.py
```

This project is designed to be self-contained, and so all outputs will populate within the DeepHalos directory in your machine.

## Installation and Dependencies

Once you install this repo, first run the data generation script, then you may run the model files. It is designed to be self-contained and so the data + model 

This repo ..?
todo: finish


## Directory Guide

todo: update this once youre wrapping up
```
├── data/
├── figures/
├── notebooks/
    ├── coverage_levels.ipynb
    └── training_models.ipynb  
├── scripts/
    ├── fcn.py
    ├── gnn.py
    └── generate_gnn_data.py
├── galaxyGNN/
    ├── models/
    └── data/  
└── README.md
```


## Scripts

- 'fcn_data.py': This generates the data for the fully connected network, which takes in summary statistics of a halo.
- 'fcn_model.py': This is the FCN model. More detailed documentation on how it works is available in the file itself or in the paper
- 'fcn.py': This loads the data, instantiates an FCN model, and trains.
- 'gnn_data.py': Also generates the data for the graph neural network. As represented above, it creates a point cloud summary as a proxy for the summary statistics vector.
- 'gnn_model.py': The GNN model. More detailed documentation available in the file and paper.
- 'gnn.py': Loads the data, instantiates a GNN model, and trains.
- 'helpers.py': Contains functions for pre, mid, and post training. Pre-training: loading, processing, and standardizing data. Mid-training: creates dataloaders, loads models. Post-training: generating and unstandardizing predictions.
Handles data, provides functions to handle standardization, used features, managing dataloaders, models, and post-training.
- 'utils.py': Provides some essential data loading, pre-processing, and path management functions. Does not need to be imported if helpers.py already is.

## Notebooks
- 'training_models.ipynb': Walks you through a model training session. Make sure to budget requisite time and memory.
- 'coverage_levels.ipynb': Walks you through generating the coverage plots shown in the paper. Requires a pretrained model.
- 'data_comparison.ipynb': Highlights some differences between the Illustris and Astrid catalogues.
- 'figs.ipynb': Walks you through generating some of the notable figures for the paper. (Notably, the halo mass function)

## Usage Examples


## Contributing Authors and Contact

- Akhmetzhanova, Aizhan
- Brittain, Helena
- Cuesta-Lazaro, Carolina 
- Ni, Yueying



## configurations

### main
root: Path                          # Project directory
model_type: Literal["GNN", "FCN"]   # Model type, either GNN or FCN
sim: Literal["TNG", "ASTRID"]       # Simulation name: ASTRID or TNG
z: Literal[3, 4, 5, 6]              # Redshift, must be 3, 4, 5, or 6

### optional hyper parameters
graph_radius: float = 2000.0        # Graph radius for GNN
train_sim: Optional[str] = None     # Simulation used for training, defaults to sim
observables_only: bool = False      # If True, only observable features (JWST frequency bands) are used
hm_present: bool = False            # If True, we train on present-day halo mass