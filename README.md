# Team: Markov Mayhem (Winner!)

### MOASEI AAMAS-2025 Competition 
Track #3: Wildfire (Both Agent and Task Openness)

- Web: [MOASEI](https://oasys-mas.github.io/moasei.html)  
- GitHub: [oasys-mas/free-range-zoo](https://github.com/oasys-mas/free-range-zoo)  
- Kaggle: [Competition Configurations](https://www.kaggle.com/datasets/picklecat/moasei-aamas-2025-competition-configurations)


## Members
University of Utah, Utah, USA
- Varun Raveendra
- Seongil Heo
- Yanxi Lin

## Repository Structure
The structure of the repository described below:
```sh
.
├── competition_configs                 # Environment configurations
│   └── wildfire
├── free_range_zoo                      # Package source
│   ├── envs                            #    Environment implementations
│   ├── utils                           #    Converters / environment abstract classes
│   └── wrappers                        #    Model wrappers and utilities
├── tests                               # Tests
│    ├── free_range_zoo                 #    Tests for the free_range_zoo package
│    │   ├── envs                       #       environment utilities
│    │   └── utils                      #       all package utilities
│    ├── profiles                       # Environment performance profiles
│    └── utils                          # Testing utilities
├── experiments                         # Experiments (**ours**)
│    ├── core.py                        #   Core classes definitions (Graph, Actor, Critic, Network)
│    ├── evaluation.py                  #   Evaluation scripts 
│    ├── quick_start.py                 #   Quick start guide and example scripts
│    ├── test.py                        #   Test scripts for the baseline models
│    ├── train_a2c.py                   #   Training script for A2C model
│    ├── train_gnn.py                   #   Training script for PL model
│    └── utils.py                       #   Utility functions
├── LICENSE                             # License file
├── poetry.lock                         # Poetry lock file
├── pyproject.toml                      # Package dependencies and package definition
├── README.md                           # Project documentation
└── setup.cfg                           # Setup configuration
```

## Installation
For installation, please refer to the [Installation Guide](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html) for detailed instructions on how to set up the environment and install the required dependencies.

## Usage
```sh
python evaluation.py [OPTIONS] <output> <model> <config>
```
### Required Arguments
- output: Path to the directory where evaluation results and logs will be saved.
- model : Path to the directory containing the trained model.
- config: Path to the environment configuration (e.g., competition_configs/WS3.pkl).

### Optional Arguments

| Option                        | Description                                                                   |
|-------------------------------|-------------------------------------------------------------------------------| 
| -h, --help                    |   Show help message and exit                                                  |
| --cuda	                    |   Use CUDA (GPU) if available                                                 |
| --threads THREADS	            |   Number of threads to use                                                    |
| --log_level LEVEL             |	Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)                   |
| --seed SEED	                |   Random seed for the evaluation process                                      |
| --dataset_seed DATASET_SEED	|   Random seed for initializing the environment configuration                  |
| --testing_episodes N          |   Number of test episodes to run (parallel_envs)                              |


## Example
```sh
python run/evaluation.py ./output logging/250519_120000/model_a2c.h5 ./competition_config/wildfire/WS1.pkl --testing_episodes 100
```
## License
This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/).