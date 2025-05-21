

# Team: Markov Mayhem 

MOASEI AAMAS-2025 Competition Configurations 

Track #3: Wildfire (Both Agent and Task Openness)

## Members
- Varun Raveendra
- Seongil Heo
- Yanxi Lin

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
python run/evaluation.py ./output model/vpg4_100.h5 ./competition_config/wildfire/WS1.pkl --testing_episodes 100
```