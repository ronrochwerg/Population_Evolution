
# Population Evolution

This repository contains code for evolving a population of models using evolutionary algorithms. The main components include defining parameters and managing a population of models to evolve over generations.[[1]](#1)

- [Installation](#installation)
- [Usage](#usage)
- [Overview of Parameters](#Overview-of-Parameters)
- [Overview of Population Class](#Overview-of-Population-Class)
- [Contributing](#contributing)
<!--- [License](#license)-->

## Installation

To use this repository, it is currently necessary to clone it to your local machine:

```bash
git clone https://github.com/ronrochwerg/Population_Evolution.git
```

Make sure you have the required dependencies installed, which you can install via pip:

```bash
pip install -r requirements.txt
```
Note: we are currently in the process of adding this project to PyPI which will make installation easier.

## Usage
```python
import LGP
import Population_Evolution as Population
from numpy.random import default_rng
from sklearn import datasets

# load dataset
iris = datasets.load_iris()
n_var = iris.data.shape[1]

# create a parameters object
rng = default_rng()
model_param = LGP.Parameters(n_var, rng)
population_param = Population.Parameters(rng, LGP.LGP, model_param, iris.data, iris.target)

population = Population.Population(population_param)
population.initialize_run()

population.run_evolution(500, file=f, n_jobs=n_jobs)
best = population.return_best()
```
Note: For the LGP model used in the above case please see [LGP](https://github.com/ronrochwerg/LGP.git) library.

## Overview of Parameters
This file defines the `Parameters` class, which contains all the necessary settings for the evolutionary process.

- **Attributes:**
  - `rng`: Random number generator.
  - `model`: The type of model in the population.
  - `model_param`: Parameters for the model.
  - `fitness_maximized`: Boolean indicating whether the fitness is maximized.
  - `multi_obj`: Boolean for multi-objective fitness.
  - `parent_selection_method`: Function for parent selection.
  - `survivor_selection_method`: Function for survivor selection.
  - `dataX`, `dataY`: Features and labels used in the models.
  - `weights`: Weights for the data points.
  - `verbose`: Level of output during the process.
  - `early_stop`: Fitness value for early stopping.

- **Methods:**
  - Getters and setters for various parameters such as `num_eval_per_gen`, `tourney_size`, `num_tourney`, `num_win_loss`, `replacement_ts`, and `num_parent_rs`.
  - `print_attributes()`: Prints all attributes of the class.

## Overview of Population Class
This file defines the `Population` class, which manages the population of models and handles the evolutionary process.

- **Attributes:**
  - `param`: Instance of the `Parameters` class.
  - `population`: List of models in the population.
  - `parent_selection_time`, `children_creation_time`, `evaluation_time`, `survivor_selection_time`: Track the time spent on different stages of the evolutionary process.

- **Methods:**
  - `initialize_run(name=True)`: Initializes the population by creating models.
  - `run_evolution(generations, file=None, n_jobs=1)`: Runs the evolutionary process over a specified number of generations.
  - `return_best()`: Returns the best individual based on fitness.
  - `return_best_multi(k_min=None, k_max=None, min_found_k=False, file=None, chosen_obj=None)`: Returns the best individuals for multi-objective optimization, with options for clustering and specific objectives.


<!--
## License
This project is licensed under the MIT License.
-->

## References

<a id="1">[1]</a> 
Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.

