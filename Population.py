'''
Population class, should store the populaiton of the models and be able to run evolutionary stuff on it, and return fittest model...
should move the reproduction file from LGP into this library
'''
import numpy as np

class Population(object):

    def __init__(self, param):
        self.param = param
        self.population = []

    # creates models equal to population size
    def initialize_run(self, name = True):
        for i in range(self.param.num_eval_per_gen):
            ind = self.param.model(self.param.model_param)
            if name:
                ind.initialize(self.param.dataX, self.param.dataY, name=i)
            else:
                ind.initialize(self.param.dataX, self.param.dataY)
            self.population.append(ind)

    # runs evolution on the population for given number of generations
    def run_evolution(self, generations):
        # looping through given number of generations
        for generation in range(generations):
            num_evals = 0
            # looping through the correct number of evaluations per generation
            while num_evals < self.param.num_eval_per_gen:
                # obtaining parents (and non_survivors if given by selection method)
                parents, non_survivors = self.param.parent_selection_method(self.param, self.population)
                children = []
                # creating children in pairs for the possibility of recombination
                while len(parents) >= 2:
                    parent_1 = parents.pop()
                    parent_2 = parents.pop()

                    child_1 = self.population[parent_1].make_copy()
                    child_2 = self.population[parent_2].make_copy()

                    # children can mutate and recombine
                    change = False
                    while not change:
                        if self.param.rng.random() < self.param.mut_rate:
                            child_1.mutate()
                            child_2.mutate()
                            change = True
                        if self.param.rng.random() < self.param.recomb_rate:
                            child_1.recombine(child_2)
                            change = True
                    # giving the children fitness
                    child_1.evaluate(self.param.dataX, self.param.dataY)
                    child_2.evaluate(self.param.dataX, self.param.dataY)
                    children.append(child_1)
                    children.append(child_2)
                # if there is an odd number of parents, last one gets mutated
                if len(parents) > 0 and self.param.mut_rate > 0:
                    child = self.population[parents.pop()].make_copy()
                    child.mutate()
                    children.append(child)

                # running the survivor selection
                self.population = self.param.survivor_selection_method(self.param, self.population, children, non_survivors)
                num_evals += len(children)

            if self.param.verbose > 0:
                highest = sorted([x.fitness for x in self.population])[-1]
                average = np.mean([x.fitness for x in self.population])
                median = np.median([x.fitness for x in self.population])

                if int(generation % max((generations // 50), 1)) == 0:
                    print("finished generation: {}, fitness: highest {}, average {}, median {} \n".format(
                        generation, highest, average, median))

    # returns the best individuals based on fitness
    def return_best(self):
        return sorted(self.population, key=lambda x: x.fitness)[-1]
