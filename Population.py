'''
Population class, should store the population of the models and be able to run evolutionary stuff on it, and return fittest model...
should move the reproduction file from LGP into this library
'''
import numpy as np
from scipy.spatial import distance
from .survivor_selection import fast_non_dominated_sort
from time import perf_counter
from multiprocessing import Pool
import time

def parallel_evaluation(individual,dataX,dataY, **kwargs):
    individual.evaluate(dataX, dataY, **kwargs)
    return individual

class Population(object):

    def __init__(self, param):
        self.param = param
        self.population = []
        self.parent_selection_time = 0
        self.children_creation_time = 0
        self.evaluation_time = 0
        self.survivor_selection_time = 0

    # creates models equal to population size
    def initialize_run(self, name = True):
        for i in range(self.param.num_eval_per_gen):
            ind = self.param.model(self.param.model_param)
            if name:
                ind.initialize(self.param.dataX, self.param.dataY, weights=self.param.weights, name=i)
            else:
                ind.initialize(self.param.dataX, self.param.dataY, weights=self.param.weights)
            self.population.append(ind)

    # runs evolution on the population for given number of generations
    def run_evolution(self, generations, file = None, n_jobs=1):
        # pool = Pool(n_jobs)
        # looping through given number of generations
        for generation in range(generations):
            num_evals = 0
            # looping through the correct number of evaluations per generation
            while num_evals < self.param.num_eval_per_gen:
                # obtaining parents (and non_survivors if given by selection method)
                start = perf_counter()
                parents, non_survivors = self.param.parent_selection_method(self.param, self.population)
                self.parent_selection_time += perf_counter() - start
                children = []
                # creating children in pairs for the possibility of recombination
                waiting_array = []
                start = perf_counter()
                while len(parents) >= 2:
                    parent_1 = parents.pop()
                    parent_2 = parents.pop()

                    child_1 = self.population[parent_1].make_copy()
                    child_2 = self.population[parent_2].make_copy()

                    # children can mutate and recombine
                    change = False
                    while not change:
                        if self.param.rng.random() < self.param.recomb_rate:
                            child_1.recombine(child_2)
                            change = True
                        if self.param.rng.random() < self.param.mut_rate:
                            child_1.mutate()
                            child_2.mutate()
                            change = True


                    # giving the children fitness

                    # result = pool.apply_async(parallel_evaluation, args=(child_1, self.param.dataX, self.param.dataY),
                    #                           kwds={'weights':self.param.weights})
                    # waiting_array.append(result)
                    #
                    # result = pool.apply_async(parallel_evaluation, args=(child_2, self.param.dataX, self.param.dataY),
                    #                           kwds={'weights': self.param.weights})
                    #
                    # waiting_array.append(result)
                    child_1.evaluate(self.param.dataX, self.param.dataY, weights=self.param.weights)
                    child_2.evaluate(self.param.dataX, self.param.dataY, weights=self.param.weights)
                    children.append(child_1)
                    children.append(child_2)
                self.evaluation_time += perf_counter() - start
                # while True:
                #     time.sleep(1)
                #     # catch exception if results are not ready yet
                #     try:
                #         ready = [result.ready() for result in waiting_array]
                #         successful = [result.successful() for result in waiting_array]
                #     except Exception:
                #         continue
                #     # exit loop if all tasks returned success
                #     if all(successful):
                #         break
                #     # raise exception reporting exceptions received from workers
                #     if all(ready) and not all(successful):
                #         raise Exception(
                #             f'Workers raised following exceptions {[result._value for result in waiting_array if not result.successful()]}')
                #
                # children = [child.get() for child in waiting_array]

                # if there is an odd number of parents, last one gets mutated
                if len(parents) > 0 and self.param.mut_rate > 0:
                    child = self.population[parents.pop()].make_copy()
                    child.mutate()
                    children.append(child)
                start = perf_counter()
                # running the survivor selection
                self.population = self.param.survivor_selection_method(self.param, self.population, children, non_survivors)
                self.survivor_selection_time += perf_counter() - start
                num_evals += len(children)

            if not self.param.multi_obj:
                fitness = [x.fitness for x in self.population]
                best = sorted(fitness, reverse=self.param.fitness_maximized)[0]
                if self.param.early_stop and best < self.param.early_stop:
                    print("Stopping evolution since best fitness is better than {}".format(self.param.early_stop),
                          file=file)
                    return
            if self.param.verbose > 0:
                if file:
                    if int((generation+1) % max((generations // 25), 1)) == 0:
                        print("finished generation: {}".format(generation+1))
                if not self.param.multi_obj:
                    average = np.mean(fitness)
                    median = np.median(fitness)

                    if int((generation+1) % max((generations // 50), 1)) == 0:
                        print("finished generation: {}, fitness: best {}, average {}, median {} \n".format(
                            generation+1, best, average, median), file=file)

                else:
                    fronts = fast_non_dominated_sort(self.population)
                    num_fronts = len(fronts)
                    num_non_dom = len(fronts[0])
                    non_dom = fronts[0]
                    best_fit_per_obj = []
                    for i in range(len(self.population[0].fitness)):
                        # sorting based on current objective
                        non_dom.sort(key=lambda x: self.population[x].fitness[i])
                        # dont take duplicate solutions
                        best_fit_per_obj.append(self.population[non_dom[0]].fitness)
                    if int((generation+1) % max((generations // 25), 1)) == 0:
                        print("finished generation: {}, number of fronts: {}, number of non-dominated solutions {}".format(
                            generation+1, num_fronts, num_non_dom), file=file)
                        print("Best fitness for each objective: {} \n".format(
                            repr([('obj ' + repr(i), best_fit_per_obj[i]) for i in range(len(best_fit_per_obj))])), file=file)


    # returns the best individuals based on fitness (single objective)
    def return_best(self):
        return sorted(self.population, key=lambda x: x.fitness, reverse=self.param.fitness_maximized)[0]

    # returns the best individuals for multi-objective (assumes objectives are to be minimized)
    # can specify k value to look for specific k value for clustering
    def return_best_multi(self, k_min=None, k_max=None, min_found_k=False, file=None, chosen_obj=None):

        non_dominated_sol = fast_non_dominated_sort(self.population)[0]

        k_vals = sorted([self.population[i].num_clusters for i in non_dominated_sol])
        print("Occurrences of k values: ", {k_val:k_vals.count(k_val) for k_val in k_vals}, file=file)

        # only using solutions with k clusters if k is specified
        if k_min and k_max:
            specific_k_sol = [i for i in non_dominated_sol if k_min <= self.population[i].num_clusters <= k_max]
            if len(specific_k_sol) > 0:
                non_dominated_sol = specific_k_sol
            elif min_found_k:
                print(
                    "returning best solutions of minimum number of clusters {}".format(
                        min(k_vals)), file=file)
                non_dominated_sol = [i for i in non_dominated_sol if self.population[i].num_clusters == min(k_vals)]
            else:
                print("No non-dominated solutions have between {} and {} clusters, returning best solutions of other clusters".format(k_min, k_max), file=file)
                k_min = min(k_vals)
                k_max = max(k_vals)
        elif min_found_k:
            non_dominated_sol = [i for i in non_dominated_sol if self.population[i].num_clusters == min(k_vals)]

        if chosen_obj is None:
            # making sure there are enough solutions in the non-dominated front
            if len(non_dominated_sol) <= len(self.population[0].fitness) + 1:
                print("All solutions returned since non-dominated front is smaller than number of objectives", file=file)
                return [self.population[i] for i in non_dominated_sol]

        chosen_solutions = []
        chosen_solutions_ind = []
        # choosing the solutions with the lowest individual objectives
        if chosen_obj is None:
            for i in range(len(self.population[0].fitness)):
                # sorting based on current objective
                non_dominated_sol.sort(key=lambda x: self.population[x].fitness[i], reverse=self.param.fitness_maximized)
                # dont take duplicate solutions
                if non_dominated_sol[0] not in chosen_solutions_ind:
                    chosen_solutions_ind.append(non_dominated_sol[0])
                    chosen_solutions.append(self.population[non_dominated_sol[0]])

            # creating a matrix of the fitness
            fitness_matrix = [self.population[i].fitness for i in non_dominated_sol]

            # finding the squared euclidean distances between all the fitness and finding the one with the smallest total sum
            distances = np.sum(distance.cdist(fitness_matrix, fitness_matrix, 'sqeuclidean'), axis=0)
            smallest_dist = np.argmin(distances)
            # adding it to the returned solutions if they are not already chosen
            if smallest_dist not in chosen_solutions_ind:
                chosen_solutions.append(self.population[non_dominated_sol[smallest_dist]])

        else:
            if isinstance(chosen_obj, list):
                for i in chosen_obj:
                    non_dominated_sol.sort(key=lambda x: self.population[x].fitness[i], reverse=self.param.fitness_maximized)
                    # dont take duplicate solutions
                    if non_dominated_sol[0] not in chosen_solutions_ind:
                        chosen_solutions_ind.append(non_dominated_sol[0])
                        chosen_solutions.append(self.population[non_dominated_sol[0]])
            else:
                non_dominated_sol.sort(key=lambda x: self.population[x].fitness[chosen_obj],
                                       reverse=self.param.fitness_maximized)
                chosen_solutions.append(self.population[non_dominated_sol[0]])

        # else: # give back best average solutions for each k value
        #     chosen_solutions = {}
        #     for k_val in range(k_min, k_max + 1):
        #         # grab all the solutions with the given k value of cluster and build a similarity matrix between points
        #         current_k_sol = [sol for sol in non_dominated_sol if self.population[sol].num_clusters == k_val]
        #         sim_matrix = np.zeros((len(self.param.dataY), len(self.param.dataY)), dtype=int)
        #         for sol in current_k_sol:
        #             for point in range(sim_matrix.shape[0]):
        #                 for other in range(point+1, sim_matrix.shape[0]):
        #                     if self.population[sol].clustering[point] == self.population[sol].clustering[other]:
        #                         sim_matrix[point,other] += 1
        #                         sim_matrix[other,point] += 1
        #         sim_matrix = sim_matrix/len(current_k_sol) # normalize the matrix, values are between 0 and 1
        #
        #         # get an average clustering using the similarity matrix
        #         cur_cluster = 1
        #         average_cluster = np.full(len(self.param.dataY), -1)
        #         for point in range(len(average_cluster)):
        #             point_sim = sim_matrix[point, :] # similarity of the current point to all other points
        #             # go through the other points from the highest similarity to the lowest
        #             for indx in np.argsort(point_sim)[::-1]:
        #                 if average_cluster[indx] != -1: # if the point has already been assigned a cluster, give the current point the same cluster
        #                     average_cluster[point] = average_cluster[indx]
        #                     break
        #                 elif point_sim[indx] < threshold: # if we have reached a point where similarity is below 0.5, start a new cluster
        #                     average_cluster[point] = cur_cluster
        #                     cur_cluster += 1
        #                     break
        #         chosen_solutions[k_val] = (average_cluster, len(current_k_sol), cur_cluster-1)
        return chosen_solutions







    def print_parameters(self, file = None):
        self.param.print_attributes(file = file)
