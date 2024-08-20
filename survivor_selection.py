import numpy as np
'''
functions for survivor selection
'''

#change these to take in population, children, survivor list, and parameter, all parameters should be in the parameter class
def dominates(ind, other):
    return np.all(np.less_equal(ind, other)) and np.any(np.less(ind, other))
# fast-non-dominated-sort from NSGA-II paper (slightly reconfigured) (smaller front is better, less dominated)
def fast_non_dominated_sort(pop):
    # variables to keep track of domination data and non-dominated fronts
    dom_data = [[0,[]] for _ in range(len(pop))]
    fronts = [[]]

    # go through each pair of individuals and update their domination with respect to each other
    for ind in range(len(pop)):
        for other in range(ind+1,len(pop)):
            if dominates(pop[ind].fitness, pop[other].fitness):
                dom_data[ind][1].append(other)
                dom_data[other][0] += 1
            elif dominates(pop[other].fitness, pop[ind].fitness):
                dom_data[other][1].append(ind)
                dom_data[ind][0] += 1
        # adding individuals who are not dominated to the first front
        if dom_data[ind][0] == 0:
            fronts[0].append(ind)
            # not keeping track of individuals rank, since not using rank based parent selection
    # going until the next front is empty
    while fronts[-1]:
        next_front = []
        for ind in fronts[-1]:
            for other in dom_data[ind][1]:
                dom_data[other][0] -= 1
                if dom_data[other][0] == 0:
                    next_front.append(other)
        fronts.append(next_front)
    #returning all fronts but the last one (it is empty)
    return fronts[:-1]

# crowding-distance-assignment from NSGA-II paper (larger distance is better since it means less crowded)
def crowding_distance_assignment(front, pop):
    distances = dict.fromkeys(front, 0)

    for i in range(len(pop[0].fitness)):
        # sorting based on current objective
        front.sort(key=lambda x: pop[x].fitness[i])
        # setting best and worst distances to inf
        distances[front[0]] = distances[front[-1]] = float('inf')
        norm = pop[front[-1]].fitness[i] - pop[front[0]].fitness[i]
        # adding the absolute normalized distance to all other points
        for ind in range(1, len(front) - 1):
            distances[front[ind]] += (pop[front[ind+1]].fitness[i] - pop[front[ind-1]].fitness[i]) / norm

    return distances



def survivor_random_selection(param, pop, children, non_survivors):
    new_pop = [individual for index, individual in enumerate(pop) if index not in set(non_survivors)]
    new_pop += children
    return param.rng.choice(new_pop, size = len(pop), replace = False)

def survivor_tourney_selection(param, pop, children, non_survivors):
    new_pop = [individual for index, individual in enumerate(pop) if index not in set(non_survivors)]
    new_pop += children
    return new_pop

# The NSGA-II form of survivor selection using non-dominated sorting and crowding distance
# this does not calculate the front assignment and crowding distance for individuals in the population that is normally
# needed for the binary tournament parents selection described in the paper. the reasoning is that this survivor
# selection is already very elitist and so the parent selection currently used with it is simply to randomly arrange all
# parents and have them all produce offspring
def survivor_NSGA_II(param, pop, children, non_survivors):
    temp_pop = pop + children
    fronts = fast_non_dominated_sort(temp_pop)
    new_pop = []

    cur_front = 0
    while len(new_pop) + len(fronts[cur_front]) <= param.num_eval_per_gen:
        new_pop += [temp_pop[i] for i in fronts[cur_front]]
        cur_front += 1
    if len(new_pop) < param.num_eval_per_gen:
        crowd_dis = crowding_distance_assignment(fronts[cur_front], temp_pop)
        sorted_ind = sorted(crowd_dis, key=crowd_dis.get, reverse=True)
        new_pop += [temp_pop[i] for i in sorted_ind[:param.num_eval_per_gen - len(new_pop)]]

    return new_pop


# def survivor_lexicase(pop, samples, target, num_return = 1):
#     predictions = []
#     for i, individual in enumerate(pop):
#         prediction = individual.predict(samples)
#         predictions.append([target - prediction,i])
#
#     for i in range(len(samples)):
#         min_err = min(predictions, key=lambda x: abs(x[0][i]))[0][i]
#         predictions = [x for x in predictions if x[0][i] <= min_err]
#         if len(predictions) == num_return:
#             return pop[predictions[0][1]]
#     return pop[predictions[0][1]]