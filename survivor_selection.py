'''
functions for survivor selection
'''

#change these to take in population, children, survivor list, and parameter, all parameters should be in the parameter class

def survivor_random_selection(pop, num_parent, rng):
    return rng.choice(pop, size = num_parent, replace = False)

def survivor_tourney_selection(param, pop, children, non_survivors):
    new_pop = [individual for index, individual in enumerate(pop) if individual not in set(non_survivors)]
    new_pop += children
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