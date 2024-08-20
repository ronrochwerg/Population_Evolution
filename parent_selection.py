'''
functions for parent selection
'''

#change these to take in population and parameter, all parameters should be in the parameter class
#functions should return list of children and list of non survivors (if not picked then empty list)

def parent_random_selection(param, pop):
    return list(param.rng.choice(len(pop), size = param.num_parent_rs, replace = param.replacement_rs)), []

# tournament selection, creates two tourneys, best fitness ind mutate or recombine to replace the losers in each tourney
# returns the indices of the winners and losers in the original population
def parent_tourney_selection(param, pop):

    #randomly select tourney individuals without replacement (for both tourneys)
    selected = param.rng.choice(len(pop), size =param.num_tourney * param.tourney_size, replace=param.replacement_ts)

    tourneys = []

    for i in range(param.num_tourney):
        tourneys.append(sorted(selected[i*param.tourney_size:(i+1)*param.tourney_size], key=lambda x: pop[x].fitness, reverse=param.fitness_maximized))
    # split the individuals selected into two tourneys and sort them based on fitness
    # tourney1 = sorted(selected[:param.tourney_size], key=lambda x: pop[x].fitness, reverse=True)
    # tourney2 = sorted(selected[param.tourney_size:], key=lambda x: pop[x].fitness, reverse=True)

    #save winners and losers
    winners = []
    losers = []
    for tourney in tourneys:
        winners += tourney[:param.num_win_loss]
        losers += tourney[-1*param.num_win_loss:]

    return winners, losers


#Lexicase selection (needs work)
# def parent_lexicase(pop, samples, target, num_return = 1):
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