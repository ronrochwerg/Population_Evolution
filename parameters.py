'''
class Parameters for the population: pop size, rates... (all the inputs to the iris test run regular function) (not data, that should be in the population class itself
'''

from .parent_selection import parent_tourney_selection
from .survivor_selection import survivor_tourney_selection

class Parameters:

    def __init__(self, rng, model, model_param, dataX, dataY, weights=None):
        #rng
        self.rng = rng

        #type of model in population
        self.model = model
        self.model_param = model_param

        #whether fitness of the model is being maximized or minimized
        self.fitness_maximized = False

        #functions for parent and survivor selection
        self.parent_selection_method = parent_tourney_selection
        self.survivor_selection_method = survivor_tourney_selection

        #number of children per generation (and starting population size)
        self._num_eval_per_gen = 1000

        #mutation chance
        self.mut_rate = 1

        #recombination chance
        self.recomb_rate = 0.5

        #data features we are using
        self.dataX = dataX

        #data we are trying to model/predict
        self.dataY = dataY

        #weights for the data points
        if weights is None:
            self.weights = [1] * len(dataY)
        else:
            self.weights = weights

        #random selection parameters
        self._num_parent_rs = 1000
        self._replacement_rs = True

        #tourney selection parameters
        self._tourney_size = 10
        self._replacement_ts = False
        self._num_tourney = 2 # number of tournaments run at once
        self._num_win_loss = 1 # number of winners and loser per tournament

        # how much the population will output while running, 0 is none, 1 is some
        self.verbose = 1

    @property
    def num_eval_per_gen(self):
        return self._num_eval_per_gen

    @num_eval_per_gen.setter
    def num_eval_per_gen(self,value):
        self._num_eval_per_gen = value
        if self.num_parent_rs > value:
            print("Warning: setting random selection number of parents to new evals per generation")
            self.num_parent_rs = value
        if self._tourney_size * self._num_tourney > value:
            self._tourney_size = value//self._num_tourney
            print("Warning: setting tourney selection tourney size to {} to allow for tourney selection".format(value//self._num_tourney))

    @property
    def tourney_size(self):
        return self._tourney_size

    @tourney_size.setter
    def tourney_size(self,value):
        self._tourney_size = value
        if self._num_eval_per_gen < self._tourney_size * self._num_tourney and not self._replacement_ts:
            print("Warning: setting evals per generation to size needed for tournaments")
            self._num_eval_per_gen = self._tourney_size * self._num_tourney

    @property
    def num_tourney(self):
        return self._num_tourney

    @num_tourney.setter
    def num_tourney(self,value):
        self._num_tourney = value
        if self._num_eval_per_gen < self._tourney_size * self._num_tourney and not self._replacement_ts:
            print("Warning: setting evals per generation to size needed for tournaments")
            self._num_eval_per_gen = self._tourney_size * self._num_tourney
        if self._num_eval_per_gen < self._num_tourney * self.num_win_loss:
            print("Warning: setting evals per generation to size needed for tournaments")
            self._num_eval_per_gen = self._num_tourney * self.num_win_loss

    @property
    def num_win_loss(self):
        return self._num_win_loss

    @num_win_loss.setter
    def num_win_loss(self,value):
        self._num_tourney = value
        if self._num_eval_per_gen < self._num_tourney * self._num_win_loss:
            print("Warning: setting evals per generation to size needed for tournaments")
            self._num_eval_per_gen = self._num_tourney * self._num_win_loss
        if self._tourney_size < 2 * self._num_win_loss:
            print("Warning: setting tourney size to value needed for number of wins and losses")

    @property
    def replacement_ts(self):
        return self._replacement_ts

    @replacement_ts.setter
    def replacement_ts(self, value):
        self._replacement_ts = value
        if self._num_eval_per_gen < self._tourney_size * self._num_tourney and not self._replacement_ts:
            print("Warning: setting evals per generation to size needed for tournaments")
            self._num_eval_per_gen = self._tourney_size * self._num_tourney

    @property
    def num_parent_rs(self):
        return self._num_parent_rs

    @num_parent_rs.setter
    def num_parent_rs(self,value):
        self._num_parent_rs = value
        if self._num_eval_per_gen < self._num_parent_rs and not self._replacement_ts:
            print("Warning: setting evals per generation to size needed for random selection")
            self._num_eval_per_gen = self.num_parent_rs

    @property
    def replacement_rs(self):
        return self._replacement_ts

    @replacement_rs.setter
    def replacement_rs(self, value):
        self._replacement_rs = value
        if self._num_eval_per_gen < self._num_parent_rs and not self._replacement_ts:
            print("Warning: setting evals per generation to size needed for random selection")
            self._num_eval_per_gen = self._num_parent_rs

    def print_attributes(self, file=None):
        for k,v in vars(self).items():
            print('{}: \t {}'.format(k,v), file=file)