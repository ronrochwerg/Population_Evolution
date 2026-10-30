"""Regression tests for the small engine fixes required by MBP."""
from types import SimpleNamespace
import numpy as np
import pytest
from Population_Evolution import Parameters, Population, parent_random_selection
from Population_Evolution.survivor_selection import crowding_distance_assignment, fast_non_dominated_sort, survivor_NSGA_II


def test_domination_sort_has_expected_fronts():
    pop = [SimpleNamespace(fitness=f) for f in ([0, 2], [2, 0], [2, 2], [3, 3])]
    assert fast_non_dominated_sort(pop) == [[0, 1], [2], [3]]


def test_crowding_constant_objective_has_no_nan():
    pop = [SimpleNamespace(fitness=[1., 1.]) for _ in range(4)]
    distances = crowding_distance_assignment(list(range(4)), pop)
    assert not np.isnan(list(distances.values())).any()
    assert crowding_distance_assignment([], pop) == {}


def test_survivors_keep_best_and_handle_exact_capacity():
    pop = [SimpleNamespace(fitness=[float(i)]) for i in range(4)]
    params = SimpleNamespace(num_eval_per_gen=2)
    assert survivor_NSGA_II(params, pop[:2], pop[2:], []) == pop[:2]
    assert survivor_NSGA_II(params, pop[:2], [], []) == pop[:2]


def test_parameter_settings_are_independent():
    p = Parameters(np.random.default_rng(1), None, None, None, None)
    p.num_win_loss = 3
    assert p.num_win_loss == 3 and p.num_tourney == 2
    p.replacement_ts = False; p.replacement_rs = True
    assert p.replacement_rs and not p.replacement_ts


class Dummy:
    def __init__(self, param): self.fitness = None
    def initialize(self, *args, **kwargs): self.fitness = [1.]
    def make_copy(self): return Dummy(None)
    def mutate(self): pass
    def recombine(self, other): pass
    def evaluate(self, *args, **kwargs): self.fitness = [1.]


def test_odd_child_is_evaluated_even_with_crossover_only():
    p = Parameters(np.random.default_rng(1), Dummy, None, None, None)
    p.num_eval_per_gen = 3; p.num_parent_rs = 3; p.replacement_rs = False
    p.parent_selection_method = parent_random_selection
    p.mut_rate = 0; p.recomb_rate = 1; p.multi_obj = True; p.verbose = 0
    def survivors(param, pop, children, non_survivors):
        assert len(children) == 3
        assert all(c.fitness == [1.] for c in children)
        return children
    p.survivor_selection_method = survivors
    engine = Population(p); engine.initialize_run(); engine.run_evolution(1)
    p.mut_rate = p.recomb_rate = 0
    with pytest.raises(ValueError): engine.run_evolution(1)


def test_infinite_penalty_front_does_not_raise_floating_point_errors():
    pop = [SimpleNamespace(fitness=[np.inf]) for _ in range(3)]
    with np.errstate(invalid="raise"):
        distances = crowding_distance_assignment([0, 1, 2], pop)
    assert not np.isnan(list(distances.values())).any()
