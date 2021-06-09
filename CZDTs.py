"""
This module contains the CZDT suite for multiobjective optimization as defined
in [Zitzler2000]_. The problems have two objectives each. Except for CZDT5,
the search spaces are continuous.
"""

import math

from BASE import BoundConstraintsChecker, Individual
from MultiObj import MultiObjectiveTestProblem


class CZDT_f2:
    """The f2 function for all CZDT problems."""

    def __init__(self, f1, g, h):
        self.f1 = f1
        self.g = g
        self.h = h

    def __call__(self, phenome):
        g_value = self.g(phenome)
        f1_value = self.f1(phenome)
        return g_value * self.h(f1_value, g_value)


def CZDT1to4_f1(phenome):
    """The f1 function for CZDT1, CZDT2, CZDT3 and CZDT4."""
    return phenome[0]


def CZDT1to3and6_constraint(phenome):
    n = len(phenome)
    temp_sum: float = 0.0
    for i in range(1, n):
        temp_sum += math.sin(phenome[i])
    return temp_sum


def CZDT4_constraint(phenome):
    n = len(phenome)
    temp_sum: float = 0.0
    for i in range(1, n):
        temp_sum += math.pow(phenome[i], 2)
    return temp_sum


class CZDT1to3_g:
    """The g function for CZDT1, CZDT2 and CZDT3."""

    def __init__(self, num_variables):
        self.num_variables = num_variables

    def __call__(self, phenome):
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(1, n):
            temp_sum += phenome[i] / float(n - 1)
        return 1.0 + (9.0 * temp_sum)


class CZDTBaseProblem(MultiObjectiveTestProblem):

    def get_optimal_solutions(self, max_number=100):
        """Return Pareto-optimal solutions.

        .. note:: The returned solutions do not yet contain the objective
            values.

        Parameters
        ----------
        max_number : int, optional
            As the number of Pareto-optimal solutions is infinite, the
            returned set has to be restricted to a finite sample.

        Returns
        -------
        solutions : list of Individual
            The Pareto-optimal solutions

        """
        assert max_number > 1
        solutions = []
        for i in range(max_number):
            phenome = [0.0] * self.num_variables
            phenome[0] = float(i) / float(max_number - 1)
            solutions.append(Individual(phenome))
        return solutions


class CZDT1(CZDTBaseProblem):
    """The CZDT1 problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        f2 = CZDT_f2(CZDT1to4_f1, CZDT1to3_g(num_variables), self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        CZDTBaseProblem.__init__(self,
                                 [CZDT1to4_f1, f2], [CZDT1to3and6_constraint],
                                 num_objectives=2, num_constraints=1,
                                 phenome_preprocessor=preprocessor,
                                 **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT1."""
        return 1.0 - math.sqrt(f1_value / g_value)


class CZDT2(CZDTBaseProblem):
    """The CZDT2 problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        f2 = CZDT_f2(CZDT1to4_f1, CZDT1to3_g(num_variables), self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        CZDTBaseProblem.__init__(self,
                                 [CZDT1to4_f1, f2], [CZDT1to3and6_constraint],
                                 num_objectives=2, num_constraints=1,
                                 phenome_preprocessor=preprocessor,
                                 **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT2."""
        return 1.0 - (f1_value / g_value) ** 2


class CZDT3(CZDTBaseProblem):
    """The CZDT3 problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        f2 = CZDT_f2(CZDT1to4_f1, CZDT1to3_g(num_variables), self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        CZDTBaseProblem.__init__(self,
                                 [CZDT1to4_f1, f2], [CZDT1to3and6_constraint],
                                 num_objectives=2, num_constraints=1,
                                 phenome_preprocessor=preprocessor,
                                 **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT3."""
        fraction = f1_value / g_value
        return_value = 1.0 - math.sqrt(fraction)
        return_value -= fraction * math.sin(10.0 * math.pi * f1_value)
        return return_value


class CZDT4(CZDTBaseProblem):
    """The CZDT4 problem."""

    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        f2 = CZDT_f2(CZDT1to4_f1, self.g, self.h)
        self.min_bounds = [-5.0] * num_variables
        self.min_bounds[0] = 0.0
        self.max_bounds = [5.0] * num_variables
        self.max_bounds[0] = 1.0
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        CZDTBaseProblem.__init__(self,
                                 [CZDT1to4_f1, f2],[CZDT4_constraint],
                                 num_objectives=2, num_constraints=1,
                                 phenome_preprocessor=preprocessor,
                                 **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT4."""
        return 1.0 - math.sqrt(f1_value / g_value)

    def g(self, phenome):
        """The g function of CZDT4."""
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        four_pi = 4 * math.pi
        for i in range(1, n):
            x = phenome[i]
            temp_sum += x ** 2 - 10.0 * math.cos(four_pi * x)
        return 1.0 + 10.0 * (n - 1) + temp_sum

class CZDT6(CZDTBaseProblem):
    """The CZDT6 problem."""

    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        f2 = CZDT_f2(self.f1, self.g, self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        CZDTBaseProblem.__init__(self,
                                 [self.f1, f2], [CZDT1to3and6_constraint],
                                 num_objectives=2, num_constraints=1,
                                 phenome_preprocessor=preprocessor,
                                 **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT6."""
        return 1.0 - (f1_value / g_value) ** 2

    @staticmethod
    def f1(phenome):
        """The f1 function of CZDT6."""
        x = phenome[0]
        return 1.0 - math.exp(-4.0 * x) * math.pow(math.sin(6.0 * math.pi * x), 6.0)

    def g(self, phenome):
        """The g function of CZDT6."""
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(1, n):
            temp_sum += phenome[i]
        return 1.0 + (9.0 * math.pow(temp_sum / (n - 1), 0.25))


class CZDT(list):
    """The whole collection.

     This class inherits from :class:`list` and by default generates all
     problems with their default configuration.

    Parameters
    ----------
    kwargs
        Arbitrary keyword arguments, passed through to the constructors of
        the CZDT problems.

    References
    ----------
    .. [Zitzler2000] Zitzler, E., Deb, K., and Thiele, L. (2000).
        Comparison of Multiobjective Evolutionary Algorithms: Empirical
        Results. Evolutionary Computation 8(2).

    """

    def __init__(self, **kwargs):
        list.__init__(self, [CZDT1(**kwargs),
                             CZDT2(**kwargs),
                             CZDT3(**kwargs),
                             CZDT4(**kwargs),
                             CZDT6(**kwargs)])
