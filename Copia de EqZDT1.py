import math

from BASE import BoundConstraintsChecker, Individual
from MultiObj import MultiObjectiveTestProblem


class EQZDT1_f2:
    """The f2 function for all CZDT problems."""

    def __init__(self, f1, g, h):
        self.f1 = f1
        self.g = g
        self.h = h

    def __call__(self, phenome):
        g_value = self.g(phenome)
        f1_value = self.f1(phenome)
        print('g', g_value)
        print('f1', f1_value)
        print('f2', g_value * self.h(f1_value, g_value))
        return g_value * self.h(f1_value, g_value)


def EQZDT1_f1(phenome):
    """The f1 function for CZDT1, CZDT2, CZDT3 and CZDT4."""
    return phenome[0]


def EQZDT1_constraint(phenome):
    h1 = (phenome[0]-0.5)*(phenome[0]-0.5)
    h2 = (phenome[1]-0.4)*(phenome[1]-0.4)
    print('h1', h1, 'h2', h2)
    print('const', h1+h2-0.25)
    return h1+h2-0.25


class EQZDT1_g:
    """The g function for CZDT1, CZDT2 and CZDT3."""

    def __init__(self, num_variables):
        self.num_variables = num_variables

    def __call__(self, phenome):
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(1, n):
            temp_sum += (phenome[i]**2) / float(n - 1)
        return 1.0 + (9.0 * temp_sum)


class EQZDT1BaseProblem(MultiObjectiveTestProblem):

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


class EQ1ZDT1(EQZDT1BaseProblem):
    """The CZDT1 problem."""

    def __init__(self, num_variables=3, phenome_preprocessor=None, **kwargs):
        f2 = EQZDT1_f2(EQZDT1_f1, EQZDT1_g(num_variables), self.h)
        print('f2', f2)
        self.min_bounds = [-0.3] * num_variables
        self.min_bounds[0] = 0.0
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        EQZDT1BaseProblem.__init__(self,
                                   [EQZDT1_f1, f2], [EQZDT1_constraint],
                                   num_objectives=2, num_constraints=1,
                                   phenome_preprocessor=preprocessor,
                                   **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT1."""
        print('f1, g = ', f1_value, g_value)
        print('sqt = ', f1_value/g_value)
        h = 2.0 - math.sqrt(f1_value / g_value)
        print('h= ', h)
        return h

class EQ2ZDT1(EQZDT1BaseProblem):
    """The CZDT1 problem."""

    def __init__(self, num_variables=3, phenome_preprocessor=None, **kwargs):
        f2 = EQZDT1_f2(EQZDT1_f1, EQZDT1_g(num_variables), self.h)
        print('f2', f2)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        EQZDT1BaseProblem.__init__(self,
                                   [EQZDT1_f1, f2], [EQZDT1_constraint],
                                   num_objectives=2, num_constraints=1,
                                   phenome_preprocessor=preprocessor,
                                   **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables

    @staticmethod
    def h(f1_value, g_value):
        """The h function of CZDT1."""
        print('f1, g = ', f1_value, g_value)
        print('sqt = ', f1_value/g_value)
        h = 2.0 - math.sqrt(f1_value / g_value)
        print('h= ', h)
        return h

class EQZDT1(list):
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
        list.__init__(self, [EQ1ZDT1(**kwargs),
                             EQ2ZDT1(**kwargs)])
