from optproblems.base import BoundConstraintsChecker, Individual
from MultiobjectiveTest import MultiObjectiveTestProblem


class convexf1:
    """The f1 function for the convex problem"""

    def __init__(self, num_variables):
        self.num_variables = num_variables

    def __call__(self, phenome):
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(0, n):
            temp_sum += (phenome[i] - 1) ** 2
        return temp_sum


class convexf2:
    """The f2 function for the convex problem"""

    def __init__(self, num_variables):
        self.num_variables = num_variables

    def __call__(self, phenome):
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(0, n):
            temp_sum += (phenome[i] + 1) ** 2
        return temp_sum


class convexh:
    """The h function for the convex problem"""

    def __init__(self, num_variables):
        self.num_variables = num_variables

    def __call__(self, phenome):
        #print('phenome: ', phenome)
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(0, n):
            temp_sum += (phenome[i]) ** 2
        return [temp_sum - 1]


class convexBaseProblem(MultiObjectiveTestProblem):

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


class convex(convexBaseProblem):
    """The convex problem."""

    def __init__(self, num_variables=2, phenome_preprocessor=None, **kwargs):
        f1 = convexf1(num_variables)
        f2 = convexf2(num_variables)
        h = convexh(num_variables)
        self.min_bounds = [-1.5] * num_variables
        self.max_bounds = [1.5] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        convexBaseProblem.__init__(self,
                                   [f1, f2], h,
                                   num_objectives=2, num_constraints=1,
                                   phenome_preprocessor=preprocessor,
                                   **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables
