import multiprocessing.dummy

from BASE import ResourcesExhausted


class Problem(object):
    """The base class for problems to be solved.

    In the simplest case you can use this class directly by wrapping
    a single objective function or a list of objective functions. For
    more sophisticated cases, creating a subclass may be necessary.

    """

    def __init__(self, objective_functions, constraint_functions=[],
                 num_objectives=None,
                 num_constraints=0,
                 max_evaluations=float("inf"),
                 worker_pool=None,
                 mp_module=None,
                 phenome_preprocessor=None,
                 name=None):
        """Constructor.

        Parameters
        ----------
        objective_functions : callable or sequence of callables
            If this argument is simply a function, it is taken 'as-is'.
            If a sequence of callables is provided, these are wrapped in a
            :class:`BundledObjectives <optproblems.base.BundledObjectives>`
            helper object, so that a single function call returns a list of
            objective values.
        num_objectives : int, optional
            The number of objectives. If omitted, this number is guessed
            from the number of objective functions.
        max_evaluations : int, optional
            The maximum budget of function evaluations. By default there
            is no restriction.
        worker_pool : Pool, optional
            A pool of worker processes. Default is None (no
            parallelization).
        mp_module : module, optional
            Either `multiprocessing`, `multiprocessing.dummy` (default),
            or a `MockMultiProcessing` instance. This is only used to create
            an internal lock around bookkeeping code in various places. The
            lock is only required for asynchronous parallelization, but not
            for the parallelization with a worker pool in
            :func:`batch_evaluate`.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned. When
            this pre-processing raises an exception, no function
            evaluations are counted. By default, no pre-processing is
            applied.
        name : str, optional
            A nice name for humans to read.

        """
        try:
            iter(objective_functions)
            # succeeded, so several functions are given
            one_function = BundledObjectives(objective_functions)
            if num_objectives is None:
                # guess the number of objectives
                num_objectives = len(objective_functions)
            assert num_objectives >= len(objective_functions)
        except TypeError:
            one_function = objective_functions
            if num_objectives is None:
                num_objectives = 1
            assert num_objectives > 0
        try:
            iter(constraint_functions)
            # succeeded, so several functions are given
            one_constraint = BundledConstraints(constraint_functions)
            if constraint_functions is None:
                # guess the number of constraints
                num_constraints = 0
            assert num_constraints >= 0
        except TypeError:
            one_constraint = constraint_functions
            if num_constraints is None:
                num_constraints = 0
            assert num_constraints >= 0
        self.objective_function = one_function
        self.num_objectives = num_objectives
        self.constraint_function = one_constraint
        self.num_constraints = num_constraints
        self.remaining_evaluations = max_evaluations
        self.consumed_evaluations = 0
        self.worker_pool = worker_pool
        self.chunksize = 1
        if mp_module is None:
            mp_module = multiprocessing.dummy
        self.mp_module = mp_module
        self.lock = mp_module.Manager().Lock()
        if phenome_preprocessor is None:
            phenome_preprocessor = identity
        self.phenome_preprocessor = phenome_preprocessor
        self.name = name

    def __str__(self):
        """Return the name of this problem."""
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__

    def __call__(self, phenome):
        """Evaluate a solution and return objective values.

        Also checks budget and counts the evaluation.

        Raises
        ------
        ResourcesExhausted
            If the budget of function evaluations is exhausted.

        """
        phenome = self.phenome_preprocessor(phenome)
        with self.lock:
            if self.remaining_evaluations > 0:
                self.consumed_evaluations += 1
                self.remaining_evaluations -= 1
            else:
                raise ResourcesExhausted("problem evaluations")
        objective_values = self.objective_function(phenome)
        try:
            num_obj_values = len(objective_values)
        except TypeError:
            num_obj_values = 1
        assert num_obj_values == self.num_objectives

        constraint_violation = self.constraint_function(phenome)
        try:
            num_constraint_values = len(constraint_violation)
        except TypeError:
            num_constraint_values = 1
        assert num_constraint_values == self.num_constraints

        return objective_values, constraint_violation

    def evaluate(self, individual):
        """Evaluate an individual.

        This method delegates the evaluation of the phenome to
        :func:`__call__ <optproblems.base.Problem.__call__>` and directly
        writes the objective values into the individual's corresponding
        attribute.

        """
        individual.objective_values, individual.constraint_violation = self.__call__(individual.phenome)

    def batch_evaluate(self, individuals):
        """Evaluate a batch of individuals.

        Objective values are written directly into the individuals'
        corresponding attributes.

        Raises
        ------
        ResourcesExhausted
            If the budget is not sufficient to evaluate all individuals,
            this exception is thrown.

        """
        if self.worker_pool is None or len(individuals) == 1:
            evaluate = self.evaluate
            for individual in individuals:
                evaluate(individual)
        else:
            preprocessor = self.phenome_preprocessor
            num_objectives = self.num_objectives
            with self.lock:
                budgeted_evaluations = min(len(individuals), self.remaining_evaluations)
                affordable_individuals = individuals[:budgeted_evaluations]
                phenomes = [preprocessor(ind.phenome) for ind in affordable_individuals]
                self.consumed_evaluations += budgeted_evaluations
                self.remaining_evaluations -= budgeted_evaluations
            results = self.worker_pool.map(self.objective_function,
                                           phenomes,
                                           chunksize=self.chunksize)
            for individual, objective_values in zip(individuals, results):
                try:
                    num_obj_values = len(objective_values)
                except TypeError:
                    num_obj_values = 1
                assert num_obj_values == num_objectives
                individual.objective_values = objective_values
            if len(individuals) > len(affordable_individuals):
                raise ResourcesExhausted("problem evaluations")


class TestProblem(Problem):
    """Abstract base class for artificial test problems."""

    def get_optimal_solutions(self, max_number=None):
        """Return globally optimal or Pareto-optimal solutions.

        This is an abstract method. Implementations must be deterministic.
        In the multi-objective case, the generated solutions should be
        evenly distributed over the whole Pareto-set.

        """
        raise NotImplementedError("Optimal solutions are unknown.")

    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions.

        This is an abstract method. Implementations must be deterministic.
        This method should be most useful for single-objective, continuous
        problems.

        """
        raise NotImplementedError("Locally optimal solutions are unknown.")


class BundledObjectives:
    """Helper class to let several distinct functions appear as one."""

    def __init__(self, objective_functions):
        self.objective_functions = objective_functions

    def __call__(self, phenome):
        """Collect objective values from objective functions.

        Objective values are returned as flattened list.

        """
        returned_values = []
        for objective_function in self.objective_functions:
            returned_values.append(objective_function(phenome))
        flattened = []
        for returned_value in returned_values:
            try:
                iter(returned_value)
                # succeeded, so this function returned several values at once
                flattened.extend(returned_value)
            except TypeError:
                flattened.append(returned_value)
        return flattened


class BundledConstraints:
    """Helper class to let several distinct functions appear as one."""

    def __init__(self, constraint_functions):
        self.constraint_functions = constraint_functions

    def __call__(self, phenome):
        """Collect objective values from objective functions.

        Objective values are returned as flattened list.

        """
        returned_values = []
        for constraint_function in self.constraint_functions:
            returned_values.append(constraint_function(phenome))
        flattened = []
        for returned_value in returned_values:
            try:
                iter(returned_value)
                # succeeded, so this function returned several values at once
                flattened.extend(returned_value)
            except TypeError:
                flattened.append(returned_value)
        return flattened
