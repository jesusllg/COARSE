"""This module contains evolutionary algorithms and some related classes.

:class:`EvolutionaryAlgorithm` is the base class for all other algorithms.

"""
import random
import threading
import array
import math

from optproblems import ResourcesExhausted, ScalingPreprocessor, Stalled
from optproblems import BoundConstraintsRepair, Problem

from evoalgos.individual import CMSAIndividual
from evoalgos.reproduction import ESReproduction
from evoalgos.selection import HyperVolumeContributionSelection
from evoalgos.selection import TruncationSelection, BackwardElimination
from evoalgos.sorting import CrowdingDistanceSorting



class Observable(object):
    """Part of the Observer/Observable design pattern."""

    def __init__(self):
        self.observers = []


    def attach(self, observer):
        """Add observer to the list of observers.

        Parameters
        ----------
        observer : callable
            The object to be informed about changes.

        """
        if observer not in self.observers:
            self.observers.append(observer)


    def detach(self, observer):
        """Remove observer from the list of observers.

        Parameters
        ----------
        observer : callable
            The object to be informed about changes.

        """
        try:
            self.observers.remove(observer)
        except ValueError:
            pass


    def notify_observers(self):
        """Inform the observers about a potential state change."""
        for observer in self.observers:
            observer(self)



class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"


    def __dir__(self):
        return list(self.keys())



class EvolutionaryAlgorithm(Observable):
    """A modular evolutionary algorithm.

    Apart from the arguments provided in the constructor, this class
    possesses the potentially useful member attributes
    `remaining_generations`, `generation`, and `last_termination`. The
    latter attribute stores the exception instance that caused the last
    termination.

    """
    def __init__(self, problem,
                 start_population,
                 population_size,
                 num_offspring,
                 max_age,
                 reproduction,
                 selection,
                 max_generations=float("inf"),
                 archive=None,
                 verbosity=1,
                 lock=None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        problem : optproblems.Problem
            An optimization problem.
        start_population : list of Individual
            A list of individuals.
        population_size : int
            The number of individuals that will survive the selection step
            in each generation.
        num_offspring : int
            The number of individuals born in every generation.
        max_age : int
            A maximum number of generations an individual can live. This
            number may be exceeded if not enough offspring is generated to
            reach the population_size.
        reproduction : Reproduction
            A :class:`Reproduction<evoalgos.reproduction.Reproduction>`
            object selecting the parents for mating and creating the
            offspring.
        selection : Selection
            A :class:`Selection<evoalgos.selection.Selection>` object
            carrying out the survivor selection.
        max_generations : int, optional
            A potential budget restriction on the number of generations.
            Default is unlimited.
        archive : list of Individual, optional
            Individuals that may influence the survivor selection. This data
            structure is by default not modified by the evolutionary
            algorithm.
        verbosity : int, optional
            A value of 0 means quiet, 1 means some information is printed
            to standard out on start and termination of this algorithm.
        lock : threading.Lock, optional
            A mutex protecting all read and write accesses to the
            population. This is necessary for asynchronous parallelization
            of the EA.
            See the :ref:`parallelization example <parallelization>`.

        """
        Observable.__init__(self)
        self.name = None
        self.problem = problem
        self.population = start_population
        assert len(start_population) > 0
        self.population_size = population_size  # mu
        assert population_size > 0
        if max_age is None:
            max_age = float("inf")
        assert max_age > 0
        self.max_age = max_age  # kappa
        self.num_offspring = num_offspring  # lambda
        assert num_offspring > 0
        self.offspring = []
        self.rejected = []
        self.deceased = []
        self.reproduction = reproduction
        self.selection = selection
        self.remaining_generations = max_generations
        self.generation = 0
        if archive is None:
            archive = []
        self.archive = archive
        self.last_termination = None
        self.verbosity = verbosity
        if lock is None:
            lock = threading.Lock()
        self.lock = lock
        if kwargs and verbosity:
            str_kwargs = ", ".join(map(str, kwargs.keys()))
            message = "Warning: EvolutionaryAlgorithm.__init__() got unexpected keyword arguments_ "
            print(message + str_kwargs)


    @property
    def consumed_generations(self):
        return self.generation


    @property
    def iteration(self):
        return self.generation


    def __str__(self):
        """Return the algorithm's name."""
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__


    def run(self):
        """Run the algorithm.

        After an initial evaluation of individuals with invalid objective
        values, the :func:`step` function is called in a loop. The algorithm
        stops when a :class:`StopIteration` exception is caught or when the
        stopping criterion evaluates to True.

        """
        # shortcuts
        stopping_criterion = self.stopping_criterion
        step = self.step
        if self.verbosity > 0:
            print(str(self) + " running on problem " + str(self.problem))
        try:
            with self.lock:
                unevaluated = []
                for individual in self.population:
                    if individual.date_of_birth is None:
                        individual.date_of_birth = self.generation
                    individual.date_of_death = None
                    if not individual.objective_values:
                        unevaluated.append(individual)
                self.problem.batch_evaluate(unevaluated)
            while not stopping_criterion():
                step()
        except StopIteration as instance:
            self.last_termination = instance
            if self.verbosity > 0:
                print(instance)
        if self.verbosity > 0:
            print("Algorithm terminated")


    def stopping_criterion(self):
        """Check if optimization should go on.

        The algorithm halts when this method returns True or raises an
        exception.

        Raises
        ------
        ResourcesExhausted 
            When the number of generations reaches the maximum.

        """
        if self.remaining_generations <= 0:
            raise ResourcesExhausted("generations")
        return False


    def step(self):
        """Carry out a single step of optimization."""
        num_offspring = self.num_offspring
        with self.lock:
            # time flies
            for individual in self.population:
                individual.age += 1
            # generate offspring
            offspring = self.reproduction.create(self.population, num_offspring)
        for individual in offspring:
            individual.date_of_birth = self.generation
        self.offspring = offspring
        # individuals are evaluated
        self.problem.batch_evaluate(offspring)
        with self.lock:
            # survivor selection
            selection_result = self.survivor_selection(self.population,
                                                       offspring,
                                                       self.population_size)
            population, rejected, deceased = selection_result
            # store for next generation
            self.population[:] = population
        for individual in deceased:
            individual.date_of_death = self.generation
        # store for potential logging
        self.rejected = rejected
        self.deceased = deceased
        self.notify_observers()
        # increment generation
        self.generation += 1
        self.remaining_generations -= 1


    def survivor_selection(self, parents, offspring, num_survivors):
        """Carry out survivor selection.

        Parents and offspring are treated differently in this method.
        A parent may be removed because it is too old or because it has
        a bad fitness. Offspring individuals can only be removed because
        of bad fitness. The fitness is determined by the EA's selection
        component. (Note that fitness is not necessarily equivalent to an
        individual's objective values.)

        This method guarantees that exactly `num_survivors` individuals
        survive, as long as
        ``len(parents) + len(offspring) >= num_survivors``. To ensure this
        invariant, the best of the too old parents may be retained in the
        population, although their maximum age is technically exceeded.
        If ``len(parents) + len(offspring) < num_survivors``, no one is
        removed.

        Parameters
        ----------
        parents : list of Individual
            Individuals in the parent population.
        offspring : list of Individual
            Individuals in the offspring population.
        num_survivors : int
            The number of surviving individuals.

        Returns
        -------
        population : list
            The survivors of the selection.
        rejected : list
            Individuals removed due to bad fitness.
        deceased : list
            Rejected + individuals who died of old age.

        """
        old_parents = []
        young_parents = []
        for parent in parents:
            if parent.age >= self.max_age:
                old_parents.append(parent)
            else:
                young_parents.append(parent)
        pop_size_diff = len(young_parents) + len(offspring) - num_survivors
        if pop_size_diff > 0:
            population = young_parents + offspring
            random.shuffle(population)
            rejected = self.selection.reduce_to(population,
                                                num_survivors,
                                                already_chosen=self.archive)
            deceased = rejected + old_parents
        elif pop_size_diff < 0:
            population = old_parents[:]
            chosen = young_parents + offspring
            random.shuffle(population)
            rejected = self.selection.reduce_to(population,
                                                abs(pop_size_diff),
                                                already_chosen=self.archive + chosen)
            population += chosen
            deceased = rejected
        else:
            population = young_parents + offspring
            random.shuffle(population)
            rejected = []
            deceased = old_parents
        if len(parents) + len(offspring) >= num_survivors:
            assert len(population) == num_survivors
        else:
            assert len(population) == len(parents) + len(offspring)
        return population, rejected, deceased



class CommaEA(EvolutionaryAlgorithm):
    """An evolutionary algorithm with so-called comma selection.

    In this EA, individuals live at most one generation. This is assumed
    to be somewhat beneficial on multimodal and dynamic problems. Typically,
    this algorithm is used in combination with the self-adaptive step-size
    control provided by
    :class:`ESIndividual<evoalgos.individual.ESIndividual>` or
    :class:`CMSAIndividual<evoalgos.individual.CMSAIndividual>`. For more
    information, see [Beyer2002]_.

    """
    def __init__(self, problem,
                 start_population,
                 population_size,
                 num_offspring,
                 reproduction=None,
                 selection=None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        problem : optproblems.Problem
            An optimization problem.
        start_population : list of Individual
            A list of individuals.
        population_size : int
            The number of individuals that will survive the selection step
            in each generation.
        num_offspring : int
            The number of individuals born in every generation.
        reproduction : Reproduction, optional
            A :class:`Reproduction<evoalgos.reproduction.Reproduction>`
            object selecting the parents for mating and creating the
            offspring. By default,
            :class:`ESReproduction<evoalgos.reproduction.ESReproduction>`
            is chosen.
        selection : Selection, optional
            A :class:`Selection<evoalgos.selection.Selection>` object
            carrying out the survivor selection. By default,
            :class:`TruncationSelection<evoalgos.selection.TruncationSelection>`
            based on
            :class:`LexicographicSorting<evoalgos.sorting.LexicographicSorting>`
            is used.
        kwargs
            Arbitrary keyword arguments, passed to the super class.

        """
        if reproduction is None:
            reproduction = ESReproduction()
        if selection is None:
            selection = TruncationSelection()
        EvolutionaryAlgorithm.__init__(self, problem,
                                       start_population,
                                       population_size,
                                       num_offspring,
                                       1,
                                       reproduction,
                                       selection,
                                       **kwargs)



class PlusEA(EvolutionaryAlgorithm):
    """An evolutionary algorithm with so-called plus selection.

    In this EA, no maximum age is set for individuals. This is especially
    suitable for unimodal problems. Typically, this algorithm is used in
    combination with the self-adaptive step-size control provided by
    :class:`ESIndividual<evoalgos.individual.ESIndividual>`. For more
    information, see [Beyer2002]_.

    """
    def __init__(self, problem,
                 start_population,
                 population_size,
                 num_offspring,
                 reproduction=None,
                 selection=None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        problem : optproblems.Problem
            An optimization problem.
        start_population : list of Individual
            A list of individuals.
        population_size : int
            The number of individuals that will survive the selection step
            in each generation.
        num_offspring : int
            The number of individuals born in every generation.
        reproduction : Reproduction, optional
            A :class:`Reproduction<evoalgos.reproduction.Reproduction>`
            object selecting the parents for mating and creating the
            offspring. By default,
            :class:`ESReproduction<evoalgos.reproduction.ESReproduction>`
            is chosen.
        selection : Selection, optional
            A :class:`Selection<evoalgos.selection.Selection>` object
            carrying out the survivor selection. By default,
            :class:`TruncationSelection<evoalgos.selection.TruncationSelection>`
            based on
            :class:`LexicographicSorting<evoalgos.sorting.LexicographicSorting>`
            is used.
        kwargs
            Arbitrary keyword arguments, passed to the super class.

        """
        if reproduction is None:
            reproduction = ESReproduction()
        if selection is None:
            selection = TruncationSelection()
        EvolutionaryAlgorithm.__init__(self, problem,
                                       start_population,
                                       population_size,
                                       num_offspring,
                                       None,
                                       reproduction,
                                       selection,
                                       **kwargs)



class CMSAES(EvolutionaryAlgorithm):
    """Covariance matrix self-adaptation ES.

    This is a convenience class for easy setup of this algorithm. For more
    information on the working principle of the algorithm, see [Beyer2008]_.

    """

    def __init__(self, problem,
                 start_population,
                 population_size=None,
                 num_offspring=None,
                 max_age=1,
                 reproduction=None,
                 selection=None,
                 xtol=None,
                 steptol=None,
                 **kwargs):
        r"""Constructor.

        Parameters
        ----------
        problem : optproblems.Problem
            An optimization problem.
        start_population : list of Individual
            A list of individuals.
        population_size : int, optional
            The number of individuals that will survive the selection step
            in each generation. The default is 
            :math:`\lfloor(4 + 3 \log(n))/2\rfloor`.
        num_offspring : int, optional
            The number of individuals born in every generation. Default is
            four times the population size.
        max_age : int, optional
            A maximum number of generations an individual can live. This
            number may be exceeded if not enough offspring is generated to
            reach the population_size.
        reproduction : Reproduction, optional
            A :class:`Reproduction<evoalgos.reproduction.Reproduction>`
            object selecting the parents for mating and creating the
            offspring. By default,
            :class:`ESReproduction<evoalgos.reproduction.ESReproduction>`
            is chosen.
        selection : Selection, optional
            A :class:`Selection<evoalgos.selection.Selection>` object
            carrying out the survivor selection. By default,
            :class:`TruncationSelection<evoalgos.selection.TruncationSelection>`
            based on
            :class:`LexicographicSorting<evoalgos.sorting.LexicographicSorting>`
            is used.
        xtol : float, optional
            The algorithm stops when the maximal absolute deviation in all
            dimensions of the search space between the best and all other
            solutions in the population drops below this value. By default,
            this criterion is off.
        steptol : float, optional
            The algorithm stops when the mutation strength parameters of all
            the individuals in the population drop below this value.
        kwargs
            Arbitrary keyword arguments, passed to the super class.

        """
        for individual in start_population:
            assert isinstance(individual, CMSAIndividual)
        if population_size is None:
            dim = len(start_population[0].phenome)
            population_size = int((4 + math.floor(3 * math.log(dim))) * 0.5)
        if num_offspring is None:
            num_offspring = 4 * population_size
        if reproduction is None:
            reproduction = ESReproduction()
        if selection is None:
            selection = TruncationSelection()
        self.xtol = xtol
        self.steptol = steptol
        EvolutionaryAlgorithm.__init__(self, problem,
                                       start_population,
                                       population_size,
                                       num_offspring,
                                       max_age,
                                       reproduction,
                                       selection,
                                       **kwargs)


    def stopping_criterion(self):
        """Check if optimization should go on.

        The algorithm halts when this method returns True or raises an
        exception.

        Raises
        ------
        ResourcesExhausted 
            When the number of generations reaches the maximum.
        Stalled
            When a dispersion-related key figure of the population drops
            below a threshold.

        """
        steptol = self.steptol
        if self.xtol is not None and len(self.population) == self.population_size:
            # assume population is sorted (may be wrong)
            best_phenome = self.population[0].phenome
            max_diff = 0.0
            for i, individual in enumerate(self.population, 1):
                for phene, best_phene in zip(individual.phenome, best_phenome):
                    diff = abs(phene - best_phene)
                    if diff > max_diff:
                        max_diff = diff
            if max_diff <= self.xtol:
                raise Stalled("xtol")
        if steptol is not None:
            is_one_larger = False
            for individual in self.population:
                if individual.mutation_strength > steptol:
                    is_one_larger = True
                    break
            if not is_one_larger:
                raise Stalled("steptol")
        return EvolutionaryAlgorithm.stopping_criterion(self)


    @classmethod
    def minimize(cls, fun,
                 x0,
                 args=(),
                 bounds=None,
                 callback=None,
                 options=None,
                 **kwargs):
        """Minimization of scalar function of one or more variables.

        This function mimics the SciPy interface for optimizers.

        Parameters
        ----------
        fun : callable
            Objective function.
        x0 : sequence
            Initial guess.
        args : tuple, optional
            Extra arguments passed to the objective function.
        bounds : sequence, optional
            Bounds for variables. ``(min, max)`` pairs for each element in
            ``x``, defining the bounds on that parameter. Use None for one of
            ``min`` or ``max`` when there is no bound in that direction.
        callback : callable, optional
            Called after each iteration, as ``callback(xk)``, where ``xk`` is
            the current parameter vector.
        options : dict, optional
            A dictionary of solver options. The following generic options are
            accepted:

                maxiter : int
                    Maximum number of iterations to perform.
                disp : bool
                    Set to True to print convergence messages.

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination.

        """
        def wrapper_function(phenome):
            return fun(phenome, *args)

        class Observer:

            def __init__(self):
                self.best_individual = None

            def __call__(self, algo):
                select = algo.selection.select
                best_ind = select(algo.population, 1)[0]
                if self.best_individual is None:
                    self.best_individual = best_ind
                else:
                    self.best_individual = select([best_ind, self.best_individual], 1)[0]
                try:
                    phenome = repair(best_ind.phenome)
                except:
                    phenome = best_ind.phenome
                try:
                    callback(phenome)
                except:
                    pass

        dim = len(x0)
        popsize = int((4 + math.floor(3 * math.log(dim))) * 0.5)
        num_offspring = 4 * popsize
        try:
            obj_function_name = fun.__name__
        except AttributeError:
            obj_function_name = str(fun)
        problem_name = "Problem(" + obj_function_name + ")"
        if bounds is not None:
            min_bounds = [min_bound for min_bound, _ in bounds]
            max_bounds = [max_bound for _, max_bound in bounds]
            unit_cube = ([0.0] * dim, [1.0] * dim)
            scale_to_orig = ScalingPreprocessor(from_cuboid=unit_cube,
                                                to_cuboid=(min_bounds, max_bounds))
            scale_to_unit = ScalingPreprocessor(from_cuboid=(min_bounds, max_bounds),
                                                to_cuboid=unit_cube)
            repair = BoundConstraintsRepair((min_bounds, max_bounds),
                                            ["reflection"] * dim,
                                            previous_preprocessor=scale_to_orig)
            problem = Problem(wrapper_function,
                              phenome_preprocessor=repair,
                              name=problem_name)
            x0 = scale_to_unit(x0)
        else:
            problem = Problem(wrapper_function, name=problem_name)
        start_individual = CMSAIndividual(num_parents=popsize,
                                          num_offspring=num_offspring,
                                          genome=array.array("d", x0))
        if options is not None and "maxiter" in options:
            kwargs["max_generations"] = options["maxiter"]
        try:
            kwargs["max_generations"] = kwargs["maxiter"]
        except KeyError:
            pass
        if options is not None and "disp" in options:
            kwargs["verbosity"] = int(options["disp"])
        try:
            kwargs["verbosity"] = int(kwargs["disp"])
        except KeyError:
            pass
        algo = cls(problem, [start_individual], popsize, num_offspring, **kwargs)
        observer = Observer()
        algo.attach(observer)
        algo.run()
        result = OptimizeResult()
        best_individual = observer.best_individual
        if bounds is None:
            result.x = best_individual.phenome
        else:
            result.x = repair(best_individual.phenome)
        result.message = str(algo.last_termination)
        result.fun = best_individual.objective_values
        result.nfev = problem.consumed_evaluations
        result.nit = algo.generation
        return result



class NSGA2b(EvolutionaryAlgorithm):
    """An enhanced non-dominated sorting genetic algorithm 2.

    The algorithm was originally devised by [Deb2000]_. In this
    implementation, the improved selection proposed by [Kukkonen2006]_ is
    used by default (although not with any special data structures as in
    the paper). Also the number of offspring can be chosen freely in
    contrast to the original definition, so that also a (mu + 1)-approach
    as in [Durillo2009]_ is possible.

    .. warning:: This algorithm should only be used for two objectives, as
        the selection criterion is not suited for higher dimensions.

    References
    ----------
    .. [Deb2000] Kalyanmoy Deb, Samir Agrawal, Amrit Pratap, and T Meyarivan
        (2000). A Fast Elitist Non-Dominated Sorting Genetic Algorithm for
        Multi-Objective Optimization: NSGA-II. In: Parallel Problem Solving
        from Nature, PPSN VI, Volume 1917 of Lecture Notes in Computer
        Science, pp 849-858, Springer.
        https://dx.doi.org/10.1007/3-540-45356-3_83

    .. [Kukkonen2006] Kukkonen, Saku; Deb, Kalyanmoy (2006).Improved Pruning
        of Non-Dominated Solutions Based on Crowding Distance for
        Bi-Objective Optimization Problems. In: IEEE Congress on
        Evolutionary Computation, pp. 1179-1186.
        https://dx.doi.org/10.1109/CEC.2006.1688443

    .. [Durillo2009] Juan J. Durillo, Antonio J. Nebro, Francisco Luna,
        Enrique Alba (2009). On the Effect of the Steady-State Selection
        Scheme in Multi-Objective Genetic Algorithms. In: Evolutionary
        Multi-Criterion Optimization, Volume 5467 of Lecture Notes in
        Computer Science, pp 183-197, Springer.
        https://dx.doi.org/10.1007/978-3-642-01020-0_18

    """
    def __init__(self, problem,
                 start_population,
                 population_size,
                 num_offspring=None,
                 reproduction=None,
                 do_backward_elimination=True,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        problem : optproblems.Problem
            A multiobjective optimization problem.
        start_population : list of Individual
            The initial population of individuals. The size of this list
            does not have to be the same as `population_size`, but will be
            adjusted subsequently.
        population_size : int
            The number of individuals that will survive the selection step
            in each generation.
        num_offspring : int, optional
            The number of individuals born in every generation. By default,
            this value is set equal to the population size.
        reproduction : Reproduction, optional
            A :class:`Reproduction<evoalgos.reproduction.Reproduction>`
            object selecting the parents for mating and creating the
            offspring. If no object is provided, a default variant is
            generated, which selects parents uniformly random.
        do_backward_elimination : bool, optional
            This argument only has influence if ``num_offspring > 1``.
            Backward elimination means that in a greedy fashion, the worst
            individuals are removed one by one. The alternative is the
            original 'super-greedy' approach, which removes the necessary
            number of individuals without recalculating the fitness of the
            other ones in between. Default is True (the former approach),
            which is also recommended, because it is more accurate.
        kwargs
            Further keyword arguments passed to the constructor of the
            super class.

        """
        selection = TruncationSelection(CrowdingDistanceSorting())
        if do_backward_elimination:
            selection = BackwardElimination(selection)
        if reproduction is None:
            reproduction = ESReproduction()
        if num_offspring is None:
            num_offspring = population_size
        EvolutionaryAlgorithm.__init__(self, problem,
                                       start_population,
                                       population_size,
                                       num_offspring,
                                       None,
                                       reproduction,
                                       selection,
                                       **kwargs)



class SMSEMOA(EvolutionaryAlgorithm):
    """The S-metric Selection EMOA.

    This multiobjective optimization algorithm uses a solution's exclusive
    contribution to the hypervolume of the worst non-dominated front of a
    population as a selection criterion. (Hypervolume was formerly known as
    S-metric.) The algorithm was proposed in [Emmerich2005]_ and
    [Naujoks2005]_.

    .. warning:: The time for calculating the hypervolume is exponential in
        the number of objectives.

    References
    ----------
    .. [Emmerich2005] Michael Emmerich, Nicola Beume, Boris Naujoks (2005).
        An EMO Algorithm Using the Hypervolume Measure as Selection
        Criterion. In: Evolutionary Multi-Criterion Optimization, Volume
        3410 of Lecture Notes in Computer Science, pp 62-76, Springer.
        https://dx.doi.org/10.1007/978-3-540-31880-4_5
    .. [Naujoks2005] Boris Naujoks, Nicola Beume, Michael Emmerich (2005).
        Multi-objective optimisation using S-metric selection: application
        to three-dimensional solution spaces. In: The 2005 IEEE Congress
        on Evolutionary Computation, vol.2, pp.1282-1289, IEEE Press.
        https://dx.doi.org/10.1109/CEC.2005.1554838

    """
    def __init__(self, problem,
                 start_population,
                 population_size,
                 num_offspring=1,
                 reproduction=None,
                 prefer_boundary_points=True,
                 selection_variant="auto",
                 offsets=None,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        problem : optproblems.Problem
            A multiobjective optimization problem.
        start_population : list of Individual
            The initial population of individuals. The size of this list
            does not have to be the same as `population_size`, but will be
            adjusted subsequently.
        population_size : int
            The number of individuals that will survive the selection step
            in each generation.
        num_offspring : int, optional
            The number of individuals born in every generation. This value
            is typically 1, but the implementation also admits larger
            values.
        reproduction : Reproduction, optional
            A :class:`Reproduction<evoalgos.reproduction.Reproduction>`
            object selecting the parents for mating and creating the
            offspring. If no object is provided, a default variant is
            generated, which selects parents uniformly random.
        prefer_boundary_points : bool, optional
            This flag only pertains to the two-objective case. If it is
            set to True, the two boundary points (but not their potentially
            existing duplicates) of a front are guaranteed to be retained.
        selection_variant : string, optional
            This argument has the possible values 'forward-greedy',
            'backward-greedy', 'super-greedy', and 'auto'. 'backward-greedy'
            means that the worst individuals are iteratively removed one by
            one (also known as backward elimination). 'forward-greedy' means
            that also iteratively, the individual maximizing the hypervolume
            together with the already selected ones is chosen. 'auto'
            selects forward selection if :math:`\lambda > \mu`, and backward
            elimination otherwise. This rule was devised based on results in
            [Schendekehl2017]_. Finally, 'super-greedy' removes the
            necessary number of individuals without recalculating the
            fitness of the other ones in between (this is faster, but not
            recommended). We have to know the variant internally, because
            the results of the wrapper approach with
            :class:`BackwardElimination <evoalgos.selection.BackwardElimination>`
            or
            :class:`ForwardSelection <evoalgos.selection.ForwardSelection>`
            are not 100% correct (due to reference point construction being
            triggered in every iteration) and because some time can be
            saved.
        offsets : list, optional
            For calculating the hypervolume, a reference point is required.
            The reference point is typically calculated as the worst
            objective values in the last front plus an offset vector, which
            can be specified here. Default offset is [1.0, ..., 1.0].
        kwargs
            Further keyword arguments passed to the constructor of the
            super class.

        """
        selection = HyperVolumeContributionSelection(offsets,
                                                     prefer_boundary_points,
                                                     selection_variant)
        if reproduction is None:
            reproduction = ESReproduction()
        EvolutionaryAlgorithm.__init__(self, problem,
                                       start_population,
                                       population_size,
                                       num_offspring,
                                       None,
                                       reproduction,
                                       selection,
                                       **kwargs)
