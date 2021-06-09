import random
import threading
import globals

from evoalgos.algo import EvolutionaryAlgorithm, Observable
from evoalgos.reproduction import ESReproduction
from Selection import BackwardElimination
from Selection import CurveFamSelection
from Sort import CurveDistanceSorting


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
                 max_generations = float("inf"),
                 archive = None,
                 verbosity = 1,
                 lock = None,
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
        f = open(f'All_Populations_{globals.num}', "w+")
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
                for individual in self.population:
                    print(individual.phenome, individual.objective_values, individual.constraint_violation,
                          file=f)
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
            print('selection: ', selection_result)
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
            print('rejected: ', rejected)
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


class coarseEMOA(EvolutionaryAlgorithm):
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
        selection = CurveFamSelection(CurveDistanceSorting(num_objectives=problem.num_objectives))
        if do_backward_elimination:
            selection = BackwardElimination(selection)
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
