import copy
import itertools
import random


class Individual(object):
    """Base class for individuals.

    This class does not make any assumptions about the individual's
    genome. Implementing the genome and an appropriate
    mutation and recombination is completely in the user's responsibility.

    Apart from the arguments provided in the constructor, this class
    possesses the member attributes `age`, `date_of_birth`, and
    `date_of_death`, which can be manipulated by evolutionary algorithms.

    """
    id_generator = itertools.count(1)

    def __init__(self, genome=None,
                 objective_values=None,
                 constraint_violation=None,
                 repair_component=None,
                 num_parents=2,
                 id_number=None):
        """Constructor.

        Parameters
        ----------
        genome : object, optional
            An arbitrary object containing the genome of the individual.
        objective_values : iterable, optional
            The objective values are usually obtained by evaluating the
            individual's phenome.
        constraint_violation: iterable, optional
            The constraint violation values are usually obtained by
            evaluating the individual phenome, against the problem
            constraints.
        repair_component : callable, optional
            A function or callable object that will get the phenome as
            input and shall return a repaired copy of the phenome.
        num_parents : int, optional
            How many individuals are involved in a recombination
            procedure. Default is 2.
        id_number : int, optional
            A currently unused identification number. If no ID is provided,
            one is automatically generated with :func:`itertools.count`.

        """
        self.genome = genome
        self.objective_values = objective_values
        self.constraint_violation = constraint_violation
        self.repair_component = repair_component
        self.num_parents = num_parents
        if id_number is None:
            self.id_number = next(Individual.id_generator)
        else:
            self.id_number = id_number
        self.age = 0
        self.date_of_birth = None
        self.date_of_death = None

    @property
    def phenome(self):
        """Accessor to obtain the phenome from the genome.

        This mapping from genome to phenome exists to provide the
        possibility of using a search space that is different to the
        pre-image of the objective function. Only read-access is provided
        for this attribute.

        """
        return self.genome

    def invalidate_objective_values(self):
        """Set objective values to None."""
        self.objective_values = None

    def invalidate_constraint_violation(self):
        """Set objective values to None."""
        self.constraint_violation = None

    def mutate(self):
        """Mutate this individual.

        This is a template method that calls three other methods in
        turn. First, the individual is mutated with :func:`_mutate`,
        then :func:`invalidate_objective_values` is called, and finally
        :func:`repair` is carried out.

        """
        self._mutate()
        self.invalidate_objective_values()
        self.invalidate_constraint_violation()
        self.repair()

    def _mutate(self):
        """Does the real work for mutation.

        This is an abstract method. Override in your own individual.
        The individual must be changed in-place and not returned.

        """
        raise NotImplementedError("Mutation not implemented.")

    def repair(self):
        """Repair this individual.

        If existing, the repair component is applied to the genome to
        do the work.

        """
        if self.repair_component is not None:
            self.genome = self.repair_component(self.genome)

    def recombine(self, others):
        """Generate offspring from several individuals.

        This is a template method. First, the parents are recombined with
        :func:`_recombine`, then :func:`invalidate_objective_values` is
        called.

        Returns
        -------
        children : list
            Newly generated offspring.

        """
        children = self._recombine(others)
        for child in children:
            child.invalidate_objective_values()
            child.invalidate_constraint_violation()
        return children

    def _recombine(self, others):
        """Does the real work for recombination.

        This is an abstract method. Override in your own individual if you
        want recombination. This method returns new individuals, so this
        individual should not be modified.

        Returns
        -------
        children : list
            Newly generated offspring.

        """
        raise NotImplementedError("Recombination not implemented.")

    def clone(self):
        """Clone the individual.

        Makes a flat copy of this individual with some exceptions. The
        clone's age is set back to 0, a new ID is assigned to the clone,
        and for objective values and the genome a deep copy is made.

        """
        child = copy.copy(self)
        child.genome = copy.deepcopy(self.genome)
        child.objective_values = copy.deepcopy(self.objective_values)
        child.constraint_violation = copy.deepcopy(self.constraint_violation)
        child.age = 0
        child.id_number = next(Individual.id_generator)
        return child

    def __str__(self):
        """Return string representation of the individual's objective values."""
        return str(self.objective_values)


class SBXIndividual(Individual):
    """An individual imitating binary variation on a real-valued genome.

    Recombination is implemented as simulated binary crossover (SBX).
    Mutation is polynomial mutation. This kind of individual is often
    used in genetic algorithms.

    """
    min_bounds = None
    max_bounds = None

    def __init__(self, min_bounds, max_bounds, crossover_dist_index=10,
                 mutation_dist_index=10,
                 crossover_prob=0.9,
                 mutation_prob=0.1,
                 symmetric_variation_prob=0.0,
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        crossover_dist_index : float, optional
            Controls the variance of the distribution used for
            recombination. The higher this value, the lower the variance.
        mutation_dist_index : float, optional
            Controls the variance of the distribution used for mutation.
            The higher this value, the lower the variance.
        crossover_prob : float, optional
            The probability to recombine a single gene.
        mutation_prob : float, optional
            The probability to mutate a single gene.
        symmetric_variation_prob : float, optional
            The probability for enforcing symmetric mutation and
            recombination distributions. If symmetry is not enforced and
            bound-constraints exist, the distributions have the whole search
            space as domain. See [Wessing2009]_ for further information on
            this parameter.
        kwargs :
            Arbitrary keyword arguments, passed to the super class.

        References
        ----------
        .. [Wessing2009] Simon Wessing. Towards Optimal Parameterizations of
            the S-Metric Selection Evolutionary Multi-Objective Algorithm.
            Diploma thesis, Algorithm Engineering Report TR09-2-006,
            Technische Universität Dortmund, 2009.
            https://ls11-www.cs.uni-dortmund.de/_media/techreports/tr09-06.pdf

        """
        Individual.__init__(self, **kwargs)
        if crossover_prob > 0:
            assert self.num_parents == 2
        self.crossover_dist_index = crossover_dist_index
        self.mutation_dist_index = mutation_dist_index
        # only used if not both limits specified, and only in mutation
        self.max_perturbation = 1.0
        # probability for mutation of one position in the genome
        self.mutation_prob = mutation_prob
        # probability for recombination of one position in the genome
        self.crossover_prob = crossover_prob
        self.symmetric_variation_prob = symmetric_variation_prob
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def _mutate(self):
        """Mutate this individual with polynomial mutation.

        This mutation follows the same concept as SBX.

        """
        is_symmetric = (random.random() < self.symmetric_variation_prob)
        min_bounds = self.min_bounds
        max_bounds = self.max_bounds
        genome = self.genome
        mutation_prob = self.mutation_prob
        max_perturbation = self.max_perturbation
        calc_delta_q = self.calc_delta_q
        for i in range(len(genome)):
            if random.random() < mutation_prob:
                y = genome[i]
                # min_y and max_y are lower and upper limits for this variable
                min_y = None
                max_y = None
                if min_bounds is not None:
                    min_y = min_bounds[i]
                if max_bounds is not None:
                    max_y = max_bounds[i]
                # keep this variable fix if upper and lower limit are identical
                if min_y == max_y and min_y is not None:
                    if y != min_y:
                        message = "x_" + str(i) + " (=" + str(y)
                        message += ") should be " + str(min_y)
                        raise ValueError(message)
                else:
                    delta_lower = 1.0
                    delta_upper = 1.0
                    if min_y is not None and max_y is not None:
                        delta_lower = (y - min_y) / (max_y - min_y)
                        delta_upper = (max_y - y) / (max_y - min_y)
                    if is_symmetric:
                        delta_lower = min(delta_upper, delta_lower)
                        delta_upper = min(delta_upper, delta_lower)
                    # random real number from [0, 1[
                    u = random.random()
                    delta_q = calc_delta_q(u, delta_lower, delta_upper)
                    dist = max_perturbation
                    if min_y is not None and max_y is not None:
                        dist = max_y - min_y
                    # mutate
                    y += delta_q * dist
                    # maybe enforce limits
                    if max_y is not None:
                        y = min(y, max_y)
                    if min_y is not None:
                        y = max(y, min_y)
                # set variable in genome
                genome[i] = y

    def calc_delta_q(self, rand, delta_lower, delta_upper):
        """Helper function for mutation."""
        mutation_dist_index = self.mutation_dist_index
        exponent = 1.0 / (mutation_dist_index + 1.0)
        if rand <= 0.5:
            factor = pow(1.0 - delta_lower, mutation_dist_index + 1.0)
            val = 2.0 * rand + (1.0 - 2.0 * rand) * factor
            delta_q = pow(val, exponent) - 1.0
        else:
            factor = pow(1.0 - delta_upper, mutation_dist_index + 1.0)
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * factor
            delta_q = 1.0 - pow(val, exponent)
        return delta_q

    def _recombine(self, others):
        """Produce two children from two parents.

        This kind of crossover is claimed to have self-adaptive features
        because the distance of the children to their parents is influenced
        by the parents' distance [Deb2001]_.

        Parameters
        ----------
        others : iterable of Individual
            Other parents to recombine this individual with.

        Returns
        -------
        children : list of Individual
            A list containing two children.

        References
        ----------
        .. [Deb2001] Kalyanmoy Deb and Hans-Georg Beyer (2001).
            Self-Adaptive Genetic Algorithms with Simulated Binary
            Crossover. Evolutionary Computation, 9:2, pp. 197-221.

        """
        assert len(others) == 1
        is_symmetric = random.random() < self.symmetric_variation_prob
        parent1 = self
        parent2 = others[0]
        child1 = parent1.clone()
        child2 = parent2.clone()

        min_bounds = self.min_bounds
        max_bounds = self.max_bounds
        genome_length = len(self.genome)
        crossover_dist_index = self.crossover_dist_index
        calc_beta_q = self.calc_beta_q
        for i in range(genome_length):
            # y1 and y2 are the two parent values
            y1 = parent1.genome[i]
            y2 = parent2.genome[i]
            # c1 and c2 become the two children's values
            c1 = parent1.genome[i]
            c2 = parent2.genome[i]
            # min_y and max_y are the lower and upper limit for this variable
            min_y = None
            max_y = None
            if min_bounds is not None:
                min_y = min_bounds[i]
            if max_bounds is not None:
                max_y = max_bounds[i]
            y_dist = abs(y1 - y2)
            is_dist_too_small = y_dist < 1.0e-14
            # keep this variable fix if the upper and lower limit are identical
            if min_y == max_y and min_y is not None:
                if y1 != min_y:
                    message = "x_" + str(i) + " (=" + str(y1)
                    message += ") should be " + str(min_y)
                    raise ValueError(message)
                if y2 != min_y:
                    message = "x_" + str(i) + " (=" + str(y2)
                    message += ") should be " + str(min_y)
                    raise ValueError(message)
            # else do simulated binary crossover
            elif not is_dist_too_small and (random.random() < self.crossover_prob or genome_length == 1):
                # ensure that y1 is smaller than or equal to y2
                if y1 > y2:
                    y1, y2 = y2, y1
                # random real number from [0, 1[
                u = random.random()
                # spread factors
                beta_lower = None
                beta_upper = None
                if min_y is not None and max_y is not None:
                    beta_lower = 1.0 + (2.0 * (y1 - min_y) / y_dist)
                    beta_upper = 1.0 + (2.0 * (max_y - y2) / y_dist)
                    if is_symmetric:
                        beta_lower = min(beta_lower, beta_upper)
                        beta_upper = min(beta_lower, beta_upper)
                elif min_y is None and max_y is not None:
                    beta_upper = 1.0 + (2.0 * (max_y - y2) / y_dist)
                    if is_symmetric:
                        beta_lower = beta_upper
                elif min_y is not None and max_y is None:
                    beta_lower = 1.0 + (2.0 * (y1 - min_y) / y_dist)
                    if is_symmetric:
                        beta_upper = beta_lower
                # compute the children's values
                if beta_lower is None:
                    # no lower limit and not symmetric variation
                    beta_q = calc_beta_q(u)
                else:
                    alpha = 2.0 - pow(beta_lower, -(crossover_dist_index + 1.0))
                    beta_q = calc_beta_q(u, alpha)
                # set value for first child
                c1 = 0.5 * ((y1 + y2) - beta_q * y_dist)
                if beta_upper is None:
                    # no upper limit and not symmetric variation
                    beta_q = self.calc_beta_q(u)
                else:
                    alpha = 2.0 - pow(beta_upper, -(crossover_dist_index + 1.0))
                    beta_q = calc_beta_q(u, alpha)
                # set value for second child
                c2 = 0.5 * ((y1 + y2) + beta_q * y_dist)
                # maybe enforce limits
                if min_y is not None:
                    c2 = max(c2, min_y)
                    c1 = max(c1, min_y)
                if max_y is not None:
                    c1 = min(c1, max_y)
                    c2 = min(c2, max_y)
            # set variables in genome
            child1.genome[i] = c1
            child2.genome[i] = c2
        return [child1, child2]

    def calc_beta_q(self, rand, alpha=2.0):
        """Helper function for crossover."""
        if rand <= 1.0 / alpha:
            ret = alpha * rand
        else:
            ret = 1.0 / (2.0 - alpha * rand)
        ret **= 1.0 / (self.crossover_dist_index + 1.0)
        return ret
