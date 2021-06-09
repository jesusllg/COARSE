# This Python file uses the following encoding: utf-8
"""Data structures to store objective values together with the solution.

The classes in this module have attributes for storing the genome (the
solution in the search space) and corresponding objective values. They also
provide methods for generating new individuals (offspring) by mutation and
recombination. :class:`Individual` is the base class for all other individuals.

"""
import itertools
import random
import array
import copy
import math
import sys

import numpy as np

from optproblems import Aborted


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


    def mutate(self):
        """Mutate this individual.

        This is a template method that calls three other methods in
        turn. First, the individual is mutated with :func:`_mutate`,
        then :func:`invalidate_objective_values` is called, and finally
        :func:`repair` is carried out.

        """
        self._mutate()
        self.invalidate_objective_values()
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
        child.age = 0
        child.id_number = next(Individual.id_generator)
        return child


    def __str__(self):
        """Return string representation of the individual's objective values."""
        return str(self.objective_values)



class BinaryIndividual(Individual):
    """Individual with a binary genome."""

    def __init__(self, bit_flip_prob=None, **kwargs):
        """Constructor.

        Parameters
        ----------
        bit_flip_prob : float, optional
            The probability to flip a bit. Default is ``1.0 / len(genome)``.
        kwargs
            Arbitrary keyword arguments, passed to the super class.

        """
        Individual.__init__(self, **kwargs)
        self.bit_flip_prob = bit_flip_prob


    def _mutate(self):
        """Carry out standard bit mutation."""
        genome = self.genome
        if self.bit_flip_prob is None:
            self.bit_flip_prob = 1.0 / len(genome)
        bit_flip_prob = self.bit_flip_prob
        for i in range(len(genome)):
            if random.random() < bit_flip_prob:
                genome[i] = 1 - genome[i]


    def _recombine(self, others):
        """Uniform recombination.

        Parameters
        ----------
        others : iterable of Individual
            Other parents to recombine this individual with.

        Returns
        -------
        children : list of Individual
            A list containing a single child.

        """
        parents = [self] + others
        parent_genomes = [parent.genome for parent in parents]
        child = self.clone()
        genome = child.genome
        for i in range(len(genome)):
            genome[i] = random.choice([pg[i] for pg in parent_genomes])
        return [child]



class FixedSizeSetIndividual(Individual):
    """Individual with a set as genome.

    Despite the name of this class, the genome must be a sequence, due to
    implementation issues (:func:`random.choice` is not supported for
    sets). However, the set property is enforced for offspring created by
    recombination.

    """

    def __init__(self, avail_elements, swap_prob=None, binary=False, **kwargs):
        """Constructor.

        Parameters
        ----------
        avail_elements : sequence
            The possible elements of the set.
        swap_prob : float, optional
            The probability to swap a set element. Default is
            ``1.0 / len(genome)``.
        binary : bool, optional
            Indicates if the phenome should be presented as bitstring
            to the outside world. This is useful for compatibility with
            binary optimization problems.
        kwargs
            Arbitrary keyword arguments, passed to the super class.

        """
        Individual.__init__(self, **kwargs)
        self.avail_elements = avail_elements
        self.swap_prob = swap_prob
        self.binary = binary


    @property
    def phenome(self):
        """Either the actual genome or a bitstring representation."""
        if self.binary:
            phenome_array = array.array("B", [0] * len(self.avail_elements))
            mapping = {el: i for i, el in enumerate(self.avail_elements)}
            for element in self.genome:
                phenome_array[mapping[element]] = 1
            return phenome_array
        else:
            return self.genome


    def _mutate(self):
        """A set mutation that retains the cardinality."""
        avail_elements = self.avail_elements
        genome = self.genome
        set_size = len(genome)
        if self.swap_prob is None:
            self.swap_prob = 1.0 / set_size
        swap_prob = self.swap_prob
        non_members = set(avail_elements)
        non_members.difference_update(genome)
        non_members = sorted(non_members)
        for i in range(set_size):
            if random.random() < swap_prob:
                swap_index = random.randrange(len(non_members))
                temp = genome[i]
                # swap one element
                genome[i] = non_members[swap_index]
                non_members[swap_index] = temp


    def _recombine(self, others):
        """Produce one child with a set the same size as this parent.

        This recombination guarantees that the intersection of the parents'
        sets is present in the child. For the remaining elements, the
        probability to be chosen is proportional to the frequency in the
        parents.

        Parameters
        ----------
        others : iterable of Individual
            Other parents to recombine this individual with.

        Returns
        -------
        children : list of Individual
            A list containing a single child.

        """
        self_genome = self.genome
        other_genomes = [other.genome for other in others]
        genomes_concat = self_genome[:]
        for other_genome in other_genomes:
            genomes_concat.extend(other_genome)
        genomes_concat.sort()
        child_genome_set = set(self_genome).intersection(*other_genomes)
        while len(child_genome_set) < len(self_genome):
            child_genome_set.add(random.choice(genomes_concat))
        child = self.clone()
        child.genome[:] = sorted(child_genome_set)
        return [child]



class SBXIndividual(Individual):
    """An individual imitating binary variation on a real-valued genome.

    Recombination is implemented as simulated binary crossover (SBX).
    Mutation is polynomial mutation. This kind of individual is often
    used in genetic algorithms.

    """
    min_bounds = None
    max_bounds = None

    def __init__(self, crossover_dist_index=10,
                 mutation_dist_index=10,
                 crossover_prob=0.7,
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



class ESIndividual(Individual):
    """An individual using Gaussian mutation.

    This variation was developed in the context of Evolution Strategies (ES)
    [Beyer2002]_. One can choose between discrete, intermediate, and
    arithmetic recombination. Self-adaptation is used to adjust strategy
    parameters.

    References
    ----------
    .. [Beyer2002] Hans-Georg Beyer, Hans-Paul Schwefel (2002). Evolution
        strategies - A comprehensive introduction. Natural Computing,
        Volume 1, Issue 1, pp. 3-52.
        https://dx.doi.org/10.1023/A:1015059928466

    """
    def __init__(self, learning_param1=None,
                 learning_param2=None,
                 strategy_params=None,
                 recombination_type="arithmetic",
                 **kwargs):
        """Constructor.

        If both learning parameters are set to zero, adaptation of strategy
        parameters is disabled. See [Beyer2002]_ for more information.

        Parameters
        ----------
        learning_param1 : float, optional
            The first learning parameter for self-adaptation. Default value
            is chosen according to [Beyer2002]_.
        learning_param2 : float, optional
            The second learning parameter for self-adaptation. Default value
            is chosen according to [Beyer2002]_.
        strategy_params : list of float, optional
            The strategy parameters (mutation strengths) to be adapted. This
            list must have either length one or the same length as the
            genome. As a default, [0.25, ..., 0.25] is chosen.
        recombination_type : str, optional
            Must be 'arithmetic', 'intermediate', 'discrete', or 'none'.
        kwargs :
            Arbitrary keyword arguments, passed to the super class.

        """
        Individual.__init__(self, **kwargs)
        self.learning_param1 = learning_param1
        self.learning_param2 = learning_param2
        self.strategy_params = strategy_params
        self.mutation_strength_bounds = [sys.float_info.min, sys.float_info.max]
        self.are_strategy_params_mutated = False
        self.mutation_vector = None
        self.recombination_type = recombination_type
        self.weights = None


    def clone(self):
        """Clone this individual.

        First, the clone method of the super class is called. Then, a deep
        copy of attribute `strategy_params` is additionally made and
        :attr:`are_strategy_params_mutated` is set to False.

        Returns
        -------
        child : ESIndividual
            A clone of this individual.

        """
        child = Individual.clone(self)
        child.strategy_params = copy.deepcopy(self.strategy_params)
        child.are_strategy_params_mutated = False
        return child


    def mutate_strategy_params(self):
        """Mutate the strategy parameters.

        By default, the mutation strengths are bounded to the smallest and
        largest representable positive values of the system,
        :attr:`sys.float_info.min` and :attr:`sys.float_info.max`. You can
        change these bounds by setting
        ``mutation_strength_bounds = [your_min, your_max]``.

        """
        gauss = random.gauss
        sqrt = math.sqrt
        exp = math.exp
        num_variables = len(self.genome)
        if self.strategy_params is None:
            self.strategy_params = array.array("d", [0.25] * num_variables)
        if self.learning_param1 is None:
            self.learning_param1 = 1.0 / sqrt(2.0 * num_variables)
        if self.learning_param2 is None:
            self.learning_param2 = 1.0 / sqrt(2.0 * sqrt(num_variables))
        strategy_params = self.strategy_params
        learning_param2 = self.learning_param2
        general_term = self.learning_param1 * gauss(0.0, 1.0)
        # strategy parameter adaptation
        assert None not in self.mutation_strength_bounds
        min_strength, max_strength = self.mutation_strength_bounds
        for i in range(len(strategy_params)):
            strategy_params[i] *= exp(general_term + learning_param2 * gauss(0.0, 1.0))
            strategy_params[i] = min(strategy_params[i], max_strength)
            strategy_params[i] = max(strategy_params[i], min_strength)
        self.are_strategy_params_mutated = True


    def _mutate(self):
        """Add normally distributed random numbers to each variable."""
        # initialization and shortcuts
        if not self.are_strategy_params_mutated:
            self.mutate_strategy_params()
        gauss = random.gauss
        genome = self.genome
        num_variables = len(genome)
        strategy_params = self.strategy_params
        if len(strategy_params) == 1:
            strategy_params = [strategy_params[0]] * num_variables
        # mutate decision variables
        mutation_vector = array.array("d", [0.0] * num_variables)
        for i in range(num_variables):
            perturbation = strategy_params[i] * gauss(0.0, 1.0)
            genome[i] += perturbation
            mutation_vector[i] = perturbation
        self.mutation_vector = mutation_vector


    def recombine(self, others):
        """Produce one child from an arbitrary number of parents.

        The kind of recombination is determined by the attribute
        `recombination_type`.

        Parameters
        ----------
        others : iterable of Individual
            Other parents to recombine this individual with.

        Returns
        -------
        children : list of Individual
            A list containing a single child.

        """
        # shortcuts
        recombination_type = self.recombination_type
        num_variables = len(self.genome)
        intermediate = self.intermediate
        parents = [self]
        parents.extend(others)
        parent_genomes = [parent.genome for parent in parents]
        for individual in parents:
            if individual.strategy_params is None:
                individual.strategy_params = [0.25] * num_variables
        # begin recombination
        child = self.clone()
        # recombine strategy parameters (intermediate)
        for i in range(len(child.strategy_params)):
            strategy_params = [parent.strategy_params[i] for parent in parents]
            child.strategy_params[i] = intermediate(strategy_params)
        child.mutate_strategy_params()
        if recombination_type == "none" or recombination_type is None:
            pass
        else:
            if recombination_type == "arithmetic":
                splits = [random.random() for _ in range(len(parents) - 1)]
                splits.extend((0.0, 1.0))
                splits.sort()
                self.weights = [splits[i+1] - splits[i] for i in range(len(splits) - 1)]
                recombine = self.arithmetic
            elif recombination_type == "intermediate":
                recombine = intermediate
            elif recombination_type == "discrete":
                recombine = self.discrete
            else:
                msg = "Unkown recombination type: "
                raise Exception(msg + str(recombination_type))
            for i in range(num_variables):
                # collect the i-th variable from all parents
                variables = [genome[i] for genome in parent_genomes]
                child.genome[i] = recombine(variables)
            child.invalidate_objective_values()
        return [child]


    def arithmetic(self, variables):
        """Return a convex combination of the given variables.

        The weights are taken from this object's :attr:`weights` attribute.
        They need to be positive and ``sum(weights) == 1``.

        """
        return sum(var * weight for var, weight in zip(variables, self.weights))


    @staticmethod
    def intermediate(variables):
        """Return the mean of the given variables."""
        return sum(variables) / float(len(variables))


    @staticmethod
    def discrete(variables):
        """Return a randomly chosen variable from the given ones."""
        return random.choice(variables)



class CMSAIndividual(Individual):
    """An individual with self-adaptive covariance matrix adaptation.

    This variation was proposed in [Beyer2008]_.

    References
    ----------
    .. [Beyer2008] Hans-Georg Beyer, Bernhard Sendhoff (2008). Covariance
        Matrix Adaptation Revisited - The CMSA Evolution Strategy -.
        In Parallel Problem Solving from Nature - PPSN X, Springer,
        pp. 123-132. https://dx.doi.org/10.1007/978-3-540-87700-4_13

    """
    def __init__(self, num_parents,
                 num_offspring=1,
                 mutation_strength=0.25,
                 learning_param=None,
                 sorting_component=None,
                 cov_matrix=None,
                 **kwargs):
        """Constructor.

        If the learning parameter is set to zero, adaptation of mutation
        strength is disabled. See [Beyer2002]_ for more information.

        Parameters
        ----------
        num_parents : int
            The number of parents used for recombination. Normally, it
            should be chosen equal to the population size.
        num_offspring : int, optional
            The number of offspring to generate from one call to
            :func:`_recombine`.
        mutation_strength : float, optional
            The strategy parameter to be adapted.
        learning_param : float, optional
            The first learning parameter for self-adaptation. Default value
            is chosen according to [Beyer2002]_.
        sorting_component : SortingComponent, optional
            If this sorting component is given, recombination uses the
            weighted mean of the parents, with weights depending on fitness
            (according to the CMA-ES weighting rule). Otherwise, it is just
            the conventional mean, as recommended by [Beyer2008]_.
        cov_matrix : numpy array, optional
            The covariance matrix used for mutation. The identity matrix is
            chosen by default.
        kwargs :
            Arbitrary keyword arguments, passed to the super class.

        """
        Individual.__init__(self, num_parents=num_parents, **kwargs)
        self.num_offspring = num_offspring
        self.learning_param = learning_param
        self.mutation_strength = mutation_strength
        self.sorting_component = sorting_component
        self._cov_matrix = cov_matrix
        self.cov2 = None
        self.mutation_strength_bounds = [sys.float_info.min, sys.float_info.max]
        self.mutation_direction = None
        self.mutation_vector = None
        self.time_constant = None


    @property
    def cov_matrix(self):
        return self._cov_matrix


    @cov_matrix.setter
    def cov_matrix(self, value):
        self._cov_matrix = value
        self.cov2 = None


    def get_weight_assignments(self, individuals):
        """Determine weights from individuals' fitness."""
        num_individuals = len(individuals)
        individuals_copy = individuals[:]
        if self.sorting_component is None:
            return [1.0] * num_individuals
        else:
            # fitter individuals are rewarded with a higher weight
            max_log = math.log(num_individuals + 1)
            weights = [max_log - math.log(rank) for rank in range(1, num_individuals + 1)]
            self.sorting_component.sort(individuals_copy)
            order_dict = {ind: weight for ind, weight in zip(individuals_copy, weights)}
            weights = [order_dict[ind] for ind in individuals]
            return weights


    @staticmethod
    def weighted_mean(genomes, weights=None):
        """Calculate weighted mean of vectors."""
        num_variables = len(genomes[0])
        if weights is None:
            weights = [1.0] * len(genomes)
        # calculate weighted mean
        mean_genome = copy.copy(genomes[0])
        weights_sum = sum(weights)
        for i in range(num_variables):
            var_sum = 0.0
            for genome, weight in zip(genomes, weights):
                var_sum += (genome[i] * weight)
            mean_genome[i] = var_sum / weights_sum
        return mean_genome


    def _mutate(self):
        """Self-adapt covariance matrix and mutate genome."""
        # shortcuts and initialization
        num_vars = len(self.genome)
        if self.learning_param is None:
            self.learning_param = 1.0 / math.sqrt(2.0 * num_vars)
        learn_param = self.learning_param
        if self.cov_matrix is None:
            self.cov_matrix = np.eye(num_vars)
        # R1
        self.mutation_strength *= math.exp(learn_param * random.gauss(0.0, 1.0))
        min_strength, max_strength = self.mutation_strength_bounds
        self.mutation_strength = min(self.mutation_strength, max_strength)
        self.mutation_strength = max(self.mutation_strength, min_strength)
        # R2
        rand_numbers = np.random.randn(num_vars)
        # numpy.linalg.cholesky returns a lower triangular matrix
        # => no transposition necessary
        if self.cov2 is None:
            try:
                self.cov2 = np.linalg.cholesky(self.cov_matrix)
            except np.linalg.LinAlgError as instance:
                raise Aborted(str(instance))
        self.mutation_direction = np.dot(self.cov2, rand_numbers)
        # R3
        self.mutation_vector = self.mutation_strength * self.mutation_direction
        self.mutation_vector = self.mutation_vector.tolist()
        # R4
        for i in range(len(self.genome)):
            self.genome[i] += self.mutation_vector[i]


    def _recombine(self, others):
        """Produce arbitrary number of kids from arbitrary number of parents.

        Parameters
        ----------
        others : iterable of Individual
            Other parents to recombine this individual with.

        Returns
        -------
        children : list of Individual
            A list containing `num_offspring` children.

        """
        # shortcuts and initialization
        parents = [self]
        parents.extend(others)
        parent_genomes = [parent.genome for parent in parents]
        num_vars = len(self.genome)
        time_const = 1.0 + (num_vars * (num_vars + 1)) / (2.0 * len(parents))
        for individual in parents:
            if individual.mutation_direction is None:
                individual.mutation_direction = np.zeros(num_vars)
            if individual.mutation_vector is None:
                individual.mutation_vector = np.zeros(num_vars)
            if individual.cov_matrix is None:
                individual.cov_matrix = np.eye(num_vars)
            if individual.time_constant is None:
                individual.time_constant = time_const
        # R6: calculate weighted mean (default is arithmetic mean)
        weights = self.get_weight_assignments(parents)
        mean_genome = self.weighted_mean(parent_genomes, weights)
        # collect and average the parents' data
        ss = np.zeros((num_vars, num_vars))
        mutation_strength = 0.0
        for i, individual in enumerate(parents):
            direction = individual.mutation_direction
            ss += weights[i] * np.outer(direction, direction)
            mutation_strength += weights[i] * individual.mutation_strength
        weights_sum = sum(weights)
        # ss is the (maybe weighted) average of the generational cross
        # momentum matrices
        ss /= weights_sum
        mutation_strength /= weights_sum
        # R7: matrix update
        alpha = 1.0 / self.time_constant
        cov_matrix = (1.0 - alpha) * self.cov_matrix + alpha * ss
        # pre-compute cov2 for all children
        try:
            cov2 = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError as instance:
            raise Aborted(str(instance))
        children = []
        for _ in range(self.num_offspring):
            child = self.clone()
            child.genome = mean_genome[:]
            child.mutation_strength = mutation_strength
            child.cov_matrix = cov_matrix
            child.cov2 = cov2
            children.append(child)
        return children



class DEIndividual(Individual):
    """An individual that uses variation from Differential Evolution.

    The variants DE/something/n and DE/current_to_something/n are supported,
    where 'something' is determined by the reproduction component.

    """
    def __init__(self, crossover_constant=0.9,
                 weighting_factor=0.8,
                 num_differences=1,
                 operation_mode="default",
                 **kwargs):
        """Constructor.

        Parameters
        ----------
        crossover_constant : float, optional
            The probability to apply this variation operator per variable.
        weighting_factor : float, optional
            A scalar multiplied with the difference vector.
        num_differences : int, optional
            The number of difference vectors to add.
        operation_mode : str, optional
            Must be 'default', 'rand', 'best', 'current_to_rand', or
            'current_to_best'.
        kwargs :
            Arbitrary keyword arguments, passed to the super class.

        """
        Individual.__init__(self, num_parents=2+2*num_differences, **kwargs)
        self.operation_mode = operation_mode
        self.crossover_constant = crossover_constant
        self.weighting_factor = weighting_factor


    def _recombine(self, others):
        """Recombination with vector differences of parent vectors.

        Parameters
        ----------
        others : iterable of Individual
            Other parents to recombine this individual with.

        Returns
        -------
        child : list of Individual
            A list containing exactly one child.

        """
        # shortcuts
        operation_mode = self.operation_mode
        crossover_constant = self.crossover_constant
        weighting_factor = self.weighting_factor
        assert self.num_parents == len(others) + 1
        num_differences = (self.num_parents - 2) // 2
        genome_indices = list(range(len(self.genome)))
        j = random.choice(genome_indices)
        child = self.clone()
        offspring_genome = child.genome
        mutation_direction = []
        for i in genome_indices:
            if random.random() < crossover_constant or i == j:
                if operation_mode in ("default", "rand", "best"):
                    offspring_genome[i] = others[0].genome[i]
                elif operation_mode.startswith("current_to"):
                    diff = others[0].genome[i] - self.genome[i]
                    offspring_genome[i] = self.genome[i] + weighting_factor * diff
                else:
                    msg = "Undefined operation mode for differential evolution: "
                    raise Exception(msg + str(operation_mode))
                for k in range(num_differences):
                    index = (k * 2) + 1
                    diff = others[index].genome[i] - others[index+1].genome[i]
                    offspring_genome[i] += weighting_factor * diff
            # else keep the value of self.genome[i] that was already copied
            # to offspring_genome[i]
            mutation_direction.append(offspring_genome[i] - self.genome[i])
        child.mutation_direction = mutation_direction
        child.repair()
        return [child]


    def _mutate(self):
        """Does nothing because mutation is integrated in recombination."""
        pass
