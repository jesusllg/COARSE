"""Sorting components for use in evolutionary algorithms.

As some important orderings in evolutionary computation are only partial,
the Python facilities for sorting are not sufficient. Thus, we provide
the classes in this module. Lower objective values are always considered
better.

"""
from collections import defaultdict

import evoalgos.performance
from evoalgos.distance import DistMatrixFunction
from evoalgos.individual import Individual


INFINITY = float("inf")


class NotComparableError(ValueError):
    """Error indicating that two objects cannot be compared.

    Inherits from :class:`ValueError`.

    """
    def __init__(self, object1, object2):
        message = "Cannot compare " + str(object1) + " and " + str(object2)
        ValueError.__init__(self, message)



class SortingComponent(object):
    """Interface for a sorting component."""

    def __init__(self, sort_key=None):
        self.sort_key = sort_key


    def sort(self, population):
        """Sort a population according to the sort keys."""
        if len(population) > 1:
            sort_keys = self.obtain_sort_keys(population)
            population.sort(key=sort_keys.get)


    def obtain_sort_keys(self, population):
        """Return a dictionary containing the sort keys for all individuals."""
        return {ind: self.sort_key(ind) for ind in population}


    def identify_groups(self, population):
        """Identify groups of equal objective values.

        Parameters
        ----------
        population : list of Individual
            The population to partition.

        Returns
        -------
        groups : list of list

        """
        population_copy = population[:]
        sort_keys = self.obtain_sort_keys(population_copy)
        population_copy.sort(key=sort_keys.get)
        prev_key = sort_keys[population_copy[0]]
        # groups of individuals with equal objective value
        groups = [[]]
        for individual in population_copy:
            curr_key = sort_keys[individual]
            if curr_key != prev_key:
                groups.append([])
            groups[-1].append(individual)
            prev_key = curr_key
        return groups


    def identify_best_group(self, population):
        """Identify the group of best objective values.

        Parameters
        ----------
        population : list of Individual
            The population to partition.

        Returns
        -------
        best_group : list of Individual

        """
        population_copy = population[:]
        if not population_copy:
            return []
        sort_keys = self.obtain_sort_keys(population_copy)
        population_copy.sort(key=sort_keys.get)
        best = [population_copy[0]]
        best_key = sort_keys[population_copy[0]]
        for individual in population_copy[1:]:
            if sort_keys[individual] == best_key:
                best.append(individual)
            else:
                break
        return best



class LexicographicSorting(SortingComponent):
    """Lexicographic sorting."""

    def __init__(self):
        SortingComponent.__init__(self, self.key)


    @staticmethod
    def key(individual):
        """Sort key for lexicographic sorting with special treatment of None.

        None is replaced with infinity (the worst possible value).

        """
        try:
            iter(individual.objective_values)
            key = []
            for objective in individual.objective_values:
                if objective is None:
                    objective = INFINITY
                key.append(objective)
        except TypeError:
            key = individual.objective_values
            if key is None:
                key = INFINITY
        return key



class RankSumSorting(SortingComponent):
    """Rank-sum sorting.

    This sorting component ranks the objective values for each objective
    and uses the sum of ranks as sorting criterion. This approach is a basic
    variant of the `Borda count <https://en.wikipedia.org/wiki/Borda_count>`_
    and is especially suited for many-objective optimization.

    """
    def __init__(self, ties="average"):
        """Constructor.

        Parameters
        ----------
        ties : str, optional
            A string determining the treatment of ties in the rank
            calculation. Possible values are "average", "min", "max",
            or "first".

        """
        SortingComponent.__init__(self)
        self.ties = ties


    def obtain_sort_keys(self, population):
        """Return a dictionary containing the sort keys for all individuals."""
        assert population
        ties = self.ties
        num_objectives = len(population[0].objective_values)
        rank_sums = {ind: 0.0 for ind in population}
        for i in range(num_objectives):

            def one_dim_key(ind):
                key = ind.objective_values[i]
                if key is None:
                    key = INFINITY
                return key

            dim_sorter = SortingComponent(one_dim_key)
            groups = dim_sorter.identify_groups(population)
            ind_counter = 1
            for group in groups:
                ranks = range(ind_counter, ind_counter + len(group))
                if ties == "average":
                    ranks = [sum(ranks) / float(len(ranks))] * len(ranks)
                elif ties == "min":
                    ranks = [min(ranks)] * len(ranks)
                elif ties == "max":
                    ranks = [max(ranks)] * len(ranks)
                elif ties == "first":
                    pass
                else:
                    raise Exception("unknown treatment of ties")
                for j, individual in enumerate(group):
                    rank_sums[individual] += ranks[j]
                ind_counter += len(group)
        return rank_sums



class NonDominatedSorting(SortingComponent):
    """Non-dominated sorting according to Pareto-dominance."""

    def __init__(self, dominance_relation=None):
        """Constructor.

        Parameters
        ----------
        dominance_relation : callable, optional
            A callable that takes two individuals as arguments and returns
            True or False. Default is :func:`dominates`.

        """
        SortingComponent.__init__(self)
        self.dominance_relation = dominance_relation
        if dominance_relation is None:
            self.dominance_relation = self.dominates
        self.lexi_sorter = LexicographicSorting()


    @staticmethod
    def weakly_dominates(individual1, individual2):
        """Evaluate weak Pareto-dominance relation.

        None is treated as worst objective value.

        """
        ind1_objectives = individual1.objective_values
        ind2_objectives = individual2.objective_values
        num_objectives = len(ind1_objectives)
        if num_objectives != len(ind2_objectives):
            raise NotComparableError(ind1_objectives, ind2_objectives)
        for i in range(num_objectives):
            value1 = ind1_objectives[i]
            value2 = ind2_objectives[i]
            if value1 is None and value2 is not None:
                # None is worse than value2
                return False
            elif not (value1 is None or value2 is None) and value1 > value2:
                # value1 worse than value2, comparison with None bypassed
                return False
        return True


    @staticmethod
    def dominates(individual1, individual2):
        """Evaluate normal Pareto-dominance relation.

        None is treated as worst objective value.

        """
        ind1_objectives = individual1.objective_values
        ind2_objectives = individual2.objective_values
        num_objectives = len(ind1_objectives)
        if num_objectives != len(ind2_objectives):
            raise NotComparableError(ind1_objectives, ind2_objectives)
        is_one_strictly_less = False
        for i in range(num_objectives):
            value1 = ind1_objectives[i]
            value2 = ind2_objectives[i]
            if value1 is None:
                if value2 is not None:
                    # None is worse than value2
                    return False
            elif not (value1 is None or value2 is None) and value1 > value2:
                # value1 worse than value2, comparison with None bypassed
                return False
            elif (value1 is not None and value2 is None) or value1 < value2:
                # value1 better than value2 or value1 better than None
                is_one_strictly_less = True
        return is_one_strictly_less


    @staticmethod
    def strongly_dominates(individual1, individual2):
        """Evaluate strong Pareto-dominance relation.

        None is treated as worst objective value.

        """
        ind1_objectives = individual1.objective_values
        ind2_objectives = individual2.objective_values
        num_objectives = len(ind1_objectives)
        if num_objectives != len(ind2_objectives):
            raise NotComparableError(ind1_objectives, ind2_objectives)
        for i in range(num_objectives):
            value1 = ind1_objectives[i]
            value2 = ind2_objectives[i]
            if value1 is None:
                # None worse or equal to everything
                return False
            elif value2 is not None and value1 >= value2:
                # value1 worse or equal to value2, comparison with None bypassed
                return False
        return True


    def sort(self, population):
        """Reorder population to be concatenation of non-dominated fronts.

        This sorting is stable.

        """
        fronts = self.identify_groups(population)
        del population[:]
        for front in fronts:
            population.extend(front)


    def obtain_sort_keys(self, population):
        fronts = self.identify_groups(population)
        sort_keys = dict()
        for i, front in enumerate(fronts):
            for ind in front:
                sort_keys[ind] = i
        return sort_keys


    def identify_groups(self, population):
        """Return a partition of the population into non-dominated fronts.

        In the identified partition, front i would be non-dominated if
        fronts 0 to i-1 were not in the population. The worst-case run time
        is :math:`O(mN^3)` for m objectives and N individuals in the
        population. Guarantees stability and exploits 1-D and 2-D special
        cases if possible.

        """
        if len(population) == 0:
            return []
        orig_indices = {ind: i for i, ind in enumerate(population)}
        fronts = []
        # check for easier special cases
        with_dim = defaultdict(int)
        for individual in population:
            with_dim[len(individual.objective_values)] += 1
        if with_dim[1] == len(population):
            # exploit 1-D special case
            fronts = self.lexi_sorter.identify_groups(population)
        else:
            input_set = set(population)
            if with_dim[2] == len(population):
                compute_non_dom_front = self.compute_non_dom_front_2d
            else:
                compute_non_dom_front = self.compute_non_dom_front_arbitrary_dim
            while len(input_set) > 0:
                # find the non-dominated elements
                min_elements = compute_non_dom_front(input_set)
                fronts.append(min_elements)
                # remove them from the input for the next iteration
                input_set.difference_update(min_elements)
        # ensure stability
        for front in fronts:
            front.sort(key=orig_indices.get)
        return fronts


    def identify_best_group(self, population):
        """Return all non-dominated in the given population.

        Guarantees stability and exploits 1-D and 2-D special cases if
        possible.

        """
        if len(population) == 0:
            return []
        orig_indices = {ind: i for i, ind in enumerate(population)}
        # check for easier special cases
        with_dim = defaultdict(int)
        for individual in population:
            with_dim[len(individual.objective_values)] += 1
        if with_dim[1] == len(population):
            min_elements = self.lexi_sorter.identify_best_group(population)
        elif with_dim[2] == len(population):
            min_elements = self.compute_non_dom_front_2d(population)
        else:
            min_elements = self.compute_non_dom_front_arbitrary_dim(population)
        # ensure stability
        min_elements.sort(key=orig_indices.get)
        return min_elements


    def compute_non_dom_front_arbitrary_dim(self, population):
        """Return the minimal elements for arbitrary dimension.

        Does neither ensure stability nor exploit special cases, but can
        handle an arbitrary number of objectives. Run time is
        :math:`O(mN^2)` for m objectives and N individuals.
        For the user it is recommended to call
        :func:`identify_best_group() <evoalgos.sorting.NonDominatedSorting.identify_best_group>`
        instead.

        """
        if len(population) == 0:
            return []
        dominance_relation = self.dominance_relation
        min_elements = set(population)
        population = set(population)
        for candidate in population:
            for individual in min_elements:
                if dominance_relation(individual, candidate):
                    # the candidate is dominated, so we can exit the loop
                    min_elements.remove(candidate)
                    break
        return list(min_elements)


    def compute_non_dom_front_2d(self, population):
        """Return the minimal elements in the special case of two objectives.

        Does not ensure stability. Only call directly if you are absolutely
        sure you have two objectives. Run time is :math:`O(N \\log N)` for N
        individuals.
        For the user it is recommended to call
        :func:`identify_best_group() <evoalgos.sorting.NonDominatedSorting.identify_best_group>`
        instead.

        """
        if len(population) == 0:
            return []
        dominance_relation = self.dominance_relation
        population_copy = list(population)
        self.lexi_sorter.sort(population_copy)
        min_elements = [population_copy[0]]
        for individual in population_copy[1:]:
            if not dominance_relation(min_elements[-1], individual):
                min_elements.append(individual)
        return min_elements



class CrowdingDistanceSorting(SortingComponent):
    """Sort by non-domination rank and crowding distance.

    Implements the sorting order suggested in [Deb2000]_.

    """
    def __init__(self):
        SortingComponent.__init__(self)
        self.non_dom_sorting = NonDominatedSorting()


    def obtain_sort_keys(self, population):
        fronts = self.non_dom_sorting.identify_groups(population)
        crowding_dist_sort = self.sort_front
        sort_keys = dict()
        for i, front in enumerate(fronts):
            cd_dict = dict()
            crowding_dist_sort(front, cd_dict)
            for ind in front:
                sort_keys[ind] = (i, -cd_dict[ind])
        return sort_keys


    def sort_front(self, front, crowding_distances=None):
        """Sort a front according to the crowding distance."""
        num_objectives = len(front[0].objective_values)
        # set initial values
        if crowding_distances is None:
            crowding_distances = dict()
        crowding_distances.update({ind: 0.0 for ind in front})
        # crowding-distance assignment
        for i in range(num_objectives):

            def one_dim_key(ind):
                key = ind.objective_values[i]
                if key is None:
                    key = INFINITY
                return key

            front.sort(key=one_dim_key)
            # mark boundary individuals
            crowding_distances[front[0]] = INFINITY
            crowding_distances[front[-1]] = INFINITY
            # calculate distance for the other individuals
            for j in range(1, len(front) - 1):
                if crowding_distances[front[j]] >= 0:
                    obj1 = front[j-1].objective_values[i]
                    obj2 = front[j+1].objective_values[i]
                    dist = abs(obj1 - obj2)
                    crowding_distances[front[j]] += dist
        front.sort(key=crowding_distances.get, reverse=True)



class HyperVolumeContributionSorting(SortingComponent):
    """Sort by non-domination rank and exclusive contribution to the front.

    This sorting is required for
    :class:`HyperVolumeContributionSelection <evoalgos.selection.HyperVolumeContributionSelection>`.
    Further information can be found in [Emmerich2005]_ and [Naujoks2005]_.

    """
    def __init__(self, reference_point=None,
                 hv_indicator=None,
                 prefer_boundary_points=True):
        """Constructor.

        Parameters
        ----------
        reference_point : Individual or iterable, optional
            This point is used to bound the hypervolume.
        hv_indicator : HyperVolumeIndicator, optional
            An instance with the interface of
            :class:`QualityIndicator <evoalgos.performance.QualityIndicator>`.
            :class:`FonsecaHyperVolume <evoalgos.performance.FonsecaHyperVolume>`
            is chosen as default. It is only used for three or more
            objectives. The 2-D special case is treated directly here, to
            save some time.
        prefer_boundary_points : bool, optional
            This flag only pertains to the two-dimensional case. If it is
            set to True, the two boundary points (but not their potentially
            existing duplicates) of a 2-D front are put to the front of
            the sorted population.

        """
        SortingComponent.__init__(self)
        if hv_indicator is None:
            hv_indicator = evoalgos.performance.FonsecaHyperVolume(reference_point)
        self.hypervolume_indicator = hv_indicator
        self._reference_point = None
        self.reference_point = reference_point
        self.prefer_boundary_points = prefer_boundary_points
        self.non_dom_sorting = NonDominatedSorting()


    @property
    def reference_point(self):
        """The reference point for hypervolume computations."""
        return self._reference_point


    @reference_point.setter
    def reference_point(self, ref_point):
        """Set a new reference point.

        The point is also propagated to the hypervolume indicator.

        """
        self.hypervolume_indicator.reference_point = ref_point
        self._reference_point = ref_point


    def obtain_sort_keys(self, population):
        fronts = self.non_dom_sorting.identify_groups(population)
        # check for easier special case
        is_2d = True
        for individual in population:
            is_2d &= len(individual.objective_values) == 2
        if is_2d:
            calc_contributions = self.hypervolume_indicator.calc_contributions_2d
        else:
            calc_contributions = self.hypervolume_indicator.calc_contributions
        sort_keys = dict()
        for i, front in enumerate(fronts):
            hv_dict = calc_contributions(front,
                                         all_non_dominated=True,
                                         prefer_boundary_points=self.prefer_boundary_points)
            for ind in front:
                sort_keys[ind] = (i, -hv_dict[ind])
        return sort_keys


    def sort_front(self, front, others=None):
        """Sort a non-dominated front.

        Parameters
        ----------
        front : sequence of Individual
            The individuals to sort.
        others : sequence of Individual, optional
            Other individuals that may influence the exclusive hypervolume
            contribution of the sorted population, but which are not
            sorted themselves. Empty by default.

        """
        dim = len(self.reference_point)
        for ind in front:
            assert len(ind.objective_values) == dim
        calc_contributions = self.hypervolume_indicator.calc_contributions
        hv_contributions = calc_contributions(front,
                                              others=others,
                                              all_non_dominated=True)
        # sort (descending)
        front.sort(key=hv_contributions.get, reverse=True)


    def sort_front_2d(self, front, others=None):
        """Sort non-dominated front in the special case of 2 objectives.

        Parameters
        ----------
        front : sequence of Individual
            The individuals to sort.
        others : sequence of Individual, optional
            Other individuals that may influence the exclusive hypervolume
            contribution of the sorted population, but which are not
            sorted themselves. Empty by default.

        """
        if self.reference_point is not None:
            assert len(self.reference_point) == 2
        for ind in front:
            assert len(ind.objective_values) == 2
        calc_contributions = self.hypervolume_indicator.calc_contributions_2d
        hv_contributions = calc_contributions(front,
                                              others=others,
                                              all_non_dominated=True,
                                              prefer_boundary_points=self.prefer_boundary_points)
        # sort (descending)
        front.sort(key=hv_contributions.get, reverse=True)



class NearestBetterSorting(SortingComponent):
    """Sort using distances to the nearest better neighbor."""

    def __init__(self, dist_matrix_function=None,
                 archive=None,
                 sorting_component=None,
                 use_neighbor_count=False,
                 multiobjective=False):
        """Constructor.

        Parameters
        ----------
        dist_matrix_function : callable, optional
            A function which takes two iterables of individuals as input
            (say, sizes n and m), and returns an :math:`(n \\times m)`-matrix
            of distances as output. Default is Euclidean distance.
        archive : list of Individual, optional
            Individuals which influence the distance calculations for the
            sorted population. Empty by default.
        sorting_component : SortingComponent, optional
            This component is used to determine the fitness levels in the
            population, by its method identify_groups(). Default is
            :class:`LexicographicSorting <evoalgos.sorting.LexicographicSorting>`.
        use_neighbor_count: bool, optional
            Indicates which criterion should be used for sorting. If False,
            the sort order is descending by nearest-better neighbor
            distances. If True, for each individual the neighbors
            being closer than the nearest-better neighbor are counted. The
            sort order is then descending by this number. In any case, the
            order of the sorting component is used as secondary criterion to
            break ties.
        multiobjective : bool, optional
            Decides if the nearest-better data and objective-related data is
            combined in a multiobjective fashion (by non-dominated sorting)
            or by lexicographic sorting.

        """
        SortingComponent.__init__(self)
        if dist_matrix_function is None:
            dist_matrix_function = DistMatrixFunction()
        self.dist_matrix_function = dist_matrix_function
        self.archive = archive
        if sorting_component is None:
            sorting_component = LexicographicSorting()
        self.sorting_component = sorting_component
        self.use_neighbor_count = use_neighbor_count
        self.multiobjective = multiobjective


    def calc_nearest_better_distances(self, population,
                                      others=None,
                                      order_dict=None):
        """Calculate the nearest-better distances required for sorting.

        Parameters
        ----------
        population : list of Individual
            The individuals to sort.
        others : list of Individual
            Individuals which influence the distance calculations for the
            sorted population. Empty by default.
        order_dict : dict, optional
            An empty dictionary into which the fitness levels of each
            individual are written. So, this additional information can
            be obtained, if desired.

        Returns
        -------
        min_distances : dict
            A dict containing the distance to the nearest-better neighbor
            for each individual. If no such neighbor exists, the value is
            infinity.

        """
        # initialization
        if others is None:
            others = []
        if order_dict is None:
            order_dict = dict()
        dist_matrix_function = self.dist_matrix_function
        population_set = set(population)
        min_distances = {ind: INFINITY for ind in population}
        pop_others_combined = population + others
        sort_keys = self.sorting_component.obtain_sort_keys(pop_others_combined)
        sorted_individuals = sorted(pop_others_combined, key=sort_keys.get)
        order_dict.update(sort_keys)
        # distance computations
        for i, individual in enumerate(sorted_individuals):
            if individual not in population_set:
                continue
            current_key = order_dict[individual]
            maybe_better_inds = sorted_individuals[:i]
            if len(maybe_better_inds) > 0:
                dists = dist_matrix_function([individual], maybe_better_inds)[0]
                min_dist = INFINITY
                for other_ind, dist in zip(maybe_better_inds, dists):
                    if dist < min_dist and order_dict[other_ind] < current_key:
                        min_dist = dist
                min_distances[individual] = min_dist
        return min_distances


    def count_neighbors_closer_than_nbn(self, population,
                                        others=None,
                                        order_dict=None,
                                        distances_dict=None):
        """Count the neighbors closer than the nearest better one.

        Parameters
        ----------
        population : list of Individual
            The individuals to sort.
        others : list of Individual
            Individuals which influence the distance calculations for the
            sorted population. Empty by default.
        order_dict : dict, optional
            An empty dictionary into which the fitness levels of each
            individual are written, so this additional information can
            be obtained, if desired.
        distances_dict : dict, optional
            An empty dictionary into which the nearest-better distances
            are written, so this information can also be obtained through
            this function call.

        Returns
        -------
        neighbor_counts : dict
            A dict containing the number of neighbors closer than the
            nearest-better one for each individual. If no such neighbor
            exists, this value is by definition equal to
            ``len(population) + len(others)``.

        """
        min_distances = self.calc_nearest_better_distances(population,
                                                           others,
                                                           order_dict)
        if others is None:
            others = []
        if distances_dict is not None:
            distances_dict.update(min_distances)
        all_dists = self.dist_matrix_function(population, population + others)
        neighbor_counts = {ind: 0 for ind in population}
        for i, ind in enumerate(population):
            for dist in all_dists[i]:
                neighbor_counts[ind] += dist < min_distances[ind]
        return neighbor_counts


    def obtain_sort_keys(self, population):
        """Obtain required criteria from the population.

        Parameters
        ----------
        population : list of Individual
            The individuals to process.

        Returns
        -------
        combined_keys : dict
            A dictionary containing the sort key for each individual.

        """
        obj_related_keys = dict()
        if self.use_neighbor_count:
            get_niching_data = self.count_neighbors_closer_than_nbn
        else:
            get_niching_data = self.calc_nearest_better_distances
        niching_related_keys = get_niching_data(population, self.archive, obj_related_keys)
        sorter = SortingComponent(obj_related_keys.get)
        obj_related_groups = sorter.identify_groups(population)
        combined_keys = dict()
        for group_index, group in enumerate(obj_related_groups):
            for ind in group:
                combined_keys[ind] = (-niching_related_keys[ind], group_index)
        if self.multiobjective:
            dummies = []
            for ind in population:
                dummy = Individual(objective_values=combined_keys[ind])
                dummy.original_individual = ind
                dummies.append(dummy)
            nds = NonDominatedSorting()
            dummy_groups = nds.identify_groups(dummies)
            for group_index, group in enumerate(dummy_groups):
                for dummy in group:
                    ind = dummy.original_individual
                    key = combined_keys[ind]
                    combined_keys[ind] = (group_index, key[1], key[0])
        return combined_keys
