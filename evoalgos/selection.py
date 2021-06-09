# This Python file uses the following encoding: utf-8
"""Selection components for evolutionary algorithms."""

import random

from evoalgos.distance import DistMatrixFunction
from evoalgos.sorting import LexicographicSorting, NearestBetterSorting
from evoalgos.sorting import HyperVolumeContributionSorting


INFINITY = float("inf")


class Selection(object):
    """Abstract base class for selection operators."""

    def select(self, population, number, already_chosen=None):
        """Return `number` individuals from the population.

        Note that `population` is not modified.

        Parameters
        ----------
        population : sequence
            The individuals to select from.
        number : int
            The number of individuals to select.
        already_chosen : sequence of Individual, optional
            Definitely surviving individuals not included in `population`,
            but which might influence these ones.

        """
        population_copy = list(population)
        self.reduce_to(population_copy, number, already_chosen)
        return population_copy


    def reduce_to(self, population, number, already_chosen=None):
        """Reduce population size to `number` and return the rejected.

        This method is abstract. Here, `population` should be modified
        in-place.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            Definitely surviving individuals not included in `population`,
            but which might influence these ones.

        Returns
        -------
        rejected : list
            The removed individuals.

        """
        raise NotImplementedError()



class UniformSelection(Selection):
    """Choose individuals random uniformly from the population."""

    def select(self, population, number, already_chosen=None):
        """Return `number` randomly drawn individuals from population.

        Note that `population` is not modified.

        Parameters
        ----------
        population : sequence
            The individuals to select from.
        number : int
            The number of individuals to select.
        already_chosen : sequence of Individual, optional
            This parameter has no influence for this selection.

        """
        if number >= len(population):
            return population[:]
        else:
            return random.sample(population, number)


    def reduce_to(self, population, number, already_chosen=None):
        """Discard randomly selected members of the population and return them.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            This parameter has no influence for this selection.

        Returns
        -------
        rejected : list
            The removed individuals.

        """
        random.shuffle(population)
        rejected = population[number:]
        del population[number:]
        return rejected



class BackwardElimination(Selection):
    """Wrapper for other selection components."""

    def __init__(self, selection):
        """Constructor.

        Parameters
        ----------
        selection : Selection
            The selection component that is forced to backward elimination
            by this wrapper.

        """
        self.selection = selection


    def reduce_to(self, population, number, already_chosen=None):
        """Reduce population size to `number` and return the rejected.

        This method iteratively calls the respective method of the selection
        component, removing one individual per call.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            Definitely surviving individuals not included in `population`,
            but which might influence these ones.

        Returns
        -------
        rejected : list
            The removed individuals.

        """
        selection = self.selection
        num_rejected = len(population) - number
        rejected = []
        for _ in range(num_rejected):
            rejected_in_step = selection.reduce_to(population,
                                                   len(population) - 1,
                                                   already_chosen)
            rejected.extend(rejected_in_step)
        return rejected



class ForwardSelection(Selection):
    """Wrapper for other selection components."""

    def __init__(self, selection):
        """Constructor.

        Parameters
        ----------
        selection : Selection
            The selection component that is forced to forward selection
            by this wrapper.

        """
        self.selection = selection


    def select(self, population, number, already_chosen=None):
        """Return `number` individuals from the population.

        Note that `population` is not modified.

        Parameters
        ----------
        population : sequence
            The individuals to select from.
        number : int
            The number of individuals to select.
        already_chosen : sequence of Individual, optional
            Definitely surviving individuals not included in `population`,
            but which might influence these ones.

        """
        if already_chosen is None:
            already_chosen = []
        population_copy = list(population)
        selection = self.selection
        selected = []
        for _ in range(number):
            selected_in_step = selection.select(population_copy,
                                                1,
                                                selected + already_chosen)[0]
            population_copy.remove(selected_in_step)
            selected.append(selected_in_step)
        return selected


    def reduce_to(self, population, number, already_chosen=None):
        """Reduce population size to `number` and return the rejected.

        This method iteratively calls the respective method of the selection
        component, selecting the best individual per call.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            Definitely surviving individuals not included in `population`,
            but which might influence these ones.

        Returns
        -------
        rejected : list
            The removed individuals.

        """
        selected = self.select(population, number, already_chosen)
        selected_set = set(selected)
        rejected = []
        for ind in population:
            if ind not in selected_set:
                rejected.append(ind)
        for ind in rejected:
            population.remove(ind)
        return rejected



class TruncationSelection(Selection):
    """Choose strictly the best individuals according to sorting component.

    The features depend heavily on the used sorting component. In
    particular, the selection is only deterministic and elitistic if the
    sorting component is, too.

    """
    def __init__(self, sorting_component=None):
        """Constructor.

        Parameters
        ----------
        sorting_component : SortingComponent, optional
            The sorting component that is used to establish a ranking of
            the individuals. Default is
            :class:`LexicographicSorting <evoalgos.sorting.LexicographicSorting>`.

        """
        Selection.__init__(self)
        self.sorting_component = sorting_component
        if sorting_component is None:
            self.sorting_component = LexicographicSorting()


    def reduce_to(self, population, number, already_chosen=None):
        """Reduce population size to `number` and return the rejected.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            Definitely surviving individuals not included in `population`,
            but which might influence these ones (depending on sorting
            component).

        Returns
        -------
        rejected : list
            The removed individuals.

        """
        if already_chosen:
            population_set = set(population)
            all_individuals = population + already_chosen
            random.shuffle(all_individuals)
            self.sorting_component.sort(all_individuals)
            selected = []
            rejected = []
            for ind in all_individuals:
                if ind in population_set:
                    if len(selected) < number:
                        selected.append(ind)
                    else:
                        rejected.append(ind)
            population[:] = selected
        else:
            random.shuffle(population)
            self.sorting_component.sort(population)
            rejected = population[number:]
            del population[number:]
        # return the worst individuals
        return rejected



class HyperVolumeContributionSelection(TruncationSelection):
    """Use the individuals' hypervolume contribution as criterion.

    This selection is designed for multi-objective problems. The selection
    criterion is the individuals' hypervolume contribution to the dominated
    hypervolume of the non-dominated front it belongs to.

    """
    def __init__(self, offsets=None,
                 prefer_boundary_points=True,
                 selection_variant="auto"):
        """Constructor.

        Parameters
        ----------
        offsets : sequence, optional
            For calculating the hypervolume, a reference point is required.
            The reference point is typically calculated as the worst
            objective values in the set of individuals plus an offset
            vector, which can be specified here. Default offset is
            [1.0, ..., 1.0].
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

        References
        ----------
        .. [Schendekehl2017] Tim Schendekehl (2017). Vergleich von gieriger
            Vorwärts- und Rückwärtsselektion bezüglich exaktem und
            approximiertem Hypervolumen. Master thesis, available as
            Algorithm engineering report TR17-2-001, Technische Universität
            Dortmund. (in German)
            https://ls11-www.cs.tu-dortmund.de/_media/techreports/tr17-01.pdf

        """
        hv_sorting = HyperVolumeContributionSorting(None,
                                                    None,
                                                    prefer_boundary_points)
        TruncationSelection.__init__(self, hv_sorting)
        self.selection_variant = selection_variant
        self.offsets = offsets


    def reduce_to(self, population, number, already_chosen=None):
        """Reduce population size to `number` and return the rejected.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            Individuals which influence the fitness values for the
            population, but do not underlie selection themselves. Empty by
            default.

        Returns
        -------
        rejected : list of Individual
            The removed individuals.

        """
        if number >= len(population):
            # no one is rejected
            return []
        else:
            if already_chosen is None:
                already_chosen = []
            # shortcuts
            population_set = set(population)
            all_individuals = population + already_chosen
            random.shuffle(all_individuals)
            selection_variant = self.selection_variant
            if selection_variant == "auto" or selection_variant is None:
                if len(population) > 2 * number:
                    selection_variant = "forward-greedy"
                else:
                    selection_variant = "backward-greedy"
            hv_sorting = self.sorting_component
            compute_fronts = hv_sorting.non_dom_sorting.identify_groups
            # the reference point is obtained from the population
            reference_point = self.construct_ref_point(all_individuals, self.offsets)
            hv_sorting.reference_point = reference_point
            # check for easier special case
            is_2d = True
            for individual in all_individuals:
                is_2d &= len(individual.objective_values) == 2
            if is_2d:
                sort_front = hv_sorting.sort_front_2d
                calc_contributions = hv_sorting.hypervolume_indicator.calc_contributions_2d
            else:
                sort_front = hv_sorting.sort_front
                calc_contributions = hv_sorting.hypervolume_indicator.calc_contributions
            fronts = compute_fronts(all_individuals)
            selected = []
            rejected = []
            if selection_variant == "forward-greedy":
                for front in fronts:
                    if len(selected) + len(front) <= number:
                        # accept all in this front
                        for ind in front:
                            if ind in population_set:
                                selected.append(ind)
                    elif len(selected) >= number:
                        # reject all in this front
                        for ind in front:
                            if ind in population_set:
                                rejected.append(ind)
                    else:
                        selected_this_front = []
                        prefer_boundary_points = hv_sorting.prefer_boundary_points and is_2d
                        if prefer_boundary_points:
                            LexicographicSorting().sort(front)
                        remaining_indices = list(range(len(front)))
                        max_index = remaining_indices[-1]
                        while len(selected) < number and remaining_indices:
                            hv_values = []
                            random.shuffle(remaining_indices)
                            for i in remaining_indices:
                                if prefer_boundary_points and i in (0, max_index):
                                    hv_values.append(INFINITY)
                                else:
                                    contribs = calc_contributions([front[i]],
                                                                  others=selected_this_front)
                                    hv_values.append(contribs[front[i]])
                            best_index = max(range(len(remaining_indices)),
                                             key=hv_values.__getitem__)
                            best_ind = front[remaining_indices[best_index]]
                            remaining_indices.pop(best_index)
                            selected_this_front.append(best_ind)
                            if best_ind in population_set:
                                selected.append(best_ind)
                        for i in remaining_indices:
                            if front[i] in population_set:
                                rejected.append(front[i])
                population[:] = selected
                return rejected
            elif selection_variant == "backward-greedy":
                while len(rejected) < len(population) - number:
                    last_front = fronts[-1]
                    # sort the last front by hypervolume contribution
                    if len(last_front) > 1:
                        random.shuffle(last_front)
                        sort_front(last_front)
                    # remove worst and update fronts to avoid recalculation
                    removed = last_front.pop()
                    if removed in population_set:
                        rejected.append(removed)
                    if len(last_front) == 0:
                        fronts.pop()
                for front in fronts:
                    for ind in front:
                        if ind in population_set:
                            selected.append(ind)
                population[:] = selected
                return rejected
            elif selection_variant == "super-greedy":
                return TruncationSelection.reduce_to(self, population, number, already_chosen)
            else:
                raise ValueError("unknown selection variant: " + str(selection_variant))


    def construct_ref_point(self, individuals, offsets=None):
        """Construct a reference point from the given individuals.

        Parameters
        ----------
        individuals : sequence
            The individuals whose objective values are considered. For
            each objective, the worst value is taken.
        offsets : sequence, optional
            Non-negative offsets to be added to the worst values. The
            default is [1.0, ..., 1.0].

        """
        dimensions = len(individuals[0].objective_values)
        if offsets is None:
            if self.offsets is None:
                offsets = [1.0] * dimensions
            else:
                offsets = self.offsets
        worst_values = list(individuals[0].objective_values)
        for individual in individuals:
            for i in range(dimensions):
                coordinate = individual.objective_values[i]
                if coordinate > worst_values[i]:
                    worst_values[i] = coordinate
        ref_point = [worst_values[i] + offsets[i] for i in range(dimensions)]
        return ref_point



class NearestBetterSelection(Selection):
    """Selection with information about nearest better neighbors."""

    def __init__(self, dist_matrix_function=None,
                 sorting_component=None,
                 use_neighbor_count=False,
                 multiobjective=False):
        """Constructor.

        Parameters
        ----------
        dist_matrix_function : callable, optional
            A function which takes two sequences of individuals as input
            (say, sizes n and m), and returns an :math:`(n \\times m)`
            matrix of distances as output. Default is Euclidean distance.
        sorting_component : SortingComponent, optional
            This component is used to determine the fitness levels in the
            population, by its method identify_groups(). Default is
            :class:`LexicographicSorting <evoalgos.sorting.LexicographicSorting>`.
        use_neighbor_count : bool, optional
            Indicates which selection criterion is used. If False, the
            population is ranked descending by nearest-better neighbor
            distances (called variant SV7 in [Wessing2016a]_). If True
            (default), for each individual the neighbors being closer than
            the nearest-better neighbor are counted. The ranking is then
            descending by this number. This approach is the basic concept
            behind "rule 3" as described on pages 101-103 of [Wessing2015]_.
            In any case, the order of the sorting component is used as
            secondary criterion to break ties.
        multiobjective : bool, optional
            Decides if the nearest-better data and objective-related data is
            combined in a multiobjective fashion (by non-dominated sorting)
            or by lexicographic sorting. The setting where
            ``multiobjective == True and use_neighbor_count == False``
            corresponds to the variant SV4 in [Wessing2016a]_.

        References
        ----------
        .. [Wessing2015] Wessing, Simon (2015). Two-stage Methods for
            Multimodal Optimization. PhD Thesis, Technische Universität
            Dortmund. http://hdl.handle.net/2003/34148
        .. [Wessing2016a] Simon Wessing, Mike Preuss (2016).
            On multiobjective selection for multimodal optimization.
            Computational Optimization and Applications, Vol. 63, Issue 3,
            pp. 875-902. https://dx.doi.org/10.1007/s10589-015-9785-x

        """
        if dist_matrix_function is None:
            dist_matrix_function = DistMatrixFunction()
        self.dist_matrix_function = dist_matrix_function
        if sorting_component is None:
            sorting_component = LexicographicSorting()
        self.nb_sorting = NearestBetterSorting(dist_matrix_function,
                                               None,
                                               sorting_component,
                                               use_neighbor_count,
                                               multiobjective)


    def reduce_to(self, population, number, already_chosen=None):
        """Reduce population size to `number` and return the rejected.

        Internally,
        :class:`NearestBetterSorting <evoalgos.sorting.NearestBetterSorting>`
        is used to obtain the necessary ranking. Then, a simple truncation
        is done for selection.

        Parameters
        ----------
        population : sequence
            The population to reduce in size.
        number : int
            The number of individuals to reduce to.
        already_chosen : sequence of Individual, optional
            Individuals which influence the distance calculations for the
            population, but do not underlie selection themselves. Empty by
            default.

        Returns
        -------
        rejected : list of Individual
            The removed individuals.

        """
        random.shuffle(population)
        self.nb_sorting.archive = already_chosen
        self.nb_sorting.sort(population)
        rejected = population[number:]
        del population[number:]
        return rejected
