# This Python file uses the following encoding: utf-8
"""Basin identification methods.

This module contains two different approaches to identify different attraction
basins of a multimodal, real-valued function from a finite sample of points in
its domain.

"""
import math

from evoalgos.distance import DistMatrixFunction
from evoalgos.sorting import LexicographicSorting


INFINITY = float("inf")


class TopographicalSelection:
    """Basin identification via a k-nearest-neighbors graph.

    This implementation is based on the description in [Toern1992]_.

    References
    ----------
    .. [Toern1992] Törn, A. and Viitanen, S. (1992). Topographical global
        optimization. In: C.A. Floudas and P.M. Pardalos (Eds.), Recent
        Advances in Global Optimization, Princeton University Press,
        pp. 384-398.
    .. [Wessing2016] Simon Wessing, Günter Rudolph, Mike Preuss (2016).
        Assessing Basin Identification Methods for Locating Multiple Optima.
        In Panos M. Pardalos, Anatoly Zhigljavsky, and Julius Zilinskas
        (eds.): Advances in Stochastic and Deterministic Global
        Optimization, Springer.

    """
    def __init__(self, num_neighbors,
                 dist_matrix_function=None,
                 use_all_distances=True,
                 sort_key=None):
        """Constructor.

        Parameters
        ----------
        num_neighbors : int
            The number of neighbors to consider in the graph.
        dist_matrix_function : callable, optional
            A callable producing a distance matrix from two sequences of
            individuals. Default is Euclidean distance.
        use_all_distances : bool, optional
            [Wessing2016]_ recommended to use all distances in the
            computations. In [Toern1992]_, only half the distances were
            considered to avoid problems caused by an inefficient graph
            data structure.
        sort_key : callable, optional
            A sort key for determining which individual is better. Default
            is lexicographic ordering of objective values.

        """
        self.num_neighbors = num_neighbors
        if dist_matrix_function is None:
            dist_matrix_function = DistMatrixFunction()
        self.dist_matrix_function = dist_matrix_function
        self.use_all_distances = use_all_distances
        if sort_key is None:
            sort_key = LexicographicSorting.key
        self.sort_key = sort_key


    def select(self, population):
        """Return individuals corresponding to different local optima.

        Note that `population` is not modified. The number of selected
        individuals is dependent on the used number of neighbors and cannot
        be controlled directly.

        Parameters
        ----------
        population : iterable
            The individuals to select from.

        """
        use_all_distances = self.use_all_distances
        sort_key = self.sort_key
        num_neighbors = self.num_neighbors
        num_individuals = len(population)
        if num_neighbors >= num_individuals:
            population_copy = list(population)
            population_copy.sort(key=sort_key)
            return population_copy[0:1]
        dist_matrix = self.dist_matrix_function(population, population)
        # set distance to self to infinity
        for i in range(num_individuals):
            dist_matrix[i][i] = INFINITY
        adjacency_list = [set() for _ in range(num_individuals)]
        for i in range(num_individuals):
            # identify the k nearest neighbors
            decorated_dists = [(dist_matrix[i][j], j) for j in range(num_individuals)]
            decorated_dists.sort()
            neighbor_indices = [index for _, index in decorated_dists][:num_neighbors]
            for index in neighbor_indices:
                if sort_key(population[index]) < sort_key(population[i]):
                    # attention: edge direction is opposite to the literature
                    # (pointing to the better one)
                    adjacency_list[i].add(index)
                elif use_all_distances and sort_key(population[i]) < sort_key(population[index]):
                    adjacency_list[index].add(i)
        # according to our edge definition, graph minima are nodes without
        # outgoing edges
        selected = []
        for i in range(num_individuals):
            if len(adjacency_list[i]) == 0:
                selected.append(population[i])
        return selected



class NearestBetterClustering:
    """Basin identification via nearest-better distances.

    This is a hierarchical clustering method introduced in [Preuss2012]_.
    In a first step, the method constructs a spanning tree, which is divided
    into several connected components in a second step.

    References
    ----------
    .. [Preuss2012] Mike Preuss (2012). Improved Topological Niching for
        Real-Valued Global Optimization. In: Applications of Evolutionary
        Computation, Volume 7248 of the series Lecture Notes in Computer
        Science, pp. 386-395, Springer.
        https://dx.doi.org/10.1007/978-3-642-29178-4_39

    """
    def __init__(self, edge_length_factor=2.0,
                 used_rules=(1, 2),
                 dist_matrix_function=None,
                 use_edge_lengths_for_threshold=False,
                 sort_key=None):
        """Constructor.

        Parameters
        ----------
        edge_length_factor : float, optional
            This factor is multiplied with the reference distance, which is
            estimated from the data. The result is a threshold for
            separating long and short edges.
        used_rules : iterable, optional
            The set of used pruning rules. This must be either (1,), (2,),
            or (1, 2).
        dist_matrix_function : callable, optional
            A callable producing a distance matrix from two sequences of
            individuals. Default is Euclidean distance.
        use_edge_lengths_for_threshold : bool, optional
            Setting this value to True recovers the original behavior of
            rule 1 as used in [Preuss2012]_. The other variant was proposed
            in [Wessing2016]_ to eliminate the influence of the actual
            number of attraction basins on the threshold calculation.
        sort_key : callable, optional
            A sort key for determining which individual is better. Default
            is lexicographic ordering of objective values.

        """
        self.edge_length_factor = edge_length_factor
        if 1 not in used_rules and 2 not in used_rules:
            raise Exception("you have to choose at least one rule")
        if dist_matrix_function is None:
            dist_matrix_function = DistMatrixFunction()
        self.dist_matrix_function = dist_matrix_function
        self.use_edge_lengths_for_threshold = use_edge_lengths_for_threshold
        self.used_rules = used_rules
        if sort_key is None:
            sort_key = LexicographicSorting.key
        self.sort_key = sort_key


    @staticmethod
    def median(data):
        """Return the median (middle value) of numeric data.

        When the number of data points is odd, return the middle data point.
        When the number of data points is even, the median is interpolated by
        taking the average of the two middle values.

        This function is needed as a helper function for rule 2 and included
        here for compatibility with Python 2.

        """
        data = sorted(data)
        n = len(data)
        if n == 0:
            raise Exception("no median for empty data")
        if n % 2 == 1:
            return data[n // 2]
        else:
            i = n // 2
            return (data[i - 1] + data[i]) / 2


    @staticmethod
    def connected_components(edges):
        """Identify weakly connected components from a list of edges.

        This method is geared for application to the output of
        :func:`cluster`.

        Parameters
        ----------
        edges : list
            A list of edges. Each edge is a three-tuple
            (tail, weight, head).

        Returns
        -------
        components : list of set
            Each set in this list corresponds to a connected component.

        """
        def dfs(graph, start_node):
            """Depth first search."""
            discovered = set()
            stack = [start_node]
            while stack:
                vertex = stack.pop()
                if vertex not in discovered:
                    discovered.add(vertex)
                    stack.extend(graph[vertex] - discovered)
            return discovered

        # build adjacency list
        graph = dict()
        vertices = set(tail for tail, _, _ in edges)
        vertices.update(head for _, _, head in edges)
        for vertex in vertices:
            graph[vertex] = set()
        for tail, _, head in edges:
            graph[tail].add(head)
            graph[head].add(tail)
        # identify weakly connected components
        components = list()
        for vertex in vertices:
            if not any(vertex in component for component in components):
                components.append(dfs(graph, vertex))
        components.sort(key=len, reverse=True)
        return components


    def build_spanning_tree(self, population):
        """Build the spanning tree using nearest-better distances.

        Returns
        -------
        edges : list
            A list of edges. Each edge is a three-tuple
            (tail, weight, head).

        """
        # shortcuts
        edges = []
        dist_matrix_function = self.dist_matrix_function
        sort_key = self.sort_key
        sorted_individuals = sorted(population, key=sort_key)
        # distance computations
        for i, individual in enumerate(sorted_individuals):
            current_key = sort_key(individual)
            maybe_better_inds = sorted_individuals[:i]
            if len(maybe_better_inds) > 0:
                dists = dist_matrix_function([individual], maybe_better_inds)[0]
                min_dist = INFINITY
                min_dist_ind = None
                for other_ind, dist in zip(maybe_better_inds, dists):
                    if dist < min_dist and sort_key(other_ind) < current_key:
                        min_dist = dist
                        min_dist_ind = other_ind
                edges.append((individual, min_dist, min_dist_ind))
        return edges


    def cluster(self, population):
        """Construct spanning tree from the population and prune it.

        The returned edges represent a disconnected graph. However, the
        edges are not grouped into connected components. The number of
        connected components is dependent on the edge-length factor and
        cannot be controlled directly.

        Returns
        -------
        edges : list
            A list of edges. Each edge is a three-tuple
            (tail, weight, head). The returned edges are a subset of those
            returned by :func:`build_spanning_tree`.

        """
        edges = self.build_spanning_tree(population)
        if self.use_edge_lengths_for_threshold:
            edge_lengths = [edge[1] for edge in edges]
            avg_edge_length = sum(edge_lengths) / len(edge_lengths)
            edge_length_threshold = self.edge_length_factor * avg_edge_length
        else:
            dist_matrix = self.dist_matrix_function(population, population)
            # set distance to self to infinity
            for i in range(len(population)):
                dist_matrix[i][i] = INFINITY
            nn_dists = [min(row) for row in dist_matrix]
            avg_nn_dist = sum(nn_dists) / len(nn_dists)
            edge_length_threshold = self.edge_length_factor * avg_nn_dist
        short_edges = edges
        if 1 in self.used_rules:
            short_edges = [edge for edge in edges if edge[1] <= edge_length_threshold]
        remaining_edges = set(short_edges)
        if 2 in self.used_rules:
            num_vars = len(population[0].phenome)
            factor_b = -0.000469 * num_vars ** 2 + 0.0263 * num_vars + 3.66 / num_vars - 0.457
            factor_b *= math.log10(len(population))
            factor_b += 0.000751 * num_vars ** 2 - 0.0421 * num_vars - 2.26 / num_vars + 1.83
            rule2_cut_edges = []
            for outgoing_edge in edges:
                predecessor = outgoing_edge[0]
                incoming_edges = [edge for edge in edges if edge[2] == predecessor]
                if len(incoming_edges) >= 3:
                    incoming_edge_lengths = [incoming_edge[1] for incoming_edge in incoming_edges]
                    med_length = self.median(incoming_edge_lengths)
                    if outgoing_edge[1] / med_length > factor_b:
                        rule2_cut_edges.append(outgoing_edge)
            remaining_edges.difference_update(rule2_cut_edges)
        return remaining_edges


    def select(self, population):
        """Build spanning tree, prune, and select sink of each component.

        The number of selected individuals is dependent on the edge-length
        factor and cannot be controlled directly.

        Returns
        -------
        graph_minima : list
            A list of individuals. They are the only nodes in the graph
            without outgoing edges.

        """
        remaining_edges = self.cluster(population)
        graph_minima = []
        for candidate in population:
            is_graph_min = True
            for remaining_edge in remaining_edges:
                if candidate == remaining_edge[0]:
                    is_graph_min = False
                    break
            if is_graph_min:
                graph_minima.append(candidate)
        return graph_minima
