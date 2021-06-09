"""This module just contains one simple distance function at the moment."""


class DistMatrixFunction:
    """A simple distance function.

    This distance should be useful on most common search spaces
    (real-valued, integer, binary).

    """
    def __init__(self, exponent=2, take_root=True):
        """Constructor.

        Parameters
        ----------
        exponent : scalar, optional
            The exponent in the distance calculation. Default is 2
            (Euclidean distance).
        take_root : bool, optional
            Determines if the root of the distances should be taken.

        """
        self.exponent = exponent
        self.take_root = take_root


    def __call__(self, individuals1, individuals2):
        """Calculate distance matrix."""
        take_root = self.take_root
        exponent = self.exponent
        distances = []
        for individual1 in individuals1:
            distances.append([])
            phenome1 = individual1.phenome
            for individual2 in individuals2:
                phenome2 = individual2.phenome
                dist = sum(abs(p1 - p2) ** exponent for p1, p2 in
                           zip(phenome1, phenome2))
                if take_root:
                    dist **= 1.0 / exponent
                distances[-1].append(dist)
        return distances
