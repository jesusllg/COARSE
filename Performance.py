import copy
import multiprocessing
import timeit
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from evoalgos.performance import QualityIndicator
from evoalgos.sorting import NotComparableError
import globals

from Distances import DistMatrixFunctionIGD

INFINITY = float("inf")


class IGDPlus:
    def __init__(self, num_objectives=4, penalty=None, indexes=None, feasible=True):
        self.num_objectives = num_objectives
        self.f_min = [INFINITY] * self.num_objectives
        if penalty is None:
            self.penalty = 1e-7
        if indexes is None and num_objectives == 2:
            self.indexes = [References(individual=None, epsilon=INFINITY),
                            References(individual=None, epsilon=INFINITY),
                            References(individual=None, epsilon=INFINITY)]
        elif indexes is None and num_objectives == 3:
            self.indexes = [References(individual=None, epsilon=INFINITY),
                            References(individual=None, epsilon=INFINITY),
                            References(individual=None, epsilon=INFINITY),
                            References(individual=None, epsilon=INFINITY)]
        self.templ = None

        self.counter = 0
        if feasible == True:
            lineList = [line.rstrip('\n') for line in open('ZDT1.txt')]  # 2GHH_RPF
            #print('PS: ', lineList)
            feasible1 = []
            for i in range(len(lineList)):
                feasible1.append(lineList[i].split())
            self.feasible1 = [[float(y) for y in x] for x in feasible1]

            lineList = [line.rstrip('\n') for line in open('Convex2GHH_RPF.txt')]
            #print('PF: ', lineList)
            feasible2 = []
            for i in range(len(lineList)):
                feasible2.append(lineList[i].split())
            self.feasible2 = [[float(y) for y in x] for x in feasible2]

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def calc_igd(self, population):
        #for ind in population:
            #print(ind.phenome, ind.objective_values, ind.constraint_violation)
        nref = self.num_objectives + 1
        assert len(self.indexes) == nref
        file = open(f'temp_{globals.num}', "w+")
        self.calc_ref(population, nref)
        self.templ = self.generate_template(population, self.indexes)
        #print(self.templ)
        templ2 = []
        for ind in self.templ:
            templ2.append(individual(objective_values=ind))
        procs = multiprocessing.cpu_count() // 2
        calc = IGD_calc(templ2)
        print('CalcIGD')
        print(self.templ)
        #input('press enter...')
        # with multiprocessing.Pool(processes=procs) as pool:
        #     #input('press enter')
        #     start = timeit.default_timer()
        #     result = pool.starmap(self.do_loops, [(population, i, calc) for i in range(len(population))])
        #     igd_contributions = {population[i]: num for num, i in result}
        #     #print(igd3)
        #     #input('press enter')
        #     stop = timeit.default_timer()
        #     print('Time: ', stop - start)
        # self.counter += 1
        # print(self.templ, file=file)
        # pool.close()
        igd_contributions2 = {ind: 0.0 for ind in population}
        for i in range(len(population)):
            Q= copy.deepcopy(population)
            Q.pop(i)
            igd_contributions2[population[i]] = calc.assess(Q)
            igd_contributions2[population[i]] *=-1
        self.counter += 1
        #print(igd_contributions)
        #print(igd3)
        #input('something')

        return igd_contributions2

        # self.IGD(population, self.templ)
        # dist = self.lessIGD(population,self.templ[0])
        # print('distancia: ', dist, 'z[0]: ', self.templ[0], population)

    def do_loops(self, popu, i, calc):

        Q = copy.deepcopy(popu)
        Q.pop(i)
        res = calc.assess(Q)
        res *= -1
        return (res, i)

    def calc_ref(self, population, nref):  # Good
        for i in range(len(self.indexes)):
            self.indexes[i].epsilon = INFINITY
        # obtain the utopic point f^* stored in f_min.
        for i in range(len(population[0].objective_values)):

            f = ([(ind.objective_values[i]) for ind in population])
            m = min(f)
            if m < self.f_min[i]:
                self.f_min[i] = m

        # print('f_min: ', self.f_min)
        if nref == 3:
            e = [[1, 1e-6], [1e-06, 1], [1 / 2, 1 / 2]]
        elif nref == 4:
            e = [[1, 1e-6, 1e-6], [1e-06, 1, 1e-6], [1e-06, 1e-6, 1], [1 / 3, 1 / 3, 1 / 3]]

        epsilons = {ind: [] for ind in population}

        for ind in population:
            for j in range(nref):
                epsilon = [(ind.objective_values[i] - self.f_min[i]) / e[j][i] for i in
                           range(len(population[0].objective_values))]
                epsilons[ind].append(max(epsilon))

        # calculate quadratic penalty.

        u = self.penalty
        penalty = 1 / u

        for ind in population:
            sum = 0.0
            if ind.constraint_violation is not None:
                for i in range(len(ind.constraint_violation)):
                    if abs(ind.constraint_violation[i]) < 1e-8 or not ind.constraint_violation:
                        sum = 0.0
                    else:
                        sum += abs(ind.constraint_violation[i]) ** 2
                # print(ind.constraint_violation[i])
                sum *= penalty
            for i in range(len(epsilons[ind])):
                epsilons[ind][i] = epsilons[ind][i] + sum

        R = [epsilons[ind] for ind in population]
        print(R[0])

        # print(R)
        for i in range(len(self.indexes)):
            r = self.column(R, i)
            #print(r)
            index = r.index(min(r))
            #print(index)
            eps = min(r)
            #print(eps)
            # print(r)
            # print('value', index, eps)
            if eps < self.indexes[i].epsilon:
                # print(index)
                self.indexes[i].individual = population[index]
                self.indexes[i].epsilon = eps
            # print(min(r))
            # print(population[r.index(min(r))])


        #print('index: ', [index.individual.objective_value for index in self.indexes])

    def generate_template(self, population, indexes):
        template = []
        factor = 0.05  # This factor would be change for more accurate approximation
        fac_convex = 1

        if len(indexes) == 3:
            r1 = [indexes[0].individual.objective_values[0], indexes[0].individual.objective_values[1]]
            r2 = [indexes[1].individual.objective_values[0], indexes[1].individual.objective_values[1]]
            rz = [indexes[1].individual.objective_values[0], indexes[0].individual.objective_values[1]]
            vertices = [r1, r2, rz]
            rp = indexes[-1].individual.objective_values

        elif len(indexes) == 4:
            r1 = [indexes[0].individual.objective_values[0], indexes[0].individual.objective_values[1],
                  indexes[0].individual.objective_values[2]]
            r2 = [indexes[1].individual.objective_values[0], indexes[1].individual.objective_values[1],
                  indexes[1].individual.objective_values[2]]
            r3 = [indexes[2].individual.objective_values[0], indexes[2].individual.objective_values[1],
                  indexes[2].individual.objective_values[2]]
            rz = [min(indexes[1].individual.objective_values[0], indexes[2].individual.objective_values[0]),
                  min(indexes[0].individual.objective_values[1], indexes[2].individual.objective_values[1]),
                  min(indexes[0].individual.objective_values[2], indexes[1].individual.objective_values[2])]
            vertices = [r1, r2, r3, rz]
            rp = indexes[-1].individual.objective_values
        print('vertices: ', vertices)
        print('r3: ', indexes[2].individual.objective_values)

        indx = list(vertices)
        indx.pop()
        #print('indx: ', indx)

        # Here we check if the interest point is inside or no the poligon
        if self.in_hull(rp, vertices) == True:
            kind_f = 1
        else:
            kind_f = 2
        #print('kind: ', kind_f)
        # if self.inside_polygon(indexes[-1].individual.objective_values, vertices,
        #                        len(population[0].objective_values) + 1) == 0:
        #     kind_f = 1
        # else:
        #     kind_f = 2

        # print('kind: ', kind_f)

        # We adjust the front in order to be closer to the interest point.

        template = self.convexity(len(population), 1)

        term = 0

        while 1:
            # print('ref: ', indexes[0].individual.objective_values, indexes[1].individual.objective_values,
            # indexes[2].individual.objective_values)
            # print('ref: ', indexes[0].individual.genome, indexes[1].individual.genome,
            # indexes[2].individual.genome)
            t_aux = copy.deepcopy(template)
            template = self.convexity(len(population), fac_convex)
            #print('N_template: ', template)

            # print('rs1: ', indexes[0].individual.objective_values[0], indexes[1].individual.objective_values[0],
            # indexes[1].individual.objective_values[0])

            # print('rs2: ', indexes[1].individual.objective_values[1], indexes[0].individual.objective_values[1],
            # indexes[0].individual.objective_values[1])

            for i in range(len(template)):
                for j in range(len(template[i])):
                    template[i][j] = template[i][j] * (indx[j][j] - min(x[j] for x in indx)) + min(x[j] for x in indx)
            dist, index_p = self.lessdistance(template, indexes[2].individual.objective_values)

            #print(dist, index_p)

            # print('index_p: ', index_p)

            # plt.scatter(template[:][0], template[:][1], color='lightblue', marker='+')
            # plt.scatter(obj[:][0], obj[:][1], color='darkgreen', marker='*')

            # #print('plot this: ', template)
            # #print('and this: ', template[index_p])

            if term == 3000:
                print('Flag1')
                template = self.convexity(len(population), 1)
                for i in range(len(template)):
                    for j in range(len(template[i])):
                        template[i][j] = template[i][j] * (indx[j][j] - min(x[j] for x in indx)) + min(
                            x[j] for x in indx)
                break
            #print(fac_convex)
            if kind_f == 2:
                print('Flag2')
                if self.dominates(indexes[2].individual.objective_values, template, index_p, kind_f):
                    break
            else:
                print('Flag3')
                if dist < 1e-12:
                    print('Flag4')
                    template = self.convexity(len(population), 1)
                    for i in range(len(template)):
                        for j in range(len(template[i])):
                            template[i][j] = template[i][j] * (indx[j][j] - min(x[j] for x in indx)) + min(
                                x[j] for x in indx)
                    break

                elif self.dominates(template, indexes[2].individual.objective_values, index_p, kind_f):
                    print('Flag5')
                    break

            if kind_f == 2:
                fac_convex = fac_convex + factor
            if kind_f == 1:
                fac_convex = fac_convex - factor
                if fac_convex < 0:
                    fac_convex = 0
            term += 1
            # print('i: ', j)

        if self.counter % 500 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'Plot from Coarse{globals.num}')
            ax1.scatter([ind.phenome[0] for ind in population],
                        [ind.phenome[1] for ind in population],
                        marker='.', color='r')
            ax1.scatter([index.individual.phenome[0] for index in indexes],
                        [index.individual.phenome[1] for index in indexes], marker='*', color='k')
            ax2.scatter([ind.objective_values[0] for ind in population],
                        [ind.objective_values[1] for ind in population],
                        marker='.', color='r')
            ax2.scatter([z[0] for z in template], [z[1] for z in template], color='g')
            ax2.scatter(indexes[2].individual.objective_values[0], indexes[2].individual.objective_values[1],
                        marker='o',
                        color='b')
            ax2.scatter(r1[0], r1[1], marker='<', color='k')
            ax2.scatter(r2[0], r2[1], marker='>', color='k')
            ax2.scatter(rz[0], rz[1], marker='^', color='b')
            fig.show()
        # print('Template: ', template)
        return template

    def convexity(self, num_points, fc):
        # print('kind: ', kind)
        # print('fc: ', fc)
        lineList = [line.rstrip('\n') for line in open('weights_3obj_101')]  # '2obj-300w.txt'
        # print('lineList: ', lineList)
        arch = []
        for i in range(len(lineList)):
            arch.append(lineList[i].split())

        # print(arch)
        arch = [[float(y) for y in x] for x in arch]

        # print('og_arch: ', arch, len(arch))

        index = len(arch) / num_points
        index = round(index)
        cont = 1
        arch_new = []
        for i in range(0, len(arch), index):
            arch_new.append(arch[i])
            cont = cont + 1
            # print('cont: ', cont)
            if cont == num_points:
                arch_new.append(arch[-1])
                break
        arch_aux = copy.deepcopy(arch_new)
        # print('arch_aux: ', arch_aux)
        # concave front
        for i in range(num_points):
            n = np.linalg.norm(arch_new[i], fc)
            # print('norm: ', arch_new[i], fc, n)
            for j in range(len(arch_new[i])):
                arch_new[i][j] = arch_new[i][j] / n
        return arch_new

    def lessdistance(self, template, z):
        num_points = len(template)
        # for point in template
        # print('lessD: ', template, z)

        tz = [template[0][i] - z[i] for i in range(len(z))]

        distance = np.linalg.norm(tz)
        index = 0

        # print('d0', distance)
        for i in range(1, len(template)):
            tz = [template[i][j] - z[j] for j in range(len(z))]
            d = np.linalg.norm(tz)
            # print(f'd{i}', d)
            if d < distance:
                distance = d
                index = i
        # print('lessD: ', distance, index)
        return distance, index

    def dominates(self, ind1, ind2, index_p, kind=1):
        if kind == 2:
            ind1_objectives = ind1
            ind2_objectives = ind2[index_p]
        elif kind == 1:
            ind1_objectives = ind1[index_p]
            ind2_objectives = ind2
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

    def in_hull(self, p, vertex):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        p = [p]
        p = np.array(p)
        vertex = np.array(vertex)
        #print('vertex: ', vertex)
        #print('p: ', p)
        from scipy.spatial import Delaunay
        if not isinstance(vertex, Delaunay):
            hull = Delaunay(vertex)

        # hull1 = ConvexHull(vertex)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
        # for s in hull1.simplices:
        #     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        #     ax.plot(vertex[s, 0], vertex[s, 1], "r-")
        #
        # # Make axis label
        # for i in ["x", "y", "z"]:
        #     eval("ax.set_{:s}label('{:s}')".format(i, i))
        #
        # ax.plot(p.T[0], p.T[1], "ko")
        #
        # # Plot defining corner points
        # ax.plot(vertex.T[0], vertex.T[1], "ko")
        # plt.show()

        return hull.find_simplex(p) >= 0


@dataclass
class References:
    individual: object
    epsilon: float


@dataclass
class individual:
    objective_values: list


class IGD_calc(QualityIndicator):
    """Averaged Hausdorff distance (AHD).

    As defined in the paper [Schuetze2012]_.

    References
    ----------
    .. [Schuetze2012] Schütze, O.; Esquivel, X.; Lara, A.; Coello Coello,
        Carlos A. (2012). Using the Averaged Hausdorff Distance as a
        Performance Measure in Evolutionary Multiobjective Optimization.
        IEEE Transactions on Evolutionary Computation, Vol.16, No.4,
        pp. 504-522. https://dx.doi.org/10.1109/TEVC.2011.2161872

    """
    do_maximize = False

    def __init__(self, reference_set, p=1.0, dist_matrix_function=None):
        """Constructor.

        Parameters
        ----------
        reference_set : sequence of Individual
            The known optima of an artificial optimization problem.
        p : float, optional
            The exponent in the AHD definition (not for the distance).
        dist_matrix_function : callable, optional
            Defines which distance function to use. Default is Euclidean.

        """
        QualityIndicator.__init__(self)
        self.reference_set = reference_set
        if dist_matrix_function is None:
            dist_matrix_function = DistMatrixFunctionIGD()
        self.dist_matrix_function = dist_matrix_function
        self.p = p

    def assess(self, population):
        # shortcuts
        p = self.p
        INFINITY = float("inf")
        if not population:
            return INFINITY
        reference_set = self.reference_set
        num_optima = len(reference_set)
        igd_part = 0.0
        distances = self.dist_matrix_function(reference_set, population)
        distances = np.transpose(np.array(distances))
        mins = distances.min(axis=0)
        index = np.argmin(distances, axis=0)
        # calculate distances
        igd_part = (sum(m ** p for m in mins) / num_optima) ** (1.0 / p)
        return igd_part
