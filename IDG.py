from dataclasses import dataclass

import numpy as np
import copy
import multiprocessing
from evoalgos.performance import QualityIndicator
import timeit


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
        num_individuals = len(population)
        igd_part = 0.0
        distances = self.dist_matrix_function(reference_set, population)
        distances = np.array(distances)
        # calculate distances
        for i in range(num_optima):
            min_dist = INFINITY
            for j in range(num_individuals):
                distance = distances[i][j]
                if distance < min_dist:
                    min_dist = distance
            print('min_dist:', min_dist)
            igd_part += min_dist ** p
        igd_part = (igd_part / num_optima) ** (1.0 / p)
        return igd_part


class IGD_calc2(QualityIndicator):
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

    def partial(self, population, igd_part):
        pass


class IGD_calc3(QualityIndicator):
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
        # igd_part = (sum(m ** p for m in mins)/num_optima)**(1.0/p)
        return np.column_stack((index, mins))

    def partial(self, population, dist):
        ref2 = []
        ref = self.reference_set
        print(ref[63])
        pop = population
        for i in range(len(pop)):
            dists = copy.copy(dist)
            row = np.where(dist[:, 0] == i)
            print('row: ', row)
            if len(row[0]) != 0:
                ref2 = [ref[a] for a in row[0]]
                distances = self.dist_matrix_function(ref2,population)
                distances = np.transpose(np.array(distances))
                mins = distances.min(axis=0)
                index = np.argmin(distances, axis=0)
            print('dist: ', mins)
            print('index:', index)
            print(dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], )

        num_optima = len(self.reference_set)
        p = 1
        igd3 = (sum(dist[:, 1]) / num_optima) ** (1.0 / p)
        return igd3


class DistMatrixFunctionIGD:
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
            phenome1 = individual1.objective_values
            for individual2 in individuals2:
                phenome2 = individual2.objective_values
                dist = sum(abs(p1 - p2) ** exponent for p1, p2 in
                           zip(phenome1, phenome2))
                if take_root:
                    dist **= 1.0 / exponent
                distances[-1].append(dist)

        return distances


def IGD2(population, template):
    A = copy.deepcopy(population)
    d1 = []
    d2 = []
    for z in template:
        tz1 = [A[0][i] - z[i] for i in range(len(z))]
        distance1 = np.linalg.norm(tz1)
        # print('Values1: ', A[0], z, A[0][0] - z[0], A[0][1] - z[1], tz1)
        # print('distance1: ', distance1)
        for a in range(1, len(A)):
            tz1 = [A[a][i] - z[i] for i in range(len(z))]
            # print('Values2: ', A[a], z, A[a][0] - z[0], A[a][1] - z[1], tz1)
            d = np.linalg.norm(tz1)
            # print('distance2: ', d)
            if d < distance1:
                distance1 = d
        # print('final distance: ', distance1)
        d1.append(distance1)
    m = 1 / len(template)
    igd1 = m * np.linalg.norm(d1, 1)
    # print(igd1, igd2)
    # print('final distance igd: ', distance_max)
    return igd1


def euclid(population, z):
    A = copy.deepcopy(population)
    num_points = len(A)
    tz = [A[0][i] - z[i] for i in range(len(z))]
    distance = np.linalg.norm(tz)
    # print('Values1: ', A[0], z, A[0][0] - z[0], A[0][1] - z[1], tz)
    # print('distance1: ', distance)
    for a in range(1, len(A)):
        tz = [A[a][i] - z[i] for i in range(len(z))]
        # print('Values2: ', A[a], z, A[a][0] - z[0], A[a][1] - z[1], tz)
        d = np.linalg.norm(tz)
        # print('distance2: ', d)
        if d < distance:
            distance = d
    # print('final distance: ', distance)
    return distance


def do_loops(pop2, i, calc2):
    Q = copy.deepcopy(pop2)
    Q.pop(i)
    res = calc2.assess(Q)
    res *= -1
    return (res, i)


def main():
    template = [[0.3774675435213997, 5.537121398477831], [0.4137520061341351, 5.4829752266116625],
                [0.45443444237981645, 5.428518123403109], [0.4970909876045293, 5.373798868123918],
                [0.5411458329125554, 5.318845325437593], [0.5863123289664633, 5.263677900703142],
                [0.6324151156751707, 5.208312995195724], [0.6793343273964771, 5.152764496830384],
                [0.7269820235230149, 5.0970445679599266], [0.7752904126676116, 5.041164114371053],
                [0.8242052809965682, 4.985133087738998], [0.8736820224362549, 4.928960692308328],
                [0.923683092724277, 4.872655532490757], [0.9741762981595463, 4.816225721999622],
                [1.0251336010139764, 4.759678966853256], [1.0765302591721069, 4.70302262999871],
                [1.1283441900358882, 4.64626378263041], [1.1805554896227755, 4.589409245640692],
                [1.233146061928553, 4.532465623598201], [1.286099328441318, 4.475439332966385],
                [1.3393999970908772, 4.4183366258125165], [1.3930338760538712, 4.361163609937902],
                [1.4469877219444123, 4.303926266133747], [1.5012491147348381, 4.246630463104147],
                [1.5558063537183844, 4.189281970478005], [1.6106483702253171, 4.13188647024268],
                [1.665764653816519, 4.074449566864935], [1.721145189421709, 4.016976796313387],
                [1.776780403442373, 3.9594736341569297], [1.8326611172560388, 3.901945502882643],
                [1.8887785068758356, 3.8443977785522745], [1.9451240677635937, 3.786835796896967],
                [2.0016895839846933, 3.729264858934449], [2.0584671010418294, 3.671690236180393],
                [2.1154489018427167, 3.6141171755155805], [2.1726274853506884, 3.556550903762307],
                [2.2299955475425977, 3.498996632016826], [2.287545964359388, 3.441459559779192],
                [2.345271776384314, 3.383944878917439], [2.403166175024415, 3.3264577774994963],
                [2.4612224900042263, 3.269003443523354], [2.5194341780083764, 3.211587068573737],
                [2.577794812332662, 3.154213851431796], [2.6362980734223878, 3.096889001663038],
                [2.694937740192812, 3.0396177432078324], [2.753707682040054, 2.9824053179983],
                [2.812601851462252, 2.9252569896252756], [2.8716142772203583, 2.8681780470791325],
                [2.9307390579761985, 2.811173808588841], [2.989970356352309, 2.75424962558441],
                [3.0493023933640324, 2.6974108868090805], [3.108729443179288, 2.6406630226091683],
                [3.168245828165742, 2.584011509431425], [3.2278459141886504, 2.527461874560149],
                [3.2875241061256615, 2.471019701129179], [3.3472748435673645, 2.414690633447293],
                [3.407092596674409, 2.3584803826796503], [3.46697186216362, 2.3023947329326404],
                [3.5269071593968273, 2.2464395477952115], [3.5868930265469174, 2.1906207773963304],
                [3.6469240168162385, 2.134944466046116], [3.7069946946825896, 2.0794167605373586],
                [3.767099632147907, 2.024043919195085], [3.8272334049641934, 1.9688323217747425],
                [3.8873905888103266, 1.913788480324932], [3.9475657553920307, 1.8589190511489992],
                [4.0077534684354665, 1.8042308480216975], [4.067948279542527, 1.7497308568436385],
                [4.1281447238729365, 1.6954262519481427], [4.188337315614513, 1.6413244143139714],
                [4.2485205431983655, 1.587432951984866], [4.308688864210087, 1.53375972305513],
                [4.368836699941083, 1.4803128616526555], [4.428958429515592, 1.4271008074406306],
                [4.489048383518407, 1.3741323392718947], [4.549100837035281, 1.3214166137723746],
                [4.609110002001775, 1.2689632098117145], [4.669070018736045, 1.2167821800528895],
                [4.728974946505536, 1.1648841110760775], [4.788818752945146, 1.1132801939704766],
                [4.8485953021029164, 1.061982307816574], [4.908298340835572, 1.0110031191921878],
                [4.967921483205984, 0.9603562018039581], [5.0274581924415, 0.9100561816847391],
                [5.0869017598870485, 0.8601189152788649], [5.14624528021647, 0.8105617104294759],
                [5.205481621929007, 0.7614036042125834], [5.264603391823604, 0.7126657174313554],
                [5.32360289166081, 0.6643717145726172], [5.382472064507183, 0.6165484121973037],
                [5.441202427168741, 0.5692266018283331], [5.499784983407802, 0.5224421925093045],
                [5.558210109838763, 0.4762378475219174], [5.61646740160901, 0.430665419447466],
                [5.67454545630632, 0.385789747058171], [5.732431557731116, 0.3416949407587293],
                [5.7901111855432745, 0.2984956455121055], [5.847567191589673, 0.2563595837012827],
                [5.904778241296172, 0.21556091981635456], [5.961715201103837, 0.17665038372713604],
                [6.018327070440263, 0.14194627005398752]]
    pop = [[4.221046100720729, 1.9480579234832522], [2.9343843153469256, 2.479069137507484],
           [3.1453977739166143, 1.3875306988789031],
           [0.7910714879408748, 4.345669722491643], [2.782584463413463, 1.3913912239519854],
           [4.835958454869886, 0.46721081627054634], [2.0150865737131896, 3.5789490316252137],
           [1.2445516367690521, 3.1083425011745707], [5.179913675373138, 0.31176039974716824],
           [3.694694495342958, 0.9663640303418706], [5.3740236575992615, 0.41225721772503876],
           [0.08513133404087403, 6.656849365260689], [4.692057464869935, 0.7936203144110319],
           [0.7267289481469914, 4.505460855924386], [3.9003856283718688, 3.008843654563583],
           [1.861001875573907, 2.188337737829665], [1.7606827417030129, 2.9344008852173626],
           [3.343146758213262, 1.8605458456202097], [1.3638894351535544, 2.8070006492809316],
           [3.800925698100587, 3.958666269451084], [4.286431029155342, 0.7639953898884264],
           [0.4385171702525158, 4.6931977277455985], [4.888185843236371, 0.724467442821656],
           [2.252307452181372, 3.6423982316683854], [3.0174023160071926, 1.2861363156772405],
           [0.02286503466146992, 7.170422724391427], [2.203964761291793, 1.9980562923746403],
           [2.2075300688012245, 3.799320230995723], [2.475991987189486, 2.6792196990734247],
           [1.872031969488105, 2.1893322477522528], [3.7797131387552128, 2.5161365647380554],
           [2.2557524282203083, 2.361420328829011], [4.358572721424002, 0.9545789641300353],
           [5.023446203431752, 0.3801226915887249], [2.6501889103298306, 2.208716510742123],
           [4.270866164308777, 1.0658521020404308], [4.4488786613396725, 1.6813362462719352],
           [3.3864868574632587, 2.341072476551014], [1.8397383263308853, 2.474742397514696],
           [0.994436090045806, 4.086300734353901], [5.33417464155508, 0.7050786455329464],
           [2.2728601580983465, 2.659593091500101], [4.553595218097644, 0.6889127955076988],
           [3.673801970988219, 3.3131870206259855], [4.272441524011554, 1.2179311355008235],
           [0.7622150228162102, 3.8804759119355543], [1.941357487960994, 3.842205333182545],
           [0.09801078280007082, 6.47108053403908], [3.176564338718623, 1.0959727665807042],
           [1.5239995050445407, 2.8584205644073815], [3.275997716449269, 1.4630577511615388],
           [1.6456819942690097, 2.7777955467183104], [3.3116400956218435, 1.0653319179590612],
           [2.4260683137155192, 3.245442362597748], [1.1718526352451217, 3.114740346774309],
           [0.35688311738944745, 5.513542942250234], [1.8749323448207487, 2.2446975131073814],
           [1.4767525549776526, 3.121770348409121], [3.7666732969963737, 2.4003480676160973],
           [3.695732712719856, 3.822188126464207], [1.8376660625686874, 2.7208856023407813],
           [3.5201183432219594, 3.870804249018689], [2.435907604487793, 1.9300711418757648],
           [3.7441064410528897, 1.166976695613072], [4.109992530136383, 0.9063296473369302],
           [0.2967632391748543, 5.6996651158203635], [4.412549905617653, 0.8487466657698546],
           [3.0220023360787343, 2.5811297131248683], [3.0048845074234753, 1.2067221405446058],
           [3.240839641366557, 3.5996787605302187], [4.995979192063475, 0.3521562006227364],
           [3.153366204522343, 1.1081213892688346], [2.944072633874737, 1.387236963520537],
           [5.515818616041047, 0.27709561376773423], [1.9203932649279631, 2.925058332252806],
           [3.1572215215673562, 2.7243347599264087], [1.7564716584792797, 3.1399197634300906],
           [1.5827795068981088, 2.845485794617492], [3.8961233529349233, 1.093264543466927],
           [2.059802646564333, 1.9656378203511808], [2.1289835125133054, 2.1703950223500934],
           [3.4597889558868506, 2.5674391537233556], [1.2833871594201567, 2.8758767141799897],
           [3.4277536256224286, 2.47780067987647], [0.19313896608552295, 6.300312489436575],
           [4.398552204434275, 0.843782186554991], [4.730867475806672, 0.8562263548549157],
           [4.662947606338949, 1.3261171534306093], [4.241529347516601, 1.178419272519938],
           [3.9894969084730123, 2.624042787688695], [4.391617534369283, 1.0483404176526054],
           [1.6495004274480882, 2.45044622924818], [0.7164781157784028, 4.936695839635769],
           [6.5903620255975754, 0.13205142341711873], [2.628714581951535, 2.95646464443321],
           [5.704413980635535, 0.39671325560165793], [4.453347582924317, 0.7291260501535115],
           [1.773664425389248, 3.347137612778472], [2.4216825389946557, 1.6824649852251743],
           [0.6244004368041103, 4.995180015291227], [2.3995515533632474, 2.0233913971269892]]

    m = 1 / len(template)
    d1 = []
    d2 = []
    for z in template:
        d2.append(euclid(pop, z))

    print('d1: ', d1)
    print('d2:', m * np.linalg.norm(d2, 1))

    print(sum(d1), sum(d2))

    Z = [[0, 10], [1, 6], [2, 2], [6, 1], [10, 0]]
    A = [[4, 2], [3, 3], [2, 4]]
    B = [[2, 8], [4, 4], [8, 2]]
    A2 = []
    B1 = []
    B2 = []
    n = 1 / len(Z)
    for z in Z:
        A2.append(euclid(A, z))
    Z2 = []
    for ind in Z:
        Z2.append(individual(objective_values=ind))
    A2 = []
    for ind in A:
        A2.append(individual(objective_values=ind))

    B2 = []
    for ind in B:
        B2.append(individual(objective_values=ind))

    calc1 = IGD_calc(reference_set=Z2)
    A1 = IGD2(A, Z)
    A2 = calc1.assess(A2)
    B1 = IGD2(B, Z)
    B2 = calc1.assess(B2)

    print(A1, A2, B1, B2)

    template2 = []
    for ind in template:
        template2.append(individual(objective_values=ind))

    pop2 = []
    for ind in pop:
        pop2.append(individual(objective_values=ind))

    for ind in template2:
        print(ind.objective_values)

    for ind in pop2:
        print(ind.objective_values)

    calc = IGD_calc(template2)

    IGDA1 = calc.assess(pop2)

    IGDA2 = IGD2(pop, template)

    print(IGDA1, IGDA2)

    igd = {ind: 0.0 for ind in range(len(pop2))}
    calc = IGD_calc(template2)
    start = timeit.default_timer()
    for i in range(len(pop2)):
        Q = copy.deepcopy(pop2)
        Q.pop(i)
        igd[i] = calc.assess(Q)
        igd[i] *= -1
    stop = timeit.default_timer()
    print('Time normal: ', stop - start)

    procs = multiprocessing.cpu_count() // 2
    calc2 = IGD_calc2(template2)
    with multiprocessing.Pool(processes=procs) as pool:
        start = timeit.default_timer()

        result = pool.starmap(do_loops, [(pop2, i, calc2) for i in range(len(pop2))])
        print('res: ', result)
        igd2 = {i: num for num, i in result}
        print('igd3: ', igd2)
        stop = timeit.default_timer()
        print('Time paralel: ', stop - start)

    calc3 = IGD_calc3(template2)
    start = timeit.default_timer()
    dists = calc3.assess(pop2)
    igd3 = calc3.partial(pop2, dists)
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    print(igd)
    print(igd2)
    print(dists)
    print(igd3)


@dataclass
class individual:
    objective_values: list


if __name__ == '__main__':
    main()
