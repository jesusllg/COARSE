import random

from Algorithm import coarseEMOA
from Individual import SBXIndividual
from optproblems.dtlz import DTLZ1
import globals

from convex import convex

globals.initialize()

globals.num += 4

def main():
    for _ in range(2):


        globals.num += 3

        print(globals.num)

        problem = DTLZ1(max_evaluations=101, num_objectives=3, num_variables=20)
        #
        #
        #
        popsize = 100
        population = []
        f = open(f'Final_Population_{globals.num}', "w+")
        p1 = [random.uniform(0.0, 1.0) for _ in range(problem.num_variables)]
        for _ in range(popsize):
            population.append(SBXIndividual(min_bounds=problem.min_bounds, max_bounds=problem.max_bounds,
                                            genome=[random.uniform(0.0, 1.0) for _ in range(problem.num_variables)]))
        #
        # for individual in population:
        #    print(individual.genome)
        ##

        ea = coarseEMOA(problem, population, popsize, num_offspring=1)
        ea.run()
        # ea = SMSEMOA(problem, population, popsize, num_offspring=1)
        # ea.run()
        # #
        # for individual in ea.population:
        #      print(individual.genome)

        pop = []
        for i in range(len(population)):
            pop.append(population[i].objective_values)

        print('pop = ', pop)
        for individual in ea.population:
            print(str(individual.phenome[0]) + ',' + str(individual.phenome[1]) + ',' + str(
                individual.objective_values[0]) + ',' + str(individual.objective_values[1]) + ',' + str(
                individual.constraint_violation[0]), file=f)
        f.close()

        # ea = SMSEMOA(problem, population, popsize, num_offspring=40)
        # ea.run()
        # for individual in ea.population:
        #     print(individual)


if __name__ == '__main__':
    main()
