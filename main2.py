import operator
import math
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

class txtParser:
    def __init__(self, txt):
        #Constructor
        self.x = []
        self.y = []
        self.populateList(txt)

    def populateList(self, file):
        with open(file) as fp:
            cnt = 0
            for line in fp:
                if(cnt<2):
                    cnt+=1
                    continue
                line = line.split()
                self.x.append(float(line[0]))
                self.y.append(float(line[1]))
        return

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def symbolicRegression(individual,toolbox, points, true):
    func = toolbox.compile(expr = individual)
    sqerrors = ((func(i)-j)**2 for i, j in zip(points, true))
    return math.fsum(sqerrors)/len(points),

def main():
    data = txtParser("regression.txt")
    pset = gp.PrimitiveSet("main", 1)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addEphemeralConstant("rand", lambda: random.randint(-1,1))
    pset.renameArguments(ARG0='x')
    creator.create("FitnessMin", base.Fitness, weights=(-10.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate",symbolicRegression,toolbox = toolbox, points = data.x, true = data.y)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
    seed = random.randint(1,10000)
    random.seed(seed)
    population = toolbox.population(n=300)
    halloffame = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(population, toolbox, 0.5, 0.1, 80, stats=mstats,
                                   halloffame=halloffame, verbose=True)
    # print log
    expr = halloffame[0]
    tree = gp.PrimitiveTree(expr)
    func = toolbox.compile(expr)
    listres = []
    print(str(tree))
    for i in range(len(data.x)):
        listres.append("x: %2f" % data.x[i])
        listres.append("y: %2f" % func(data.x[i]))
    i=0
    j=0
    SSE = 0.0
    while i < len(listres):
        print("%s %s    Actual: %2f     Square error: %2f" %(listres[i],listres[i+1],data.y[j],(func(data.x[j])-data.y[j])**2))
        SSE+=(func(data.x[j])-data.y[j])**2
        j=j+1
        i=i+2
    SSE/=len(data.x)
    print("Seed: %d, Fitness: %2f"%(seed,SSE))
    return

if __name__ == "__main__":
    main()
