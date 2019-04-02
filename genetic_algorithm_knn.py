import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def cal_pop_fitness(population, x_train, y_train, x_test, y_test):
    '''
    Calculating the fitness value of each solution in the current population.
    '''

    rmse_val = []
    for k in population:
        model = KNeighborsRegressor(n_neighbors = k)

        model.fit(x_train, y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
    
    # fitness = numpy.sum(pop*equation_inputs, axis=1)
    return rmse_val

def select_mating_pool(pop, fitness, num_parents):
    '''
    Selecting the best individuals in the current generation as parents 
    for producing the offspring of the next generation.
    '''
    population = pop.copy()
    parents = []
    for i in range(num_parents):
        index = fitness.index(min(fitness))
        fitness.pop(index)
        parents.append(population.pop(index))

    return parents

def crossover(parents, offspring_size):

    #binarios = [bin(i) for i in parents]

    #offspring = binarios[0][0:len(binarios[0])//2] + binarios[1][len(binarios[1])//2:]

    offspring = sum(parents) / len(parents)

    return int(offspring)

def mutation(offspring_crossover):
    '''
    Mutation changes a single gene in each offspring randomly.
    '''
    
    number = np.random.randint(low=-3, high=3, size=1)
    offspring_crossover += number

    if offspring_crossover > 20:
        return 20
    elif offspring_crossover < 0:
        return 0
    return int(offspring_crossover)