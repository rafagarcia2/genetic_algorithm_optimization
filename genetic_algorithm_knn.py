import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def cal_pop_fitness(population, x_train, y_train, x_test, y_test):
    '''
    Calculating the fitness value of each solution in the current population.
    '''

    rmse_val = []
    for element in population:
        model = KNeighborsRegressor(n_neighbors = element[0]) #, algorithm=element[1], weights=element[1])

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

    k_parents = [k[0] for k in parents]
    k = int(sum(k_parents) / len(k_parents))
    offspring = (k, parents[0][1], parents[1][2])

    return offspring

def mutation(offspring_crossover):
    '''
    Mutation changes a single gene in each offspring randomly.
    '''
    
    number = np.random.randint(low=-3, high=3, size=1)
    k = int(offspring_crossover[0] + number)
    crossover = (k, offspring_crossover[1], offspring_crossover[2])

    if crossover[0] > 20:
        return (20, crossover[1], crossover[2])
    elif crossover[0] < 1:
        return (1, crossover[1], crossover[2])
    return crossover