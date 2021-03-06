import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/Train_UWu5bXk.csv')
# print(df.head())

#Limpando
df.isnull().sum()
mean = df['Item_Weight'].mean()
df['Item_Weight'].fillna(mean, inplace=True)

mode = df['Outlet_Size'].mode()
df['Outlet_Size'].fillna(mode[0], inplace=True)

df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
df = pd.get_dummies(df)

#Modelo/treino
train , test = train_test_split(df, test_size = 0.3)

x_train = train.drop('Item_Outlet_Sales', axis=1)
y_train = train['Item_Outlet_Sales']

x_test = test.drop('Item_Outlet_Sales', axis = 1)
y_test = test['Item_Outlet_Sales']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

# Genetic Algorthm

import genetic_algorithm_knn as GA

population_size = 4
num_parents_mating = 2
num_generations = 6

#Creating the initial population.
new_population = list(np.random.randint(low=1, high=20, size=population_size))
algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
weights = ['uniform', 'distance']

k_neighboars = list(np.random.randint(low=1, high=20, size=population_size))
index_algorithms = list(np.random.randint(low=0, high=3, size=population_size))
index_weights = list(np.random.randint(low=0, high=1, size=population_size))

new_population = []
for i in range(population_size):
    element = (k_neighboars[i], algorithms[index_algorithms[i]], weights[index_weights[i]])
    new_population.append(element)

last_values = []
for generation in range(num_generations):
    print("\nGeneration : ", generation)
    print(new_population)
    # Measing the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(new_population, x_train, y_train, x_test, y_test)

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents, 2)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)

    best_element = new_population[fitness.index(min(fitness))]
    # The best result in the current iteration.
    print("Best result : ", str(min(fitness)))
    print("Best element : ", str(best_element))

    # save last values
    last_values.append(new_population[fitness.index(min(fitness))])
    if (len(last_values) > 5) and ([best_element]*3 == last_values[-3:]):
        break

    # Creating the new population based on the parents and offspring.
    new_population = parents
    new_population.append(offspring_crossover)
    new_population.append(offspring_mutation)


# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(new_population, x_train, y_train, x_test, y_test)

# The best result in the current iteration.
print("Population: " + str(new_population))
print("Best result : ", str(min(fitness)))
print("Best element : ", str(new_population[fitness.index(min(fitness))]))
