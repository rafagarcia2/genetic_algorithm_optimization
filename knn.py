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

num_weights = 6

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 4
num_parents_mating = 2

population_size = 6
#Creating the initial population.
new_population = list(np.random.randint(low=0, high=20, size=population_size))

num_generations = 10
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

    # The best result in the current iteration.
    print("Best result : ", str(min(fitness)))
    print("Best result : ", str(new_population[fitness.index(min(fitness))]))

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
print("Best result : ", str(new_population[fitness.index(min(fitness))]))
