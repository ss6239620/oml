import numpy as np
import random
from itertools import permutations


distance_matrix = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
])

num_cities = distance_matrix.shape[0]
pop_size = 200
generations = 1000
mutation_rate = 0.2

def create_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

def fitness(route):
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1)) + distance_matrix[route[-1], route[0]]

def select_parents(population):
    fitness_values = np.array([1 / fitness(route) for route in population])
    probabilities = fitness_values / fitness_values.sum()
    return population[np.random.choice(len(population), p=probabilities)], population[np.random.choice(len(population), p=probabilities)]

def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    remaining = [gene for gene in parent2 if gene not in child]
    child = [gene if gene is not None else remaining.pop(0) for gene in child]
    return child

def mutate(route):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm():
    population = create_population(pop_size, num_cities)
    for _ in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])
        population = sorted(new_population, key=fitness)[:pop_size]
    return min(population, key=fitness)

best_route = genetic_algorithm()
print("Best Route:", best_route)
print("Minimum Distance:", fitness(best_route))