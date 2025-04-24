import random


CHROMOSOME_LENGTH =  5
POP_SIZE = 20 
GENERATIONS = 100  
MUTATION_RATE = 0.01  


def binary_to_decimal(binary_str):
    return int(binary_str, 2)

def fitness(binary_str):
    x = binary_to_decimal(binary_str)
    return x ** 2

def initialize_population():
    return [''.join(random.choice('01') for _ in range(CHROMOSOME_LENGTH)) for _ in range(POP_SIZE)]

def select_parent(population):
    fitness_values = [fitness(chromosome) for chromosome in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    return population[random.choices(range(POP_SIZE), probabilities)[0]]

def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    mutated = ''.join(bit if random.random() > MUTATION_RATE else '1' if bit == '0' else '0' for bit in chromosome)
    return mutated

def genetic_algorithm():
    population = initialize_population()
    
    for _ in range(GENERATIONS):
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = select_parent(population), select_parent(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
    
    best_solution = max(population, key=fitness)
    return best_solution, binary_to_decimal(best_solution), fitness(best_solution)

best_chromosome, best_x, max_fitness = genetic_algorithm()
print("Best Chromosome:", best_chromosome)
print("Best x:", best_x)
print("Max Fitness (x^2):", max_fitness)
