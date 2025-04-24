import numpy as np
import random

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, alpha=1, beta=2, evaporation_rate=0.5, q=100):
        self.distances = distances
        self.pheromones = np.ones_like(distances, dtype=float)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.n_cities = distances.shape[0]

    def run(self):
        best_tour = None
        best_length = float('inf')
        
        for iteration in range(self.n_iterations):
            all_tours = []
            all_lengths = []
            
            for _ in range(self.n_ants):
                tour, length = self.construct_solution()
                all_tours.append(tour)
                all_lengths.append(length)
                
                if length < best_length:
                    best_tour, best_length = tour, length
            
            self.update_pheromones(all_tours, all_lengths)
        
        return best_tour, best_length

    def construct_solution(self):
        tour = [random.randint(0, self.n_cities - 1)]
        while len(tour) < self.n_cities:
            current = tour[-1]
            next_city = self.select_next_city(current, tour)
            tour.append(next_city)
        tour.append(tour[0])  # Return to start
        length = self.calculate_tour_length(tour)
        return tour, length

    def select_next_city(self, current, tour):
        probabilities = []
        for j in range(self.n_cities):
            if j not in tour:
                pheromone = self.pheromones[current][j] ** self.alpha
                heuristic = (1 / self.distances[current][j]) ** self.beta
                probabilities.append((j, pheromone * heuristic))
        
        total = sum(p[1] for p in probabilities)
        probabilities = [(p[0], p[1] / total) for p in probabilities]
        
        r = random.random()
        cumulative = 0
        for city, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return city

    def calculate_tour_length(self, tour):
        return sum(self.distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

    def update_pheromones(self, all_tours, all_lengths):
        self.pheromones *= (1 - self.evaporation_rate)
        for tour, length in zip(all_tours, all_lengths):
            for i in range(len(tour) - 1):
                self.pheromones[tour[i]][tour[i + 1]] += self.q / length
                self.pheromones[tour[i + 1]][tour[i]] += self.q / length

# Example usage
distances = np.array([
    [0, 2, 2, 5],
    [2, 0, 3, 4],
    [2, 3, 0, 1],
    [5, 4, 1, 0]
])

aco = AntColony(distances, n_ants=10, n_iterations=100)
best_tour, best_length = aco.run()
print("Best Tour:", best_tour)
print("Best Length:", best_length)
