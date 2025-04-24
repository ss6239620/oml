import numpy as np
import random

def fitness(x):
    return x ** 2

class Particle:
    def __init__(self, min_x, max_x):
        self.position = random.uniform(min_x, max_x)
        self.velocity = random.uniform(-1, 1)
        self.best_position = self.position
        self.best_value = fitness(self.position)
    
    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, min_x, max_x):
        self.position += self.velocity
        self.position = np.clip(self.position, min_x, max_x)
        value = fitness(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position

class PSO:
    def __init__(self, num_particles=30, min_x=-5, max_x=5, iterations=10):
        self.particles = [Particle(min_x, max_x) for _ in range(num_particles)]
        self.global_best_position = min(self.particles, key=lambda p: p.best_value).best_position
        self.min_x, self.max_x = min_x, max_x
        self.iterations = iterations
    
    def optimize(self):
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.min_x, self.max_x)
            
            best_particle = min(self.particles, key=lambda p: p.best_value)
            self.global_best_position = best_particle.best_position
        
        return self.global_best_position, fitness(self.global_best_position)

pso = PSO()
best_x, best_fitness = pso.optimize()
print(f"Best solution: x = {round(best_x, 4)}, f(x) = {round(best_fitness, 4)}")

