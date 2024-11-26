import random
import numpy as np
from typing import List, Tuple

class NeuralArchitectureSearch:
    def __init__(self, search_space: dict, population_size: int = 20):
        self.search_space = search_space
        self.population_size = population_size
        self.population = []
        
    def initialize_population(self):
        """Initialize random architectures"""
        for _ in range(self.population_size):
            architecture = {
                'num_layers': random.choice(self.search_space['num_layers']),
                'channels': [random.choice(self.search_space['channels']) 
                           for _ in range(max(self.search_space['num_layers']))],
                'kernel_sizes': [random.choice(self.search_space['kernel_sizes']) 
                               for _ in range(max(self.search_space['num_layers']))]
            }
            self.population.append(architecture)
    
    def evolve(self, fitness_scores: List[float]) -> List[dict]:
        """Evolve the population based on fitness scores"""
        # Sort population by fitness
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), 
                                                reverse=True)]
        
        # Keep top performers
        new_population = sorted_population[:self.population_size // 2]
        
        # Create mutations and crossovers
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # Crossover
                parent1, parent2 = random.sample(new_population, 2)
                child = self._crossover(parent1, parent2)
            else:  # Mutation
                parent = random.choice(new_population)
                child = self._mutate(parent)
            new_population.append(child)
        
        self.population = new_population
        return new_population
    
    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """Perform crossover between two parent architectures"""
        child = {}
        child['num_layers'] = random.choice([parent1['num_layers'], parent2['num_layers']])
        child['channels'] = [random.choice([p1, p2]) for p1, p2 in 
                           zip(parent1['channels'], parent2['channels'])]
        child['kernel_sizes'] = [random.choice([p1, p2]) for p1, p2 in 
                               zip(parent1['kernel_sizes'], parent2['kernel_sizes'])]
        return child
    
    def _mutate(self, parent: dict) -> dict:
        """Mutate a parent architecture"""
        child = parent.copy()
        mutation_point = random.choice(['num_layers', 'channels', 'kernel_sizes'])
        
        if mutation_point == 'num_layers':
            child['num_layers'] = random.choice(self.search_space['num_layers'])
        elif mutation_point == 'channels':
            idx = random.randint(0, len(child['channels']) - 1)
            child['channels'][idx] = random.choice(self.search_space['channels'])
        else:
            idx = random.randint(0, len(child['kernel_sizes']) - 1)
            child['kernel_sizes'][idx] = random.choice(self.search_space['kernel_sizes'])
            
        return child