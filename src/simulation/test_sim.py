from model.common_genome_data import *
from model.model import *
from model.model_constants import INPUT_NETWORK_SIZE, OUTPUT_NETWORK_SIZE
from game.model_scripts.game_with_ai import TetrisGameWithAI
from game.constants import DEFAULT_SEED, FPS
from model.genome import InnovationDatabase
from misc.visualizers import visualize_phenotype
import numpy as np
import pygame
import sys

class ExpSpecimen:
    def __init__(self, model:Model, fitness:int):
        self.fitness = fitness
        self.model = model

class Experiment:
    def __init__(self, iteration_count:int, population_size:int, common_rates:CommonRates):
        self.iteration_count = iteration_count
        self.population_size = population_size
        self.common_rates = common_rates
        self.rng = np.random.default_rng()
        
    def __call__(self):
        # Initialize pygame
        pygame.init()
        
        # Set up the display
        #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        #pygame.display.set_caption(TITLE)
        
        # Create a clock for controlling the frame rate
        clock = pygame.time.Clock()
        
        # Create a game instance
        #game = TetrisGame()
        #game = TetrisGameWithAI(seed=seed, ai_model=ai_model)
        corrected_size = self.population_size - 1
        
        population = [None] * corrected_size
        next_population = [None] * corrected_size
        common_innovation_db = InnovationDatabase(INPUT_NETWORK_SIZE + OUTPUT_NETWORK_SIZE)
        # create initial population
        for i in range(corrected_size):
            print(f'Creating new specimen.')
            model = Model.generate_network(input_size=INPUT_NETWORK_SIZE, output_size=OUTPUT_NETWORK_SIZE, common_rates=self.common_rates, innovation_db=common_innovation_db)
            spec = ExpSpecimen(model, 0)
            population[i] = spec
        
        current_iteration = 1
        best_specimen = population[0]
        
        # main 'training' loop
        while (current_iteration <= self.iteration_count):
            print(f'Current iteration: {current_iteration}')
            # run test (headless) for each and set fitness
            for specimen in population:
                game = TetrisGameWithAI(seed=DEFAULT_SEED, ai_model=specimen.model)
                while (not game.game_over):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            game.handle_input(event.key)
                    game.update()
                    clock.tick(FPS)
                specimen.fitness = game.score
                if (specimen.fitness > best_specimen.fitness):
                    best_specimen = specimen
                    
            # sort population by fitness (descending order)
            sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)

            # elitism
            next_population[0] = sorted_population[0]

            # tournament selection and crossover
            for i in range(1, corrected_size):
                # this is a simple tournament selection, might be replaced with a more sophisticated selection method
                parent1 = self.rng.choice(sorted_population[:max(2, corrected_size // 2)])
                parent2 = self.rng.choice(sorted_population[:max(2, corrected_size // 2)])
                while parent2 == parent1:
                    parent2 = self.rng.choice(sorted_population[:max(2, corrected_size // 2)])

                # crossover
                if self.rng.random() < self.common_rates.crossover_rate:
                    genome1 = parent1.model.genome
                    genome2 = parent2.model.genome
                    fitness1 = parent1.fitness
                    fitness2 = parent2.fitness

                    child_genome = genome1.crossover(genome2, fitness1, fitness2)
                else:
                    # if no crossover, just clone better parent
                    if parent1.fitness >= parent2.fitness:
                        child_genome = parent1.model.genome.copy()
                    else:
                        child_genome = parent2.model.genome.copy()

                child_model = Model(genome=child_genome, previous_network_fitness=0)
                next_population[i] = ExpSpecimen(child_model, 0)

            # mutation for each
            for specimen in next_population:
                if self.rng.random() < self.common_rates.node_addition_mutation_rate:
                    specimen.model.genome.mutation_add_node()
                if self.rng.random() < self.common_rates.connection_addition_mutation_rate:
                    specimen.model.genome.mutation_add_connection()
                if self.rng.random() < self.common_rates.weight_mutation_rate:
                    specimen.model.genome.mutation_change_random_weight()
                if self.rng.random() < self.common_rates.activation_mutation_rate:
                    specimen.model.genome.mutation_change_activation_function()
            
            # replace old population
            for i in range(corrected_size):
                population[i] = next_population[i]
            current_iteration += 1
        
        print(f'Best fitness is: {best_specimen.fitness}')
        visualize_phenotype(best_specimen.model.genome)
        
    
    