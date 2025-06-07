class CommonRates:
    def __init__(self, crossover_rate=float, weight_mutation_rate=float, activation_mutation_rate=float, connection_addition_mutation_rate=float, node_addition_mutation_rate=float, start_connection_probability=int, max_start_connection_count=int):
        # this should be high, but not too high, otherwise it will not converge
        self.crossover_rate = crossover_rate
        # this can be found empirically
        self.weight_mutation_rate = weight_mutation_rate
        # this should be low, sigmoid is ok, maybe add some more activation functions
        self.activation_mutation_rate = activation_mutation_rate
        # this should be bigger than node addition mutation rate, compare lower and bigger
        self.connection_addition_mutation_rate = connection_addition_mutation_rate
        # as above, compare
        self.node_addition_mutation_rate = node_addition_mutation_rate
        # starting population data
        self.start_connection_probability = start_connection_probability
        self.max_start_connection_count = max_start_connection_count