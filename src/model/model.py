from model.genome import (NodeGene as Node, ConnectionGene as Connection, Genome, InnovationDatabase)
from model.input_data import InputData
from model.model_constants import (INPUT_NODE, OUTPUT_NODE, HIDDEN_NODE)
from model.common_genome_data import *
import numpy as np
from collections import defaultdict, deque
from model.activation_functions import *

node_names = {
    INPUT_NODE : "INPUT",
    OUTPUT_NODE : "OUTPUT",
    HIDDEN_NODE : "HIDDEN"
}

class Model:
    # construct a model/network (phenotype) from the genome
    def __init__(self, genome:Genome, previous_network_fitness:int=0):
        self.genome = genome
        self.phenotype = Model.topological_sort(self.genome.nodes, self.genome.connections)
        self.fitness = previous_network_fitness
        
    # construct the model/network 
    # max possible starting connections should be low
    @classmethod
    def generate_network(self, input_size:int, output_size:int, common_rates:CommonRates, innovation_db:InnovationDatabase, seed=None):
        self.fitness = 0
        input_neurons = []
        output_neurons = []
        connections = []
        id_count = 0
        
        rng = None
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        
        for i in range(0, input_size):
            node = Node(id_count, INPUT_NODE, Sigmoid())
            input_neurons.append(node)
            id_count += 1
        
        for i in range(0, output_size):
            node = Node(id_count, OUTPUT_NODE, Sigmoid())
            output_neurons.append(node)
            id_count += 1
        
        for _ in range(common_rates.max_start_connection_count):
            if rng.uniform(0, 1) > common_rates.start_connection_probability:
                in_node = rng.choice(input_neurons)
                out_node = rng.choice(output_neurons)

                # Check if this exact connection already exists
                if any(conn.in_node.id == in_node.id and conn.out_node.id == out_node.id for conn in connections):
                    continue

                innov = innovation_db.get_or_create_connection_innovation(in_node_id=in_node.id, out_node_id=out_node.id)

                connection = Connection(in_node, out_node, weight=rng.random(), innovation_number=innov)
                connections.append(connection)
        
        genome = Genome(nodes=input_neurons + output_neurons, connections=connections, input_nodes_count=input_size, output_nodes_count=output_size, innovation_db=innovation_db, rng=rng, common_rates=common_rates)
        return Model(genome=genome)
    
    def feed_forward(self, input:dict):
        # 1. set inputs to input neurons
        # 2. sort topologically the network - done in the constructor
        # 3.1. sum each neuron input
        # 3.2. apply activation function
        # 3.3. store output
        # 4. set outputs to the output neurons
        
        # 1.
        for node in self.genome.nodes:
            node.input_sum = 0.0
            node.output = 0.0

        for node in self.genome.nodes:
            if node.id in input:
                node.output = input[node.id]
        
        # 2. - implemented elsewhere, not needed
        
        # 3.
        #nodes_without_input = filter(lambda x: x not in input, sorted_nodes)
        #
        #for node in nodes_without_input:
        #    for connection in node.inputs:
        #        if not connection.is_disabled:
        #            input_val =  connection.in_node.output
        #            node.input_sum += input_val * connection.weight
        #    node.output = node.activation_function(node.input_sum)
        #    print(f'Node {node.id} is a node of type {node.type} and outputs value {node.output}.')
        
        for node in self.phenotype:
            if node.id not in input:  # Skip input nodes
                node.input_sum = 0.0
                for conn in node.inputs:
                    if not conn.is_disabled:
                        input_val = conn.in_node.output
                        node.input_sum += input_val * conn.weight
                        #print(f'Added input_val: {input_val} with weight {conn.weight}')
                node.output = node.activation_function(node.input_sum)
                #print(f'Node {node.id} output is: {node.output}.')
                #print(f'Node {node.id} input sum is: {node.input_sum}.')
        #for node in self.phenotype:
        #    if conn in node.inputs:
        #        if not conn.is_disabled:
        #            print(f'Node id: {node.id}, node type: {node_names[node.type]}, node output: {node.output}.')                
            
        # 4.
        outputs = {}
        for node in self.genome.nodes:
            if node.type == OUTPUT_NODE:
                outputs[node.id] = node.output
                #print(f'Node {node.id} is an output node and outputs value {node.output}.')
        
        return outputs
        
    # simulate inputs to the neurons to check if they are sorted, this should be called once a change (mutation/crossover) occurs and probably stored
    # this is not for recurrent networks!
    @classmethod
    def topological_sort(self, nodes=list[Node], connections=list[Connection]):
        graph = defaultdict(list) #
        in_degree = defaultdict(int) # in_degree - amount of connections a node has as inputs
        
        node_map = {node.id: node for node in nodes}
        
        # add each enabled connection nodes ids (others are not used) to the graph (adjacency lists) 
        for conn in connections:
            if not conn.is_disabled:
                graph[conn.in_node.id].append(conn.out_node.id)
                in_degree[conn.out_node.id] += 1
        
        # add nodes that have no inputs to the queue
        queue = deque()
        for node in nodes:
            if in_degree[node.id] == 0:
                queue.append(node.id)
        
        sorted_ids = []
        # each element in queue has no input, so we search the neigborhood to 'simulate' the input for them and add more elements to the queue
        while queue:
            current = queue.popleft()
            sorted_ids.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return [node_map[nid] for nid in sorted_ids]
    
    # forward the data
    def __call__(self, input=InputData):
        #for connection in self.genome.connections:
            #print(f'There is a connection between node {connection.in_node.id} and node {connection.out_node.id} with weight {connection.weight}.')
        return self.feed_forward(input.to_dict())