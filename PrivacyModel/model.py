from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all the different types of agents
from agents.BasicAgent import BasicAgent
from agents.EpsilonAgent import EpsilonAgent

NUM_OF_AGENTS = 50

# External function for counting how many happy agents are there
def average_happy(model):
    happy_agents = [agent.happy for agent in model.schedule.agents]
    return sum(happy_agents)


class PrivacyModel(Model):
    """A model with some number of agents."""

    def __init__(self, agent_model, N, num_of_friends=6, rewire=0.1):
        self.num_agents = N
        # self.random.seed(42)
        self.grid = MultiGrid(9, 1, False)
        self.schedule = RandomActivation(self)
        self.running = True

        # Initialise relationship between agents as a Watts-Strogatz graph
        self.relationship = nx.watts_strogatz_graph(N, num_of_friends, rewire)

        # For keeping track of time for agent's history
        self.timeStep = 0

        # Create agents
        for i in range(self.num_agents):
            a = agent_model(i, self)
            self.schedule.add(a)
            # Start off with every agent in a random place
            random_place = self.random.randint(0, 8)
            self.grid.place_agent(a, (random_place, 0))
        self.datacollector = DataCollector(
            model_reporters={"Happiness": average_happy}
        )

    def step(self):
        self.datacollector.collect(self)
        '''Advance the model by one step.'''
        self.schedule.step()
        self.timeStep += 1


def run_simulation(steps, agent_model):
    num_of_friends = 20
    rewire = 0.3
    model_inst = PrivacyModel(agent_model, NUM_OF_AGENTS, num_of_friends, rewire)
    for i in range(steps):
        model_inst.step()
    modelDF = model_inst.datacollector.get_model_vars_dataframe()

    return modelDF.Happiness


steps = 10

total_happiness = run_simulation(steps, BasicAgent)
print('Total happiness after ' + str(steps) + ' steps: ' + str(total_happiness / NUM_OF_AGENTS))
