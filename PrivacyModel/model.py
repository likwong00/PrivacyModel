from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Import all the different types of agents
from agents.RandomAgent import RandomAgent
from agents.BasicAgent import BasicAgent
from agents.MajorityAgent import MajorityAgent
from agents.EpsilonAgent import EpsilonAgent

NUM_OF_AGENTS = 20


# External metric functions
def average_happy(model):
    happy_agents = [agent.happy for agent in model.schedule.agents]
    return sum(happy_agents) / NUM_OF_AGENTS


def max_happy(model):
    all_agents = [agent.happy for agent in model.schedule.agents]
    return max(all_agents)


def min_happy(model):
    all_agents = [agent.happy for agent in model.schedule.agents]
    return min(all_agents)


def average_reward(model):
    reward_agents = [agent.reward for agent in model.schedule.agents]
    return sum(reward_agents) / NUM_OF_AGENTS

def below_average(model):
    average = average_happy(model)
    below_average_agents = [agent.happy for agent in model.schedule.agents if agent.happy < average]
    return len(below_average_agents)


class PrivacyModel(Model):
    """A model with some number of agents."""

    def __init__(self, agent_model, N, num_of_friends=6, rewire=0.1):
        self.num_agents = N
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
            model_reporters={"Average_Happiness": average_happy,
                             "Max_Happiness": max_happy,
                             "Min_Happiness": min_happy,
                             "Average_Reward": average_reward,
                             "Below_Average": below_average}
        )

    def step(self):
        self.datacollector.collect(self)
        '''Advance the model by one step.'''
        self.schedule.step()
        self.timeStep += 1


def run_simulation(steps, agent_model):
    num_of_friends = 8
    rewire = 0.3
    model_inst = PrivacyModel(agent_model, NUM_OF_AGENTS, num_of_friends, rewire)
    for i in range(steps):
        model_inst.step()
    modelDF = model_inst.datacollector.get_model_vars_dataframe()

    return modelDF


# Function for writing the results of a agent model into a .csv file
def write_results(df, agent):
    with open('./results/' + agent + '_results5.csv', 'w', newline='') as file:
        i = 0
        writer = csv.writer(file)
        for row in df.iterrows():
            writer.writerow([i, row[1][0], row[1][1], row[1][2], row[1][3], row[1][4]])
            i += 1


# Main function for running all agent models, then write all their results on their respective .csv files
def run_all_agents(steps):
    # print('random running ...')
    # random = run_simulation(steps, RandomAgent)
    # write_results(random, 'random')
    #
    # print('basic running ...')
    # basic = run_simulation(steps, BasicAgent)
    # write_results(basic, 'basic')
    #
    # print('majority running ...')
    # majority = run_simulation(steps, MajorityAgent)
    # write_results(majority, 'majority')

    print('learning running ...')
    learning = run_simulation(steps, EpsilonAgent)
    write_results(learning, 'learning')


steps = 200

run_all_agents(steps)

# one more level of average - run simulation multiple times
# check when a plot stabilises, so we can stop earlier
# check for violating norms
# in history, pick out policies of other agents



