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

def happy_individual(model):
    individual_agents = [agent.happy for agent in model.schedule.agents]
    return individual_agents

def agent_privacy(model):
    all_agents = [agent.privacyType for agent in model.schedule.agents]
    return all_agents

def location_agents(model):
    all_agents = [agent.pos for agent in model.schedule.agents]
    return all_agents


class PrivacyModel(Model):
    """A model with some number of agents."""

    def __init__(self, agent_model, N, num_of_friends=6, rewire=0.1):
        self.num_agents = N
        self.grid = MultiGrid(9, 1, False)
        self.schedule = RandomActivation(self)
        self.running = True
        # Using random seeds for replicating results (100, 101, 102)
        self.random.seed(102)

        # Setting privacy type of all agents (-1 for spread, 0-2 for fixed)
        self.privacyPopulation = 2

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
            # model_reporters={"Average_Happiness": average_happy,
            #                  "Max_Happiness": max_happy,
            #                  "Min_Happiness": min_happy,
            #                  "Average_Reward": average_reward,
            #                  "Below_Average": below_average}
            model_reporters={"Individual_Happiness": happy_individual,
                             "Agent_Privacy": agent_privacy}
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
    with open('./results/' + agent + '_casual_results.csv', 'w', newline='') as file:
        i = 0
        writer = csv.writer(file)


        for row in df.iterrows():
            #writer.writerow([i, row[1][0], row[1][1], row[1][2], row[1][3], row[1][4]])
            #writer.writerow([i, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10],
                             #row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19]])
            if i == 0:
                array = np.array_split(row[1][1], 20)
                writer.writerow([i, array[0][0], array[1][0], array[2][0], array[3][0], array[4][0], array[5][0],
                                 array[6][0], array[7][0], array[8][0], array[9][0], array[10][0], array[11][0],
                                 array[12][0], array[13][0], array[14][0], array[15][0], array[16][0], array[17][0],
                                 array[18][0], array[19][0]])
            else:
                # Splitting arr into ints
                array = np.array_split(row[1][0], 20)
                writer.writerow([i, array[0][0], array[1][0], array[2][0], array[3][0], array[4][0], array[5][0],
                                 array[6][0], array[7][0], array[8][0], array[9][0], array[10][0], array[11][0],
                                 array[12][0], array[13][0], array[14][0], array[15][0], array[16][0], array[17][0],
                                 array[18][0], array[19][0]])
            i += 1


# Main function for running all agent models, then write all their results on their respective .csv files
def run_all_agents(steps):
    print('random running ...')
    random = run_simulation(steps, RandomAgent)
    write_results(random, 'random')

    print('basic running ...')
    basic = run_simulation(steps, BasicAgent)
    write_results(basic, 'basic')

    print('majority running ...')
    majority = run_simulation(steps, MajorityAgent)
    write_results(majority, 'majority')

    print('learning running ...')
    learning = run_simulation(steps, EpsilonAgent)
    write_results(learning, 'learning')


steps = 200

run_all_agents(steps)

# one more level of average - run simulation multiple times
# check when a plot stabilises, so we can stop earlier
# check for violating norms
# in history, pick out policies of other agents



