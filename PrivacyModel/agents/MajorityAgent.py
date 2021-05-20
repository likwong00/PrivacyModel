# Basic version of agents where they only go for "selfish" actions, meaning they only care about their own preferences

from mesa import Agent
import pandas as pd
from collections import Counter

from . import AgentConstants


class MajorityAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        p = self.random.uniform(0, 1)
        if p <= 0.455:
            self.privacyType = AgentConstants.CAUTIOUS
            self.pleasure = 0.1
            self.recognition = 0.2
            self.privacy = 1
            self.security = 0.7
        elif p <= 0.818:
            self.privacyType = AgentConstants.CONSCIENTIOUS
            self.pleasure = 0.4
            self.recognition = 0.6
            self.privacy = 0.5
            self.security = 0.6
        else:
            self.privacyType = AgentConstants.CASUAL
            self.pleasure = 1
            self.recognition = 0.7
            self.privacy = 0
            self.security = 0.3
        self.happy = 0
        self.currentAction = AgentConstants.SHARE_NO
        self.friends = self.model.relationship.adj[unique_id]
        self.currentCompanions = []
        self.reward = 0


    def step(self):
        self.decision()
        self.move()

    # Function for mobility pattern modeling

    def move(self):
        x = self.pos

        # Might need y later
        newY = 0

        p = self.random.uniform(0, 1)
        # Chance of each place is uniform
        if p <= (1 / 9):
            newX = AgentConstants.BEACH
        elif p <= (2 / 9):
            newX = AgentConstants.MUSEUM
        elif p <= (3 / 9):
            newX = AgentConstants.COMPANY
        elif p <= (4 / 9):
            newX = AgentConstants.SURGERY
        elif p <= (5 / 9):
            newX = AgentConstants.EXAM
        elif p <= (6 / 9):
            newX = AgentConstants.COMPETITION
        elif p <= (7 / 9):
            newX = AgentConstants.FUNERAL
        elif p <= (8 / 9):
            newX = AgentConstants.TYPHOON
        else:
            newX = AgentConstants.SPEED_TICKET
        self.model.grid.move_agent(self, (newX, newY))

    # Function for the decision making process of the agent
    # At each given location, the agent decides whether or not it wants to share a photo with the public,
    # common friends, or no one.
    def decision(self):
        location = self.pos
        str_location = AgentConstants.map_cords_to_places(location)

        actions_values = self.processLocation(str_location)

        # Determine which action to take
        # Basic version: add up all the values in each row, and see which one is largest
        no_value = (actions_values.loc[:, 'SHARE_NO']).sum()
        friends_value = (actions_values.loc[:, 'SHARE_FRIENDS']).sum()
        public_value = (actions_values.loc[:, 'SHARE_PUBLIC']).sum()

        best_action = max(no_value,
                          friends_value,
                          public_value)

        # Set an intermediate currentAction for other agents to see what action you've chosen
        if best_action == no_value:
            self.currentAction = AgentConstants.SHARE_NO
        elif best_action == friends_value:
            self.currentAction = AgentConstants.SHARE_FRIENDS
        elif best_action == public_value:
            self.currentAction = AgentConstants.SHARE_PUBLIC

        current_companions = self.updateCompanions()
        best_action = self.majorityVote(current_companions, no_value, friends_value, public_value, best_action)
        best_action, reward = self.processCompanions(best_action, current_companions)

        # Now set currentAction after looking through companions
        if best_action == (no_value + reward):
            self.currentAction = AgentConstants.SHARE_NO
        elif best_action == (friends_value + reward):
            self.currentAction = AgentConstants.SHARE_FRIENDS
        elif best_action == (public_value + reward):
            self.currentAction = AgentConstants.SHARE_PUBLIC

        self.reward = reward
        # Check if the agent is happy with the action taken
        # best_action is an int from 0 to 40, we determine an agent to be happy if it is greater than 8
        self.happy = best_action

    # Function for agents to evaluate their preferences in a given location, returns an array of with attributes of
    # each actions
    def processLocation(self, location):
        attributes = AgentConstants.places.loc[:, location]

        # Calculate the new values with places_attribute * agent_attribute, then return all of it as an array
        preferences = pd.Series(data=[self.pleasure, self.recognition, self.privacy, self.security],
                                index=AgentConstants.places.index)
        new_preferences = pd.Series(data=(preferences.values * attributes.values), index=AgentConstants.places.index)

        # Compute the values of each action, action_values * new_preference.values
        # Format:
        #               SHARE_NO  SHARE_FRIENDS SHARE_PUBLIC
        # pleasure          x          x             x
        # recognition       x          x             x
        # privacy           x          x             x
        # security          x          x             x
        no_values = AgentConstants.actions.loc[:, 'SHARE_NO']
        friend_values = AgentConstants.actions.loc[:, 'SHARE_FRIENDS']
        public_values = AgentConstants.actions.loc[:, 'SHARE_PUBLIC']
        actions_values = pd.DataFrame(data={'SHARE_NO': (no_values * new_preferences.values),
                                            'SHARE_FRIENDS': (friend_values * new_preferences.values),
                                            'SHARE_PUBLIC': (public_values * new_preferences.values)},
                                      index=AgentConstants.actions.index, columns=AgentConstants.actions.columns)
        return actions_values

    # Function for getting the agent's current companions, and updating that list for later use
    def updateCompanions(self):
        self.currentCompanions.clear()

        # Get all agents that are friends and in the same location with the user
        current_companions = [agent for agent in self.model.schedule.agents
                              if agent.unique_id in self.friends and agent.pos == self.pos]

        for k in current_companions:
            self.currentCompanions.append(k.unique_id)

        return current_companions

    # Function for checking if anyone in the agent's social circle is in the same location, alter the values for actions
    # based on companion's preferences if necessary
    def processCompanions(self, best_action, current_companions):

        reward = 0

        # Look through other companion's preferences, then tweak values for actions accordingly
        for i in current_companions:
            if i.currentAction == self.currentAction:
                reward += 5
            elif i.currentAction != self.currentAction:
                reward -= 2

        if reward != 0:
            reward = (reward * 2) / len(current_companions)
        best_action += reward

        return best_action, reward

    def majorityVote(self, current_companions, no_value, friends_value, public_value, best_action):
        action_pool = []
        for i in current_companions:
            action_pool.append(i.currentAction)

        freq_dict = Counter(action_pool)
        for (key, value) in freq_dict.items():
            if value > len(action_pool) / 2:
                self.currentAction = key
                if key == 0:
                    best_action = no_value
                elif key == 1:
                    best_action = friends_value
                elif key == 2:
                    best_action = public_value
                return best_action

        return best_action
