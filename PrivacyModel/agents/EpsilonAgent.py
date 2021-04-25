# More sophisticated version of agents where they learn to maximise reward using epsilon-greedy algorithm

from mesa import Agent
import pandas as pd

from . import AgentConstants


class EpsilonAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pleasure = self.random.uniform(0, 1)
        self.privacy = self.random.uniform(0, 1)
        self.recognition = self.random.uniform(0, 1)
        self.security = self.random.uniform(0, 1)
        self.happy = 0
        self.currentAction = AgentConstants.SHARE_NO
        self.friends = self.model.relationship.adj[unique_id]
        self.currentCompanions = []
        # History dictionary for initialising the structure for each agent,
        # used later for agents to learn from rewards.
        self.history = pd.DataFrame(columns=['timeStep', 'agentID', 'othersID', 'place',
                                             'action', 'reward', 'happiness'])

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

        action_choice = self.feedback()

        if action_choice != 4:
            if action_choice == 0:
                best_action = no_value
            elif action_choice == 1:
                best_action = friends_value
            elif action_choice == 2:
                best_action = public_value

        best_action, reward = self.processCompanions(best_action, current_companions)

        # Now set currentAction after looking through companions
        if best_action == (no_value + reward):
            self.currentAction = AgentConstants.SHARE_NO
        elif best_action == (friends_value + reward):
            self.currentAction = AgentConstants.SHARE_FRIENDS
        elif best_action == (public_value + reward):
            self.currentAction = AgentConstants.SHARE_PUBLIC

        # Check if the agent is happy with the action taken
        # best_action is an int from 0 to 40, we determine an agent to be happy if it is greater than 8
        self.happy = best_action

        self.appendHistory(reward)

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
                reward += 10
            elif i.currentAction != self.currentAction:
                reward -= 2

        best_action += reward

        return best_action, reward

    # Function for adding everything that happened in this time-step into the history dictionary
    def appendHistory(self, reward):
        self.history.append({'timeStep': self.model.timeStep, 'agentID': self.unique_id,
                             'othersID': self.currentCompanions, 'place': self.pos, 'action': self.currentAction,
                             'reward': reward, 'happiness': self.happy}, ignore_index=True)

    # Function for agents to look into past interactions with other agents to maximise reward
    def feedback(self):

        # Create a temp dataframe for easier accessing
        query_history = pd.DataFrame(columns=['timeStep', 'agentID', 'othersID', 'place',
                                              'action', 'reward', 'happiness'])

        i = 0
        # Now lookup past entries of the agent's current companions
        for query in self.currentCompanions:
            for row in self.history.iterrows():
                if query in row[1]['othersID']:
                    query_history.loc[i] = row[1]
                    i += 1

        # Break out of this function if there are no past interactions, returns action_choice = 4 to indicate this
        if query_history.empty:
            return 4

        action_pool = [0, 1, 2]
        interaction_choices = []
        # From past interactions, pick out action with the highest average reward
        # Calculated by taking the total weighting of rewards of each companion, then for each action, divide all the
        # accumulated rewards by the total weighting to obtain the probability of that action being chosen
        for query in self.currentCompanions:
            no_weighting = 0
            friends_weighting = 0
            public_weighting = 0

            # Run through the query DF and add the reward value of each action to the weightings
            for row in query_history.iterrows():
                if query in row[1]['othersID']:
                    if row[1]['action'] == 0:
                        no_weighting += row[1]['reward']
                    elif row[1]['action'] == 1:
                        friends_weighting += row[1]['reward']
                    elif row[1]['action'] == 2:
                        public_weighting += row[1]['reward']
                else:
                    continue

            # Do a check if any of the actions have no weighting, eg. never been done then randomly select an action
            # with equal probability
            if no_weighting == 0 or friends_weighting == 0 or public_weighting == 0:
                interaction_choices.append([0, 1, 2])
            else:
                weights = [no_weighting, friends_weighting, public_weighting]

                # To deal with negatives
                offset = min(weights)
                positive_weights = [z - offset + 1 for z in weights]

                # Pick an action given weights
                interaction = self.random.choices(population=action_pool, weights=positive_weights, k=1)
                interaction_choices.append(interaction)

        # At the end going through all companions or not having an action in past history, randomly pick one out of the
        # interaction_choices.
        action_choice = self.random.choice(interaction_choices)
        return action_choice