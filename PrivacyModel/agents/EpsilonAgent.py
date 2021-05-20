# More sophisticated version of agents where they learn to maximise reward using epsilon-greedy algorithm

from mesa import Agent
import pandas as pd

from . import AgentConstants

NUM_OF_AGENTS = 20


class EpsilonAgent(Agent):
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
        # History dictionary for initialising the structure for each agent,
        # used later for agents to learn from rewards.
        self.history = pd.DataFrame(columns=['timeStep', 'agentID', 'othersID', 'place',
                                             'action', 'reward', 'happiness'])
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

        selfish_action = max(no_value,
                             friends_value,
                             public_value)

        # Set an intermediate currentAction for other agents to see what action you've chosen
        self.changeCurrentAction(no_value, friends_value, public_value, selfish_action)

        agent_happiness = [agent.happy for agent in self.model.schedule.agents]
        average_happiness = sum(agent_happiness) / NUM_OF_AGENTS

        current_companions, unhappy_companions = self.updateCompanions(average_happiness)

        epsilon_choice = self.epsilon(average_happiness, unhappy_companions)

        if epsilon_choice != 4:
            epsilon_action = 0
            if epsilon_choice == 0:
                epsilon_action = no_value
                self.currentAction = AgentConstants.SHARE_NO
            elif epsilon_choice == 1:
                epsilon_action = friends_value
                self.currentAction = AgentConstants.SHARE_FRIENDS
            elif epsilon_choice == 2:
                epsilon_action = public_value
                self.currentAction = AgentConstants.SHARE_PUBLIC
            # Check if the epsilon action to help other's happiness is worth it,
            # compare happiness of epsilon action plus the happiness gained by other companions with selfish action
            epsilon_action, reward = self.processCompanions(epsilon_action, current_companions)
            if epsilon_action + reward > selfish_action or self.model.timeStep < 50:
                best_action = epsilon_action
            else:
                best_action = selfish_action
                self.changeCurrentAction(no_value, friends_value, public_value, best_action)
                best_action, reward = self.processCompanions(best_action, current_companions)

        else:
            best_action = selfish_action
            self.changeCurrentAction(no_value, friends_value, public_value, best_action)
            best_action, reward = self.processCompanions(best_action, current_companions)

        self.reward = reward

        # Check if the agent is happy with the action taken
        # best_action is an int from 0 to 40, we determine an agent to be happy if it is greater than 8
        self.happy = best_action

        self.appendHistory(reward)

    # Function for changing currentAction
    def changeCurrentAction(self, no_value, friends_value, public_value, action):
        if action == no_value:
            self.currentAction = AgentConstants.SHARE_NO
        elif action == friends_value:
            self.currentAction = AgentConstants.SHARE_FRIENDS
        elif action == public_value:
            self.currentAction = AgentConstants.SHARE_PUBLIC

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
    def updateCompanions(self, average_happiness):
        self.currentCompanions.clear()

        # Get all agents that are friends and in the same location with the user
        current_companions = [agent for agent in self.model.schedule.agents
                              if agent.unique_id in self.friends and agent.pos == self.pos]

        # Get all companions that are 'unhappy' (below the average happiness) for Rawls check
        unhappy_companions = [agent.unique_id for agent in current_companions if agent.happy < average_happiness]

        for k in current_companions:
            self.currentCompanions.append(k.unique_id)

        return current_companions, unhappy_companions

    # Function for checking if anyone in the agent's social circle is in the same location, alter the values for actions
    # based on companion's preferences if necessary
    def processCompanions(self, action, current_companions):
        reward = 0

        # Look through other companion's preferences, then tweak values for actions accordingly
        for i in current_companions:
            if i.currentAction == self.currentAction:
                reward += 5
            elif i.currentAction != self.currentAction:
                reward -= 2

        if reward != 0:
            reward = (reward * 2) / len(current_companions)

        action += reward

        return action, reward

    # Function for adding everything that happened in this time-step into the history dictionary
    def appendHistory(self, reward):
        self.history = self.history.append({'timeStep': self.model.timeStep, 'agentID': self.unique_id,
                                            'othersID': self.currentCompanions, 'place': self.pos,
                                            'action': self.currentAction,
                                            'reward': reward, 'happiness': self.happy}, ignore_index=True)

    # Function for agents to look into past interactions with other agents to maximise reward
    def epsilon(self, average_happiness, unhappy_companions):

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

        interaction_choices = []
        # Epsilon-decreasing: take a probability, if it is smaller than epsilon, explore by performing a
        # random action, otherwise, from past experiences, calculate the action-value estimate of each action for that
        # companion and do the action with the highest estimate
        #
        # In greedy version, initial probability is random, but in decreasing, we start with high chance of exploring,
        # then as more and more time goes by, we decrease that probability
        #
        # Check for rawls condition in exploit phase, if agent is unhappy, break out and do selfish action
        explore_steps = 50
        if self.model.timeStep < explore_steps:
            interaction_choices.append(0)
            interaction_choices.append(1)
            interaction_choices.append(2)
        else:
            if self.happy < average_happiness:
                return 4
            else:
                for query in unhappy_companions:
                    no_weighting = 0
                    no_count = 0
                    friends_weighting = 0
                    friends_count = 0
                    public_weighting = 0
                    public_count = 0

                    # Run through the query DF and add the reward value of each action to the weightings
                    for row in query_history.iterrows():
                        if query in row[1]['othersID']:
                            if row[1]['action'] == 0:
                                no_weighting += row[1]['reward']
                                no_count += 1
                            elif row[1]['action'] == 1:
                                friends_weighting += row[1]['reward']
                                friends_count += 1
                            elif row[1]['action'] == 2:
                                public_weighting += row[1]['reward']
                                public_count += 1
                        else:
                            continue

                    # Calculate the action-value estimates
                    no_estimate = self.safeDivision(no_weighting, no_count)
                    friends_estimate = self.safeDivision(friends_weighting, friends_count)
                    public_estimate = self.safeDivision(public_weighting, public_count)

                    best_action = max(no_estimate, friends_estimate, public_estimate)
                    if best_action == no_estimate:
                        interaction_choices.append(0)
                    elif best_action == friends_estimate:
                        interaction_choices.append(1)
                    elif best_action == public_estimate:
                        interaction_choices.append(2)

        # At the end going through all companions or not having an action in past history, randomly pick one
        # out of the interaction_choices.
        if not interaction_choices:
            return 4
        else:
            action_choice = self.random.choice(interaction_choices)

            return action_choice

    def safeDivision(self, x, y):
        if y == 0:
            return 0
        else:
            return x / y
