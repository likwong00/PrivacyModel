from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

NUM_OF_AGENTS = 1000

# PLACES
HOME = 0
BEACH = 1
MUSEUM = 2
COMPANY = 3
SURGERY = 4
EXAM = 5
COMPETITION = 6
FUNERAL = 7
TYPHOON = 8
SPEED_TICKET = 9


# INFECTION STATES
NOT_INFECTED = 0
INFECTED_A = 1
INFECTED_S = 2
CRITICAL = 3
CURED = 4
DECEASED = 5

# Capacity of quarantine center
QC_LIMIT = 100 # Capacity of quarantine center

class HumanAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.homeId = unique_id // 4
        self.pleasure = self.random.uniform(0, 1)
        self.privacy = self.random.uniform(0, 1)
        self.recognition = self.random.uniform(0, 1)
        self.security = self.random.uniform(0, 1)

    def step(self):
        self.updateHealth()
        self.move()

    # Function for mobility pattern modeling

    def move(self):
        x, y = self.pos

        # Might need y later
        newY = 0

        p = self.random.uniform(0,1)
        # Chance of each place is uniform
        if p <= (1/8):
            newX = BEACH
        elif p <= (2/8):
            newX = MUSEUM
        elif p <= (3/8):
            newX = COMPANY
        elif p <= (4/8):
            newX = SURGERY
        elif p <= (5/8):
            newX = EXAM
        elif p <= (6/8):
            newX = COMPETITION
        elif p <= (7/8):
            newX = FUNERAL
        else:
            newX = SPEED_TICKET
        self.model.grid.move_agent(self, (newX, newY))

    # Function for modeling the spread of the virus

    def updateHealth(self):
        p = self.random.uniform(0, 1)
        if self.health == INFECTED_A and p > 0.75:
            self.health += 1
        elif self.health == INFECTED_S:
            if p > 0.75 and p <= 0.85:
                self.health += 1
            elif p > 0.85:
                self.health = CURED
                x, y = self.pos
                if x == QC:
                    self.model.QC_Occupancy -= 1
        elif self.health == CRITICAL:
            if p > 0.75 and p <= 0.95:
                self.health += 1
                x, y = self.pos
                if x == QC:
                    self.model.QC_Occupancy -= 1
            elif p > 0.95:
                self.health = DECEASED
                self.model.deceasedCount += 1
                x, y = self.pos
                if x == QC:
                    self.model.QC_Occupancy -= 1
        return

    def infect(self):
        p = self.random.uniform(0, 1)
        if self.health == NOT_INFECTED and p > self.model.p0b:
            self.health += 1


def compute_infected(model):
    infectedAgents = [agent for agent in model.schedule.agents if agent.health in (INFECTED_A, INFECTED_S, CRITICAL)]
    return len(infectedAgents)


class PrivacyModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, startingState, quarantine=False, socialDistancing=False):
        self.num_agents = N
        # self.random.seed(42)
        self.QC_Occupancy = 0
        self.deceasedCount = 0
        self.quarantine = quarantine
        self.grid = MultiGrid(4 if quarantine else 3, 250, False)
        self.schedule = RandomActivation(self)
        self.running = True

        # With socialdistancing the probability of infection is 10% (p0b = 0.9) otherwise it is 50%
        self.p0b = 0.5

        ###########
        # HINT: 
        # You need to update the the next if block to correctly set infection probability when social distancing is TRUE
        # You need to change quarantine center capacity too
        ###########
        if socialDistancing:
            self.p0b = 0.9
        else:
            self.p0b = 0.5
        ###########

        # Create agents
        for i in range(self.num_agents):
            a = HumanAgent(i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, (HOME, a.homeId))
        for a in self.random.sample(self.schedule.agents, int(startingState * N)):
            a.health = INFECTED_A
        self.datacollector = DataCollector(
            model_reporters={"Infected": compute_infected,
                             "QC_Occupancy": "QC_Occupancy"}
        )

    def step(self):
        self.datacollector.collect(self)
        '''Advance the model by one step.'''
        self.schedule.step()
        self.identifyAgentsAndUpdateSpread()

    def identifyAgentsAndUpdateSpread(self):
        # Park
        for y in range(2):
            agents = self.grid.get_cell_list_contents([(PARK, y)])
            self.updateSpread(agents)
        # Grocery
        for y in range(5):
            agents = self.grid.get_cell_list_contents([(GROCERY, y)])
            self.updateSpread(agents)
        # Home
        for y in range(250):
            agents = self.grid.get_cell_list_contents([(HOME, y)])
            self.updateSpread(agents)

    def updateSpread(self, agents):
        if any(a.health in (INFECTED_A, INFECTED_S, CRITICAL) for a in agents):
            [a.infect() for a in agents]


def runSimulation(startingState, quarantine=False, socialDistancing=False):
    modelInst = CovidModel(NUM_OF_AGENTS, startingState, quarantine, socialDistancing)
    i = 0
    while any(a.health in (INFECTED_A, INFECTED_S, CRITICAL) for a in modelInst.schedule.agents):
        i += 1
        modelInst.step()
    modelDF = modelInst.datacollector.get_model_vars_dataframe()

    return i, modelInst.deceasedCount, modelDF.QC_Occupancy, modelDF.Infected

StartingState = 0.1
quarantine=False
socialDistancing=False

averageDays = 0
dayWithMaxInfectionRate = 0
maxInfection = 0
averageCasualties = 0
for i in range(10):
    days, Casualties, QC_Occupancy, Infected = runSimulation(StartingState,quarantine,socialDistancing)
    averageDays+=days
    dayWithMaxInfectionRate+=Infected.idxmax()
    maxInfection += Infected.max()
    averageCasualties += Casualties
    if i == 9:
        # QC_Occupancy.plot(legend=True)
        Infected.plot(legend=True)
print('Average number of days before stabilization: ' + str(averageDays/10))
print('Average of the Day with peak Infection ('+str(maxInfection/10)+'): ' + str(dayWithMaxInfectionRate/10))
print('Average Number of Casualties: ' + str(averageCasualties/10))