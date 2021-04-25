# File for storing all the constants of the simulation needed for agents

import pandas as pd

# PLACES
BEACH = 0
MUSEUM = 1
COMPANY = 2
SURGERY = 3
EXAM = 4
COMPETITION = 5
FUNERAL = 6
TYPHOON = 7
SPEED_TICKET = 8

# Storing attributes for each place
# format- 'place': [pleasure, recognition, privacy, security]
places_dict = {
    'BEACH': [5, 5, 1, 1],
    'MUSEUM': [4, 4, 2, 2],
    'COMPANY': [2, 1, 4, 4],
    'SURGERY': [0, 0, 5, 4],
    'EXAM': [1, 2, 3, 3],
    'COMPETITION': [5, 5, 0, 0],
    'FUNERAL': [0, 1, 4, 5],
    'TYPHOON': [4, 2, 3, 5],
    'SPEED_TICKET': [1, 0, 4, 5]
}
places = pd.DataFrame(data=places_dict, index=['pleasure', 'recognition', 'privacy', 'security'])


# Since in the model, each place has a cord and in the df its all in string, need a function to map a cord to a
# corresponding location
def map_cords_to_places(cords):
    if cords == (0, 0):
        return 'BEACH'
    elif cords == (1, 0):
        return 'MUSEUM'
    elif cords == (2, 0):
        return 'COMPANY'
    elif cords == (3, 0):
        return 'SURGERY'
    elif cords == (4, 0):
        return 'EXAM'
    elif cords == (5, 0):
        return 'COMPETITION'
    elif cords == (6, 0):
        return 'FUNERAL'
    elif cords == (7, 0):
        return 'TYPHOON'
    elif cords == (8, 0):
        return 'SPEED_TICKET'


# ACTIONS
SHARE_NO = 0
SHARE_FRIENDS = 1
SHARE_PUBLIC = 2

# Storing constants for each action
# format- 'action': [pleasure, recognition, privacy, security]
actions_dict = {
    'SHARE_NO': [0, 0, 2, 2],
    'SHARE_FRIENDS': [1.5, 1, 0.5, 1],
    'SHARE_PUBLIC': [1.5, 2, 0, 0],
}
actions = pd.DataFrame(data=actions_dict, index=['pleasure', 'recognition', 'privacy', 'security'])
