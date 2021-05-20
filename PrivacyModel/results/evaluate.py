import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

plt.close("all")

# PLOTTING

# The averages of the results for each model
random_results1 = pd.read_csv("random_results/random_results1.csv")
random_results2 = pd.read_csv("random_results/random_results2.csv")
random_results3 = pd.read_csv("random_results/random_results3.csv")
random_results4 = pd.read_csv("random_results/random_results4.csv")
random_results5 = pd.read_csv("random_results/random_results5.csv")

random_results_total = pd.concat([random_results1, random_results2, random_results3, random_results4, random_results5])
random_results_mean = random_results_total.groupby(random_results_total.index).mean()

basic_results1 = pd.read_csv("basic_results/basic_results1.csv")
basic_results2 = pd.read_csv("basic_results/basic_results2.csv")
basic_results3 = pd.read_csv("basic_results/basic_results3.csv")
basic_results4 = pd.read_csv("basic_results/basic_results4.csv")
basic_results5 = pd.read_csv("basic_results/basic_results5.csv")

basic_results_total = pd.concat([basic_results1, basic_results2, basic_results3, basic_results4, basic_results5])
basic_results_mean = basic_results_total.groupby(basic_results_total.index).mean()

majority_results1 = pd.read_csv("majority_results/majority_results1.csv")
majority_results2 = pd.read_csv("majority_results/majority_results2.csv")
majority_results3 = pd.read_csv("majority_results/majority_results3.csv")
majority_results4 = pd.read_csv("majority_results/majority_results4.csv")
majority_results5 = pd.read_csv("majority_results/majority_results5.csv")

majority_results_total = pd.concat([majority_results1, majority_results2, majority_results3, majority_results4,
                                    majority_results5])
majority_results_mean = majority_results_total.groupby(majority_results_total.index).mean()

learning_results1 = pd.read_csv("learning_results/learning_results1.csv")
learning_results2 = pd.read_csv("learning_results/learning_results2.csv")
learning_results3 = pd.read_csv("learning_results/learning_results3.csv")
learning_results4 = pd.read_csv("learning_results/learning_results4.csv")
learning_results5 = pd.read_csv("learning_results/learning_results5.csv")

learning_results_total = pd.concat([learning_results1, learning_results2, learning_results3, learning_results4,
                                    learning_results5])
learning_results_mean = learning_results_total.groupby(learning_results_total.index).mean()

# plotting each model with each other for each metric
average_social_experience = pd.DataFrame(data={'Time Step': learning_results_mean["0"],
                                               'Random': random_results_mean["0.0"],
                                               'Selfish': basic_results_mean["0.0"],
                                               'Majority': majority_results_mean["0.0"],
                                               'SIPA': learning_results_mean["0.0"]},
                                         columns=['Time Step', 'Random', 'Selfish', 'Majority', 'SIPA'])

max_social_experience = pd.DataFrame(data={'Time Step': learning_results_mean["0"],
                                           'Random': random_results_mean["0.0.1"],
                                           'Selfish': basic_results_mean["0.0.1"],
                                           'Majority': majority_results_mean["0.0.1"],
                                           'SIPA': learning_results_mean["0.0.1"]},
                                     columns=['Time Step', 'Random', 'Selfish', 'Majority', 'SIPA'])

min_social_experience = pd.DataFrame(data={'Time Step': learning_results_mean["0"],
                                           'Random': random_results_mean["0.0.2"],
                                           'Selfish': basic_results_mean["0.0.2"],
                                           'Majority': majority_results_mean["0.0.2"],
                                           'SIPA': learning_results_mean["0.0.2"]},
                                     columns=['Time Step', 'Random', 'Selfish', 'Majority', 'SIPA'])

average_reward = pd.DataFrame(data={'Time Step': learning_results_mean["0"],
                                    'Random': random_results_mean["0.0.3"],
                                    'Selfish': basic_results_mean["0.0.3"],
                                    'Majority': majority_results_mean["0.0.3"],
                                    'SIPA': learning_results_mean["0.0.3"]},
                              columns=['Time Step', 'Random', 'Selfish', 'Majority', 'SIPA'])

below_average = pd.DataFrame(data={'Time Step': learning_results_mean["0"],
                                   'Random': random_results_mean["0.0.4"],
                                   'Selfish': basic_results_mean["0.0.4"],
                                   'Majority': majority_results_mean["0.0.4"],
                                   'SIPA': learning_results_mean["0.0.4"]},
                             columns=['Time Step', 'Random', 'Selfish', 'Majority', 'SIPA'])

# filter after 50 time steps
drop_list = []
for i in range(51):
    drop_list.append(i)

random_results_mean = random_results_mean.drop(drop_list)
basic_results_mean = basic_results_mean.drop(drop_list)
majority_results_mean = majority_results_mean.drop(drop_list)
learning_results_mean = learning_results_mean.drop(drop_list)


test1 = ttest_rel(learning_results_mean["0.0"], majority_results_mean["0.0"], alternative='greater')
test2 = ttest_rel(learning_results_mean["0.0.1"], majority_results_mean["0.0.1"], alternative='greater')
test3 = ttest_rel(learning_results_mean["0.0.2"], majority_results_mean["0.0.2"], alternative='greater')
test4 = ttest_rel(learning_results_mean["0.0.3"], majority_results_mean["0.0.3"], alternative='greater')
test5 = ttest_rel(learning_results_mean["0.0.4"], majority_results_mean["0.0.4"], alternative='less')

# print(test1)
# print(test2)
# print(test3)
# print(test4)
# print(test5)

average_social_experience.plot(x="Time Step", ylabel="Social Experience");
max_social_experience.plot(x="Time Step", ylabel="Social Experience");
min_social_experience.plot(x="Time Step", ylabel="Social Experience");
average_reward.plot(x="Time Step", ylabel="Reward (Sanctions) Received");
below_average.plot(x="Time Step", ylabel="Number of Agents with Social Experience Lower Than Average")

plt.show()
