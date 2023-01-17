import matplotlib.pyplot as plt
import numpy as np

from FlowGeneration import *


mode1, mode2, mode3, mode4 = np.transpose(generation_from_original(u_all))

center = (stats(mode1)[0], stats(mode2)[0], stats(mode3)[0])

# Taking into account all the points present
distance = np.sqrt((mode1 - center[0])**2 + (mode2 - center[1])**2 + (mode3 - center[2])**2)
mean_distance = np.mean(distance)
std_distance = np.std(distance)

# Eliminate the outliers
percentage = 0.4
distance2 = distance[distance > percentage]
mean_distance2 = np.mean(distance2)
std_distance2 = np.std(distance2)
# Printing and Plotting

#plt.subplot(221)
plt.plot(np.arange(0, np.shape(distance)[-1]), distance[0,0,:], '*')
plt.ylim(0, 0.7)
# plt.subplot(222)
# plt.plot(np.arange(0, len(distance2)), distance2, '*')
# plt.ylim(0, 0.7)
# plt.subplot(223)
# plt.boxplot(distance[0,0,:])
# plt.ylim(0, 0.7)
# plt.subplot(224)
# plt.boxplot(distance2)
# plt.ylim(0, 0.7)
plt.show()


print(f'Points included in the cut: {np.shape(distance2)}, ({np.shape(distance2)[0]/np.shape(distance)[-1] * 100}%)')
print(f'Physical range: [{percentage} - {round(np.max(distance2),3)}]')
print(f'All points considered: mean = {mean_distance}, std = {std_distance}')
print(f'Cut dataset: mean = {mean_distance2}, std = {std_distance2}')

# distance > 0.4 --> 151/200 points are present 75.5%
# distance > 0.35 --> 162/200 points are present 81%
# distance > 0.3 --> 177/200 points are present 88.5%

