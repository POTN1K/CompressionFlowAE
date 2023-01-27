"""
Quick tool to check which AutoEncoders are saved in StandardDims (and which are not)
"""

import os

lst = os.listdir('StandardDims')

dict_ = {}
for i in range(1, 10):
    for file in lst:
        if str(i) == file[-4] and not file[-5].isnumeric():
            if i not in dict_:
                dict_[i] = [file]
            else:
                dict_[i].append(file)

for i in range(10, 100):
    for file in lst:
        if str(i) == file[-5:-3]:
            if i not in dict_:
                dict_[i] = [file]
            else:
                dict_[i].append(file)

complete = []
for key, contents in dict_.items():
    if len(contents) == 3:
        complete.append(key)

need = []
for i in range(1, 65):
    if i not in complete:
        need.append(i)

print('Complete models:', complete)
print('Missing (dim<=64)', need)

