import pickle as pk
import matplotlib.pyplot as plt
import numpy as np

f = open('saved_dictionary.pkl','rb')

# f = open('saved_dictionary_mlp.pkl','rb')

data = pk.load(f)

print(data)
imp = {}

for key in data[0].keys():
    imp[key] = np.array(data[0][key]['importance'])
    imp[key] /= np.max(imp[key])

N = 10
label_x = ['PITCH','SPEED','STW','WIND_SPEED','WIND_ANGLE','SOG','SOGmSTW','HEADING','DISP','TORQUE']

width = 0.35

fig = plt.figure()
color = ['b','g','k','c','m','y','r','#008000','#008080','#808080']
sum = np.zeros(10)
for i, key in enumerate(imp):
    if not i:
        plt.bar(label_x, imp[key], color=color[i], label =key.upper())
    else:
        plt.bar(label_x, imp[key], color=color[i], bottom=sum, label =key.upper())
    sum += imp[key]


plt.xticks(rotation = -45)
plt.ylabel('Weight importance', fontweight ='bold', fontsize = 12)
plt.xlabel('Features', fontweight ='bold', fontsize = 12)
plt.legend(loc=3)
plt.show()