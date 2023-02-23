import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt

plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')

df = pd.read_csv("https://zenodo.org/record/4328047/files/toy.csv?download=1")
print(df.head())
print(df)

df = df[:100]


def generated_triangular_trajectory():

    # triangular

    flag_value = 0
    side_length = 2
    generated_traj = [[0,0,flag_value],
                        [0,side_length,flag_value],
                        [side_length,side_length,flag_value],
                        [0,0,flag_value]]

    temp_one = []

    num_of_points = 50
    for i in range(num_of_points):
        temp_one.append([0,i*side_length/num_of_points,flag_value])
    for i in range(num_of_points):
        temp_one.append([i*side_length/num_of_points,side_length,flag_value])
    for i in range(num_of_points):
        temp_one.append([side_length - i*side_length/num_of_points,side_length - i*side_length/num_of_points,flag_value])

    return temp_one


def generated_inf_circle_trajectory():

    phi = np.arange(0, 10*np.pi, 0.1)

    x = phi*np.cos(phi)
    x = x.reshape((len(x),-1))
    y = phi*np.sin(phi)
    y = y.reshape((len(y),-1))
    z = np.zeros((len(y),1))

    circle = np.concatenate((x,y,z),axis=1)

    plt.axis([-32,32,-32,32])

    return circle

def get_trajectory(spec):

    return generated_inf_circle_trajectory()
    # return generated_triangular_trajectory()

def plot_multi_dimensions_graph(ideal_trajectory, directory_to_save):

    plt.plot(np.asarray(ideal_trajectory)[:,0],
        np.asarray(ideal_trajectory)[:,1],'o-r',label='ideal trajectory')
    
    plt.legend(fontsize=12)
    plt.savefig(directory_to_save)
    plt.close()

input_traj = np.array(get_trajectory(None),dtype=float)
plot_multi_dimensions_graph(input_traj, "circle_trajectory.png")

df = pd.DataFrame(input_traj, columns = ['T1','T2','T3'])
print(df)

# exit()

ig, axs = plt.subplots(df.shape[1], sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Can You Spot The Multi-dimensional Motif?', fontsize='30')

for i in range(df.shape[1]):
    axs[i].set_ylabel(f'T{i + 1}', fontsize='20')
    axs[i].set_xlabel('Time', fontsize ='20')
    axs[i].plot(df[f'T{i + 1}'])

# plt.show()
plt.close()

m = 30
m = 50
mps = {}  # Store the 1-dimensional matrix profiles
motifs_idx = {}  # Store the index locations for each pair of 1-dimensional motifs (i.e., the index location of two smallest matrix profile values within each dimension)
for dim_name in df.columns:
    mps[dim_name] = stumpy.stump(df[dim_name], m)
    motif_distance = np.round(mps[dim_name][:, 0].min(), 1)
    print(f"The motif pair matrix profile value in {dim_name} is {motif_distance}")
    motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]

fig, axs = plt.subplots(len(mps), sharex=True, gridspec_kw={'hspace': 0})

for i, dim_name in enumerate(list(mps.keys())):
    axs[i].set_ylabel(dim_name, fontsize='20')
    axs[i].plot(df[dim_name])
    axs[i].set_xlabel('Time', fontsize ='20')
    for idx in motifs_idx[dim_name]:
        axs[i].plot(df[dim_name].iloc[idx:idx+m], c='red', linewidth=4)
        axs[i].axvline(x=idx, linestyle="dashed", c='black')

plt.savefig('motif.png')
plt.close()

for idx in motifs_idx['T1']:
    plt.plot(df['T1'].iloc[idx:idx+m],df['T2'].iloc[idx:idx+m], c='red', linewidth=4)
for idx in motifs_idx['T2']:
    plt.plot(df['T1'].iloc[idx:idx+m],df['T2'].iloc[idx:idx+m], c='red', linewidth=4)
plt.show()



