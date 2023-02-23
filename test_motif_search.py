import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt

plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')

df = pd.read_csv("https://zenodo.org/record/4328047/files/toy.csv?download=1")
df.head()

ig, axs = plt.subplots(df.shape[1], sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Can You Spot The Multi-dimensional Motif?', fontsize='30')

for i in range(df.shape[1]):
    axs[i].set_ylabel(f'T{i + 1}', fontsize='20')
    axs[i].set_xlabel('Time', fontsize ='20')
    axs[i].plot(df[f'T{i + 1}'])

# plt.show()
plt.close()

m = 30
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

plt.show()


# mps, indices = stumpy.mstump(df, m)
# motifs_idx = np.argmin(mps, axis=1)
# nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]
# fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})

# for k, dim_name in enumerate(df.columns):
#     axs[k].set_ylabel(dim_name, fontsize='20')
#     axs[k].plot(df[dim_name])
#     axs[k].set_xlabel('Time', fontsize ='20')

#     axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
#     axs[k + mps.shape[0]].plot(mps[k], c='orange')
#     axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')

#     axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#     axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
#     axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#     axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

#     if dim_name != 'T3':
#         axs[k].plot(range(motifs_idx[k], motifs_idx[k] + m), df[dim_name].iloc[motifs_idx[k] : motifs_idx[k] + m], c='red', linewidth=4)
#         axs[k].plot(range(nn_idx[k], nn_idx[k] + m), df[dim_name].iloc[nn_idx[k] : nn_idx[k] + m], c='red', linewidth=4)
#         axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
#         axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')
#     else:
#         axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='black')
#         axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='black')

# plt.show()


def generated_inf_circle_trajectory():
    a = 1
    phi = np.arange(0, 10*np.pi, 0.1)
    x = a*phi*np.cos(phi)
    y = a*phi*np.sin(phi)

    dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
    r = np.zeros_like(x)
    r[1:] = np.cumsum(dr) # integrate path
    r_int = np.linspace(0, r.max(), 200) # regular spaced path
    x_int = np.interp(r_int, r, x) # interpolate
    y_int = np.interp(r_int, r, y)

    print(x_int)
    print(y_int)


    def func(phi):
        return np.concatenate((a*phi*np.cos(phi), a*phi*np.sin(phi)),axis=1)

    phiaxis = np.linspace(0, r.max(), 20000)
    result = func(phiaxis[:,None])

    print(result)
    plot_multi_dimensions_graph(result, 'circle_trajectory.png')
    exit()

    plt.subplot(1,2,1)
    plt.plot(x, y, 'o-')
    plt.title('Original')
    plt.axis([-32,32,-32,32])

    plt.subplot(1,2,2)
    plt.plot(x_int, y_int, 'o-')
    plt.title('Interpolated')
    plt.axis([-32,32,-32,32])
    plt.savefig('circle_trajectory.png')

    