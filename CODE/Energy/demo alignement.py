# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:51:29 2019

@author: d596324
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:00:58 2019

@author: d596324
"""

from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def getKey(item):
    return item[1]


def moving_average(x,size=200):
    averages = []
    for i in range(len(x)):
        averages.append(np.mean(x[i:i+size]))
    return averages

outil = 'ToolRef'
  
#change this for the path of file location:
os.chdir('C:\\Users\\D596324\\Desktop\\Lucid\\Essais Mars 2019 UF1\\')

excel_sheet = []
f = open('Projet-Lucid-ACIER_stats.csv','r')
for ln in f:
    excel_sheet.append(ln.split(';'))

excel_sheet = np.array(excel_sheet)

df_excel_sheet = pd.DataFrame(data = excel_sheet[1:], columns = excel_sheet[0])

Pc1_values = [ float(u) for u in df_excel_sheet['Pc1'].values ]
Pc2_values = [ float(u) for u in df_excel_sheet['Pc2'].values ]


### groupby ToolRef
df_excel_grouped_by_outil = df_excel_sheet.groupby(outil)
        
dict_tools = {}
for tool in df_excel_grouped_by_outil.size().keys():
    df_lucid_temp = df_excel_sheet[df_excel_sheet[outil]==tool]
    dict_tools[tool] = df_lucid_temp# [ float(u) for u in df_lucid_temp['Pc1'].values] 

plt.figure(-5)
for k in dict_tools:
    Pc1_values_temp = [ float(u) for u in dict_tools[k]['Pc1'].values] 
    plt.plot( dict_tools[k].index.values, Pc1_values_temp,  label = k  )
plt.legend()


#Open a file containing a curve, for example file 03150911.27F

os.chdir('courbes')
data_compressed = []
f = open('03150911.27F','r') #  03141439.47F
for ln in f:
    data_compressed.append([int(u) for u in ln.split('\t')])
f.close()
data_compressed = np.array(data_compressed)


##### We have the correspondance of tool F4042R.T22.025.Z03.10 starting from entry 4000 with curve 03150911.27F
##### '03150901.13F' - 'F4042R.T22.025.Z03.10' 400:3200
##### '03150911.27F' - 'F4042R.T22.025.Z03.10' 4000:

pc_outil_1 = [ float(u) for u in  dict_tools['F4042R.T22.025.Z03.10']['Pc1'].values]
x = pc_outil_1.copy()[4000:]
real_x = data_compressed[:,0] # first column contains Pc1


plt.figure(-1)
indices_zero = [i for i,u in enumerate(x) if u == 0]
x_segments_non_zero = [ (u,v)  for u,v in zip(indices_zero[:-1] , indices_zero[1:]  ) if v-u > 1 ]
plt.plot(x)

# Split the simulation in segments corresponding to entrees/sorties:
for i,(u,v) in enumerate(x_segments_non_zero):
    plt.figure(i)
    plt.plot(range(u,v), x[u:v],c='red')
    
# Show the real curve:
plt.figure(-2)
plt.plot(real_x)


### Use KDE to approximate the density of Pc1 (its moving average, to be more precisely)
real_x_av = np.array( moving_average(real_x, 2000) )
kde_skl = KernelDensity(bandwidth=3)
kde_skl.fit(real_x_av.reshape(-1,1))
X_plot = np.linspace( min(real_x_av),max(real_x_av), 1000 ).reshape(-1,1)
samples = np.exp(kde_skl.score_samples(X_plot))
samples_ordered = sorted( [ (u,v) for u,v in enumerate(samples[:int(np.floor(max(real_x_av)/10))]) ], key=getKey)

# the first peak in the density is noise, the second peak corresponds to the power when turning outside the material
# (P_vide). We are therefore interested in finding the location of this latter peak, so to know when we are inside
# or outside the material

plt.figure(-3)
plt.hist(real_x_av, range(int(np.floor(min(real_x_av))),int(np.ceil(max(real_x_av)))),normed=True)
plt.plot(X_plot[:,0],samples)
threshold = 0.7*1e-3
sample_indices_min = samples_ordered[0][0]

for v in range(sample_indices_min, len(samples)):
    if samples[v] > threshold:
        start = v
        break

current_max = start
for v in range(start+1, len(samples)):
    if samples[v] > samples[current_max]:
        current_max = v
    else:
        break
    
stop = start +  (current_max - start)

plt.scatter( X_plot[start:stop], samples[start:stop], c='red'  )


# detect and plot the locations where the tool is outside the material:
plt.figure(-4)
plt.plot(real_x)
zero_values = ( X_plot[start,0], X_plot[stop,0])
indices_returns_to_zero = [ u for u in range(len(real_x)) if    real_x[u] < zero_values[1] ]

min_segment_length = 0
plt.scatter(indices_returns_to_zero, real_x[indices_returns_to_zero], c='red' )
real_x_segments_non_zero = []
for i in range(1,len(indices_returns_to_zero)):
    len_segm = indices_returns_to_zero[i] - indices_returns_to_zero[i-1]
    if len_segm > min_segment_length:
        real_x_segments_non_zero.append( (indices_returns_to_zero[i-1], indices_returns_to_zero[i] ) )



real_x_segments_non_zero = set(real_x_segments_non_zero)
real_x_segments_non_zero = sorted(list(real_x_segments_non_zero))

nr_segments = min( len(x_segments_non_zero), len(real_x_segments_non_zero ))

lens_real_x_segments_non_zero = [ (i, v-u) for i,(u,v) in enumerate(real_x_segments_non_zero) ]
lens_real_x_segments_non_zero = sorted(lens_real_x_segments_non_zero, key=getKey,reverse=True)
segs_to_keep = sorted(lens_real_x_segments_non_zero[:nr_segments])
real_x_segments_non_zero_to_keep  = [ real_x_segments_non_zero[i] for i,l in segs_to_keep ]


lens_x_segments_non_zero  = [ (i, v-u) for i,(u,v) in enumerate(x_segments_non_zero) ]
lens_x_segments_non_zero = sorted(lens_x_segments_non_zero, key=getKey,reverse=True)
segs_to_keep = sorted(lens_x_segments_non_zero[:nr_segments])
x_segments_non_zero_to_keep  = [ x_segments_non_zero[i] for i,l in segs_to_keep ]

for i, ((u,v),(w,z)) in enumerate(zip( real_x_segments_non_zero_to_keep , x_segments_non_zero_to_keep )):
    plt.figure(i)
    plt.subplot(2,1,1)
    plt.plot(range(u,v), real_x[u:v])
    plt.subplot(2,1,2)
    plt.plot(range(w,z), x[w:z])

plt.figure(-10)
plt.plot( real_x )
for (w,z) in real_x_segments_non_zero_to_keep:
    plt.plot(range(w,z), real_x[w:z],c='red')

plt.figure(-10)
plt.plot( x )
for i,(w,z) in enumerate( x_segments_non_zero_to_keep):
    plt.figure(i)
    plt.plot(range(w,z), x[w:z],c='red')
    


# now we have found a correspondance between the real segments and the simulated segments, we need to rescale both of them:
scaler = MinMaxScaler()
#x = Pc1_values[4188:4273]
j=2
x_seg =  x[x_segments_non_zero[j][0]:x_segments_non_zero[j][1]]
x_seg = scaler.fit_transform(np.array(x_seg).reshape(-1,1))
plt.figure(100)
plt.plot(x_seg,c='orange')

y= real_x[ real_x_segments_non_zero_to_keep[j][0] :  real_x_segments_non_zero_to_keep[j][1]]
y_mov = moving_average(y,1000) 
y_mov = scaler.fit_transform(np.array(y_mov).reshape(-1,1))
plt.figure(102)
plt.plot(y_mov)



# we then use DTW to align the two curves:

euclidean_norm = lambda x1, y1: np.abs(x1 - y1)

dist, cost_matrix, acc_cost_matrix, path = dtw(x_seg, y_mov , dist=euclidean_norm)

plt.figure(144)
x_mapped = [ x_seg[u] for u in path[0] ]
y = scaler.transform( np.array(y).reshape(-1,1) )
y_mapped = [y[u] for u in path[1] ] # # [y_mov[u] for u in path[1] ] #
plt.plot(y_mapped[500:])
plt.plot(x_mapped)

# plot the path:
plt.figure(145)
plt.scatter(path[0],path[1],s=3)

'''
dispersion = []
for i in range(len(x_mapped) - 1):
    if x_mapped[i+1] == x_mapped[i]:
        dispersion.append(0)
    else:
        dispersion.append(1)
        
        
av_dispersion = moving_average(dispersion)
threshold = np.mean(av_dispersion) + np.std(av_dispersion)
break_points = [ i for i,u in enumerate(av_dispersion) if u > threshold ]
plt.scatter(break_points, [x_mapped[u] for u in break_points],s=5,c='red')
plt.scatter(range(len(dispersion)),dispersion,s=2,c='red')
'''

scaler = MinMaxScaler()
#x = Pc1_values[4188:4273]
x_seg =  x[x_segments_non_zero[-1][0]:x_segments_non_zero[-1][1]]
x_seg = scaler.fit_transform(np.array(x_seg).reshape(-1,1)[60:])
plt.figure(100)
plt.plot(x_seg)

y= real_x[ real_x_segments_non_zero_to_keep[-1][0] :  real_x_segments_non_zero_to_keep[-1][1]  ]
y_mov = moving_average(y,1000)[7500:20000]
y_mov = scaler.fit_transform(np.array(y_mov).reshape(-1,1))
plt.figure(102)
plt.plot(y_mov)


euclidean_norm = lambda x1, y1: np.abs(x1 - y1)

dist, cost_matrix, acc_cost_matrix, path = dtw(x_seg, y_mov , dist=euclidean_norm)

plt.figure(144)
x_mapped = [ x_seg[u] for u in path[0] ]
y_mapped = [y_mov[u] for u in path[1] ]
plt.plot(y_mapped)
plt.plot(x_mapped)




# overview:
for i, f_name in enumerate(os.listdir()):
    data_compressed = []
    f = open(f_name,'r') #  03141439.47F
    for ln in f:
        data_compressed.append([int(u) for u in ln.split('\t')])
    f.close()
    data_compressed = np.array(data_compressed)
    real_x = data_compressed[:,0]
    kde_skl = KernelDensity(bandwidth=3)
    kde_skl.fit(real_x.reshape(-1,1))
    X_plot = np.linspace( min(real_x),max(real_x), 1000 ).reshape(-1,1)
    samples = np.exp(kde_skl.score_samples(X_plot))
    samples_ordered = sorted( [ (u,v) for u,v in enumerate(samples[:int(np.floor(max(real_x)/10))]) ], key=getKey)
    plt.figure(-1*i)
    plt.hist(real_x, range(int(np.floor(min(real_x))),int(np.ceil(max(real_x)))),normed=True)
    plt.plot(X_plot[:,0],samples)
    plt.figure(i)
    plt.plot(real_x)