import pandas
from optimise_alignement import alignment
import matplotlib.pyplot as plt
from Predictor_model import predictions
from optimise_alignement import normalise

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np

adr = "../../Data/energy_data/sac_data/GP2R/"
ncsimul_file = "ACIER_S_1_ncsimul_F4042R.T22.025.Z03.10.csv"
real_file = "ACIER_S_1_real_F4042R.T22.025.Z03.10.csv"

normala = normalise()

def prep_dtw(y, y_, min, max, file_):
    print('y')
    print(y)
    print('y_')
    print(y_)
    print(type(y_))
    print(y_.shape)

    try:
        len(y) >= max and  len(y_) >= max
    except:
        raise NameError('the maximum lengh not respects lenght of inputs')
    else:
        path = dtw.warping_path(y[min:max], y_[min:max])
        distance, paths = dtw.warping_paths(y[min:max], y_[min:max])
        dtwvis.plot_warping(y[min:max], y_[min:max], path, filename= file_+"warp_results.png")

        best_path = dtw.best_path(paths)
        dtwvis.plot_warpingpaths(y[min:max], y_[min:max], paths, best_path, filename= file_+"best_path_results.png")

    return path, distance

def prepare_data_inputs(real_file, ncsimul_file):

    df_ncsimul = pandas.read_csv(filepath_or_buffer=adr + "NCSIMUL/" + ncsimul_file, sep=';')
    ncsimul_ = df_ncsimul.values

    df_real = pandas.read_csv(filepath_or_buffer=adr + "Real/" + real_file, sep=';')
    real_ = df_real.values
    return real_, ncsimul_

def compute_alignment_score(alignment_object, low_bound_real, low_bound_nc):

    real_, real_index_, ncsimul, ncsimul_index = alignment_object.prepare_remove(low_bound_real, low_bound_nc)
    real, real_index_after_smooth = alignment_object.prepare_smooth_x(real_, ncsimul)
    normal_real, normal_ncsimul = alignment_object.prepare_normal_y(real, ncsimul)

    plt.plot(normal_real, color='blue')
    plt.plot(normal_ncsimul, color='red')
    plt.show()

    pathi, distanci = alignment_object.prep_dtw(normal_real, normal_ncsimul, 0, len(normal_ncsimul), '')
    ncsimul_to_real = alignment_object.real_mapping_ncsimul_indexes(pathi, real_index_after_smooth)

    """draw graphs after alignment"""
    print(distanci)
    new_nc, new_real = alignment_object.re_assign_after_alignment(ncsimul_to_real, real_index_, ncsimul_index)
    alignment_object.joint_real_to_ncsimul('Acier', '1', new_nc, new_real)
    #alignment_object.replot_after_dtw(ncsimul_to_real, real_index_, ncsimul_index)

    return


real_data, ncsimul_data = prepare_data_inputs(real_file, ncsimul_file)
alg = alignment(real_data, ncsimul_data)#(real_data[:,0], ncsimul_data[:,56])
#alg.joint_real_to_ncsimul('Acier', 'sequence1')
#compute_alignment_score(alignment_object = alg, low_bound_real=300, low_bound_nc= 10)

"""learning on F S Ae Ap parameters"""
pred = predictions()
reg_ln = pred.train_COM()
#to_test_X = np.column_stack((ncsimul_data[:,41], ncsimul_data[:,31], ncsimul_data[:,10:12]))
#to_test_Y = ncsimul_data[:,56]

juncture_file = 'Acier_1.csv'
df_junture = pandas.read_csv(filepath_or_buffer=adr + "Juncture/" +juncture_file , sep=';')
juncture_data = df_junture.values
#
to_test_X = np.column_stack((juncture_data[:,41], juncture_data[:,31], juncture_data[:,10:12]))
to_test_Y = juncture_data[:,59]

"real powers"
# plt.plot(juncture_data[:,59], color = 'blue')
"ncsimul powers"
# plt.plot(juncture_data[:,56], color = 'red')
# plt.show()
#if we test the NCSIMUL set here
#to_test_Y = juncture_data[:,56]
#
y_pred = pred.predict_on_COM(x_test = to_test_X, y_test = to_test_Y)

plt.plot(normala.normalize_y(to_test_Y))
plt.plot(normala.normalize_y(y_pred))
plt.show()

prep_dtw(y= to_test_Y, y_= list(y_pred), min=0, max=len(y_pred), file_='LR_COM_REAL')

