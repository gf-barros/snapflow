""" Auxiliary functions for the Reduced Order Model of the LOCK """

# Loading high fidelity data:
def load_high_fidelity_data_fct(mode:str,
                                data:str,
                                format:str,
                                colab_dir:str,
                                ):
  

  ###############################################################################
  # Defines which data we import (reduced version - first 10% of time data for each parameter)
  # or the full snapshot matrices.
  if data == "reduced":
    # Load reduced (for prototyping code) snapshots - Jump this part if you have
    #  already the stacked version of it saved ahead.
    list_train_files = ['snapshots_theta_0_60.pkl', 'snapshots_theta_2_60.pkl', 'snapshots_theta_4_60.pkl',
                'snapshots_theta_6_60.pkl', 'snapshots_theta_8_60.pkl', 'snapshots_theta_10_60.pkl']
    val_file = 'snapshots_theta_5_60.pkl' # in future, make as a list.

  elif data == "full":
    list_train_files = ['snapshots_theta_0_5_to_895.pkl', 'snapshots_theta_2_5_to_895.pkl', 'snapshots_theta_4_5_to_895.pkl',
                'snapshots_theta_6_5_to_895.pkl', 'snapshots_theta_8_5_to_895.pkl', 'snapshots_theta_10_5_to_895.pkl']
    val_file = 'snapshots_theta_5_5_to_895.pkl' # in future, make as a list.

  else:
    print("Error: you need to choose the type_data as 'reduced' or 'full'.\n")
  ################################################################################

  #################################
  if data == "reduced":
        f_name = 'S_red_60.pkl'
  elif data == "full":
        f_name = 'S_600.pkl'



  # colab version:
  if mode == "colab":
    # mount colab drive
    #from google.colab import drive
    #drive.mount('/content/drive')

    # Load functions from mor_fcts.py file
    #import sys
    #sys.path.append(colab_dir)
    #from mor_fcts import *

    directory_path = '/content/drive/MyDrive/turbi/snapshots/'

    if format == "each":
      load_fct = lambda x: joblib.load(directory_path + x)

      # Produces a list whose each element is the snapshot for a given parameter for
      #  all times (no split yet).
      list_snapshots_per_parameter = list(map(load_fct, list_train_files))

      # Assembling a stack of full order solutions for the values of parameters:
      S_train = np.hstack((list_snapshots_per_parameter)) # np.hstack is the same as np.concatenate with axis=1
      print("The shape of S_train is: ", S_train.shape)
      joblib.dump(S_train, directory_path + f_name)

      S_val = joblib.load(directory_path + val_file)
      print("The shape of S_val is: ", S_val.shape)

    elif format == "stacked":

      # if type_data == "reduced":
      #   f_name = 'S_red_60.pkl'
      # elif type_data == "full":
      #   f_name = 'S_600.pkl'

      S_train = joblib.load(directory_path + f_name)
      print("The shape of S_train is: ", S_train.shape)

      S_val = joblib.load(directory_path + val_file)
      print("The shape of S_val is: ", S_val.shape)


  # local version (running on your own notebook):
  elif mode == "local":

    # Load functions from mor_fcts.py file
    #from mor_fcts import *

    directory_path = '~/snapshots/'

    if format == "each":
      load_fct = lambda x: joblib.load(directory_path + x)

      # Produces a list whose each element is the snapshot for a given parameter for
      #  all times (no split yet).
      list_snapshots_per_parameter = list(map(load_fct, list_train_files))

      # Assembling a stack of full order solutions for the values of parameters:
      S_train = np.hstack((list_snapshots_per_parameter)) # np.hstack is the same as np.concatenate with axis=1
      print("The shape of S_train is: ", S_train.shape)
      joblib.dump(S_train, directory_path + f_name)

      S_val = joblib.load(directory_path + val_file)
      print("The shape of S_val is: ", S_val.shape)

    elif format == "stacked":

      # if type_data == "reduced":
      #   f_name = 'S_red_60.pkl'
      # elif type_data == "full":
      #   f_name = 'S_600.pkl'

      S_train = joblib.load(directory_path + f_name)
      print("The shape of S_train is: ", S_train.shape)

      S_val = joblib.load(directory_path + val_file)
      print("The shape of S_val is: ", S_val.shape)

  else:
    print("You need to choose the 'run_mode' as 'colab' or 'local'.\n")

  return S_train, S_val


# SVD POD needs:
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Autoencoder needs:
import time
#import torch
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
#import matplotlib.pyplot as plt # already imported
#import numpy as np # already imported


# Auxiliary functions:

def svd_sklearn(mat, n_comp:int):
    X = csr_matrix(mat)
    # Instantiate svd from sklearn:
    svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
    # Performs svd itself
    svd.fit(X)
    return svd

# print("Explained Variance Ratio (EVR)", svd2.explained_variance_ratio_)
# print("Cum_sum EVR", svd2.explained_variance_ratio_.cumsum())
# print("Sum EVR", svd2.explained_variance_ratio_.sum())
# print("Singular Values", svd2.singular_values_)

def explain_var(svd, print_EV:bool=True, plot:bool=False):
    if print_EV:
        print("Explained Variance Ratio (EVR)", svd.explained_variance_ratio_)
        print("Cum_sum EVR", svd.explained_variance_ratio_.cumsum())
        print("Sum EVR", svd.explained_variance_ratio_.sum())
        print("Singular Values", svd.singular_values_)

    if plot:
        plt.plot([x+1 for x in range(svd.n_components)], svd.explained_variance_ratio_.cumsum())
        plt.show()
        
    return svd.explained_variance_ratio_.cumsum()

def plot_svd_dim(snapshot_matrix, svd_dim:int, printEV:bool=False):
    print("Max. number of components is ", snapshot_matrix.shape[1])
    svd_matrix = svd_sklearn(snapshot_matrix, svd_dim)
    explain_var(svd = svd_matrix, print_EV = printEV, plot = True)

# SVD numpy - faster than SVD scikit learn
def svd(matrix):
    u_vectors, s_values, vt_vectors = np.linalg.svd(matrix,
                                                full_matrices=False,
                                                compute_uv=True,
                                                hermitian=False,
                                               )
    return u_vectors, s_values, vt_vectors

# Finds number of SVD modes so that "eps" error is achieved.
def find_N(vec_eig, eps:float):
  sigma = np.diag(vec_eig)
  r = np.linalg.matrix_rank(sigma)

  v_sq_cum_sum = np.cumsum(vec_eig**2)
  a = v_sq_cum_sum * eps

  vec_flip = np.flip(vec_eig)
  b = np.cumsum(vec_flip**2)

  for i in range(1,r):
    #print(i)
    if a[[-1]] >= b[[r-1-i]]:
      return(i)
      #print(i)
      #Nv1 = i
    else:
      i +=1

  return(None)


  # N = 1
  # while N < r:
  #   #print('N is: ',N)
  #   #print('b test is: ', b[[r-1-N]])
  #   if a[[-1]] >= b[[r-1-N]]:
  #     print(N)
  #     return N
  #     break
  #   N += 1
  
  # aux functions for Autoencoder:
  
def error_fct(original_data, rec_data):
  fct3 = lambda x: x.detach().numpy()

  # Adjusting format/type of rec_data to numpy format with same dimensions as original_data:
  r1 = list(map(fct3, rec_data))
  rec_data_stacked = np.stack(r1, axis=1)
  rec_data = rec_data_stacked[0]
  
  dif = rec_data - original_data
  
  #erro_rel = erro_dif / original_data
  #erro_rel_perc = erro_rel * 100
  #erro_rel_perc

  #print( "Maximum relative percentual point error is: ", round(np.max(erro_rel_perc), 3), "%" )
  #print( "Minimum relative percentual point error is: ", round(np.min(erro_rel_perc), 3), "%" )
  #print( "Average absolute relative percentual point error is: ", round(np.mean(np.abs(erro_rel_perc)), 3), "%" )
  vector_norm_L1 = np.linalg.norm(original_data, ord=1)
  error_L1 = ( np.linalg.norm(dif, ord=1) ) / vector_norm_L1
  print("Vector error L1 is: ", error_L1)

  vector_norm_L2 = np.linalg.norm(original_data, ord=2)
  error_L2 = ( np.linalg.norm(dif, ord=2) ) / vector_norm_L2
  print("Vector error L2 is: ", error_L2)

  vector_norm_Linf = np.linalg.norm(original_data, np.inf)
  error_Linf = ( np.linalg.norm(dif, np.inf) ) / vector_norm_Linf
  print("Vector error L_infinity is: ", error_Linf)
  return error_L1, error_L2, error_Linf

