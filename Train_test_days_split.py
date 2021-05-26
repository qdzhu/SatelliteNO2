import numpy as np
import os
import datetime
import pandas as pd
import netCDF4 as nc
from glob import glob
import dask.array as da
from dask.distributed import Client
import geopy.distance
import csv
import scipy.io

orig_file_path = '/global/home/users/qindan_zhu/myscratch/jlgrant/ML-WRF/ML-WRF'
#orig_file_path = '/Volumes/share-wrf3/ML-WRF/'

def train_test_filename(orig_filenames):
    """
    Read in available filenames, shuffle them and assign files to train and test collections
    :return: filename list in train dataset and filename list in test dataset
    """
    np.random.shuffle(orig_filenames)
    train_size = int(len(orig_filenames))
    train_filenames = orig_filenames[:train_size]
    test_filenames = orig_filenames[train_size:]
    return train_filenames, test_filenames


def read_orig_file_from_wrf(filename):
    keep_indx = np.load('target_cells_index.npy')
    reader = csv.reader(open("surrouding_cells.csv", "r"))
    surr_arr = []
    for row in reader:
        this_arr = list(map(int, row[1].strip(']|[').split(',')))
        surr_arr.append(this_arr)
    surr_arr = np.array(surr_arr)[keep_indx, :]
    data = nc.Dataset(filename)
    labels = []

    for variable in data.variables:
        #print(variable + ":" + str(data[variable].shape))
        labels.append(variable)
    for i in range(0, 24):
        labels.append('anthro_surr_emis_{:02d}'.format(i))
        labels.append('lightning_surr_emis_{:02d}'.format(i))
    labels.append('total_lightning')
    data_dict = {label: [] for label in labels}
    extra_vars = ['xlon', 'xlat', 'hour', 'date', 'IC_FLASHCOUNT', 'CG_FLASHCOUNT', 'E_NO', 'U', 'V']
    #print('Variables require extra processing steps: {}'.format(extra_vars[:]))
    dims = data['no2'].shape
    ntime = dims[0]-1
    ngrid = len(keep_indx)
    nvel = dims[2]
    data_hours = da.array(data['hour'][1:], dtype='float32')
    data_dict['hour'] = da.repeat(da.repeat(data_hours[:, :, np.newaxis], ngrid, axis=1), nvel, axis=2)
    xlon = da.array(data['xlon'][:], dtype='float32').flatten()[np.newaxis, keep_indx, np.newaxis]
    data_dict['xlon'] = da.repeat(da.repeat(xlon, ntime, axis=0), nvel, axis=2)
    xlat = da.array(data['xlat'][:], dtype='float32').flatten()[np.newaxis, keep_indx, np.newaxis]
    data_dict['xlat'] = da.repeat(da.repeat(xlat, ntime, axis=0), nvel, axis=2)
    data_dict['date'] = da.zeros((ntime, ngrid, nvel)) + da.mean(data['date'][:], dtype='float32')
    data_dict['date'] = data_dict['date']
    cum_ic_flash = da.array(data['IC_FLASHCOUNT'][:], dtype='float32')
    cum_cg_flash = da.array(data['CG_FLASHCOUNT'][:], dtype='float32')
    ic_flash = da.repeat(cum_ic_flash[1:, :, :]-cum_ic_flash[:-1, :, :], nvel, axis=2)
    cg_flash = da.repeat(cum_cg_flash[1:, :, :]-cum_cg_flash[:-1, :, :], nvel, axis=2)
    e_lightning = ic_flash + cg_flash
    data_dict['IC_FLASHCOUNT'] = ic_flash[:, keep_indx, :]
    data_dict['CG_FLASHCOUNT'] = cg_flash[:, keep_indx, :]
    data_dict['total_lightning'] = e_lightning[:, keep_indx, :]
    e_no_lower = da.array(data['E_NO'], dtype='float32')[1:, :, :]
    e_no_upper = da.zeros((ntime, e_no_lower.shape[1], nvel - e_no_lower.shape[2]), dtype='float32')
    e_no = da.concatenate([e_no_lower, e_no_upper], axis=2)
    data_dict['E_NO'] = e_no[:, keep_indx, :]
    for i in range(0, 24):
        this_label = 'anthro_surr_emis_{:02d}'.format(i)
        surr_indx = surr_arr[:, i]
        data_dict[this_label] = e_no[:, surr_indx, :]
        this_label = 'lightning_surr_emis_{:02d}'.format(i)
        data_dict[this_label] = e_lightning[:, surr_indx, :]
    stg_u = da.array(data['U'], dtype='float32')
    stg_v = da.array(data['V'], dtype='float32')
    u_indx_left, u_indx_right, v_indx_bot, v_indx_up = find_indx_for_wind()
    wind_u = (stg_u[1:, u_indx_left, :] + stg_u[1:, u_indx_right, :])/2
    data_dict['U'] = wind_u[:, keep_indx, :]
    wind_v = (stg_v[1:, v_indx_up, :] + stg_v[1:, v_indx_bot, :])/2
    data_dict['V'] = wind_v[:, keep_indx, :]

    match_vars = ['no2', 'pres', 'temp', 'CLDFRA']
    #print('Variables read directly from wrf: {}'.format(match_vars[:]))
    for var in match_vars:
        data_dict[var] = da.array(data[var], dtype='float32')[1:, keep_indx, :]

    reduce_dim_vars = ['elev', 'W']
    #print('Variables average vertically: {}'.format(reduce_dim_vars[:]))
    for var in reduce_dim_vars:
        this_value = da.array(data[var], dtype='float32')[1:, keep_indx, :]
        data_dict[var] = (this_value[:, :, 1:] + this_value[:, :, :-1])/2

    add_dim_vars = ['COSZEN', 'PBLH', 'LAI', 'HGT', 'SWDOWN', 'GLW']
    #print('Variables add vertical layers: {}'.format(add_dim_vars[:]))

    for var in add_dim_vars:
        this_value = da.array(data[var], dtype='float32')[1:, keep_indx, :]
        data_dict[var] = da.repeat(this_value, nvel, axis=2)

    #print('Key of dict:{}'.format(data_dict.keys()))
    additional_features = ['xlon', 'xlat', 'date', 'elev', 'hour', 'IC_FLASHCOUNT', 'CG_FLASHCOUNT']
    y_label = ['no2']
    x_labels = [label for label in labels if label not in additional_features and label not in y_label]
    
    additional_arr = []
    x_arr = []
    y_arr = []
    for var in labels:
        #print('Reading this variable:{}'.format(var))
        this_value = data_dict[var].flatten()
        if var in additional_features:
            additional_arr.append(this_value)
        elif var in x_labels:
            x_arr.append(this_value.compute())
        elif var in y_label:
            y_arr.append(this_value.compute())
    return additional_arr, x_arr, y_arr, x_labels, additional_features


def create_index_list_for_patch_region():
    """
    Remove the boundary cells that surrounding cells are beyond domain
    :return: list of index kept for cells
    """
    patch_range_xlon = range(2, 25)
    patch_range_xlat = range(2, 12)
    patch_indx = []
    iter_combs = [(i, j) for i in patch_range_xlon for j in patch_range_xlat]

    for i_indx_iter, j_indx_iter in iter_combs:
        patch_indx.append(405 * j_indx_iter + i_indx_iter)

    np.save('patch_cells_index', np.array(patch_indx))


def create_index_list_for_target_cells():
    """
    Remove the boundary cells that surrounding cells are beyond domain
    :return: list of index kept for cells
    """
    grids = (254, 405)
    iter_combs = [(i, j) for i in [0, 1, grids[1]-2, grids[1]-1] for j in range(0, grids[0])] +\
                 [(i, j) for j in [0, 1, grids[0]-1, grids[0] - 2] for i in range(0, grids[1])]
    full_indx = np.arange(0, 405*(grids[0]-1)+grids[1]-1).tolist()
    for i_indx_iter, j_indx_iter in iter_combs:
        try:
            full_indx.remove(405 * j_indx_iter + i_indx_iter)
        except ValueError:
            continue
    np.save('target_cells_index', np.array(full_indx))

def create_index_dict_for_surrouding_emis():
    """
    create indexs matrix for surrounding emission based on the index (i+-2, j+-2)
    :return: indx list of surrounding cells
    """
    orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus*')))
    data = nc.Dataset(orig_filenames[0])
    xlon = data['xlon'][:]
    xlat = data['xlat'][:]
    grids = xlon.shape
    surr_index = dict()
    for j_indx in range(grids[0]):
        print('Make surrouding cells for cell lat {}'.format(j_indx))
        for i_indx in range(grids[1]):
            indx = []
            iter_combs = [(i_indx_iter, j_indx_iter) for i_indx_iter in range(i_indx-2, i_indx+3)
                          for j_indx_iter in range(j_indx-2, j_indx+3)]
            for i_indx_iter, j_indx_iter in iter_combs:
                if i_indx_iter == i_indx and j_indx_iter == j_indx:
                    continue
                else:
                    indx.append(405*j_indx_iter+i_indx_iter)
            surr_index[405*j_indx+i_indx] = indx
    w = csv.writer(open("surrouding_cells.csv", "w"))
    for key, val in surr_index.items():
        w.writerow([key, val])


def find_indx_for_surrounding_emis_by_distance(i_indx, j_indx):
    """
    find the surrounding cells based on the distance less than 35km.
    it causes extra computations to reconciles number of surrounding cells across difference cells
    :param i_indx: indx along x axis
    :param j_indx: indx aloing y axis
    :return: indx list of surrounding cells
    """
    mat = scipy.io.loadmat('geo_info.mat')
    wrf_lon = np.array(mat['xlon'])
    wrf_lat = np.array(mat['xlat'])
    i_indx_low = max([0, i_indx - 5])
    i_indx_high = min([404, i_indx + 5])
    j_indx_low = max([0, j_indx - 5])
    j_indx_high = min([253, j_indx + 5])
    indx = []
    iter_combs = [(i_indx_iter, j_indx_iter) for i_indx_iter in range(i_indx_low, i_indx_high + 1)
                  for j_indx_iter in range(j_indx_low, j_indx_high + 1)]
    for i_indx_iter, j_indx_iter in iter_combs:
        this_dist = geopy.distance.distance((wrf_lat[i_indx, j_indx], wrf_lon[i_indx, j_indx]),
                                            (wrf_lat[i_indx_iter, j_indx_iter], wrf_lon[i_indx_iter, j_indx_iter])).km
        if 35 >= this_dist > 0:
            indx.append(405 * j_indx_iter + i_indx_iter)
    return indx


def find_indx_for_wind():
    """
    U and V are in stagger dimension
    find the corresponding index to do transformation
    :return: u_indx_left, u_indx_right, v_indx_bot, v_indx_up
    """
    grids = (254, 405)
    u_indx_left = []
    u_indx_right = []
    v_indx_up = []
    v_indx_bot = []
    for j_indx in range(grids[0]):
        for i_indx in range(grids[1]):
            u_indx_right.append(406*j_indx+i_indx+1)
            u_indx_left.append(406*j_indx+i_indx)
            v_indx_up.append(405*(j_indx+1)+i_indx)
            v_indx_bot.append(405*j_indx+i_indx)
    return u_indx_left, u_indx_right, v_indx_bot, v_indx_up


def prescribed_lightning_distribution():
    height = np.arange(0, 17)
    lightning_dis = [0.015, 0.235, 0.32, 0.0505, 0.065, 0.0575, 0.044, 0.0465, 0.057,
                     0.0555, 0.035, 0.0115, 0.0045, 0.003, 0, 0, 0]

def const_features_for_single_grid_single_file(grid_indx, wind_grid_indx, data):
    client = Client()
    dims = data['no2'].shape
    ntime = dims[0] - 1
    nvel = dims[2]
    data_dict = dict()
    data_hours = da.array(data['hour'][1:])
    data_dict['hour'] = da.repeat(data_hours[:, :], nvel, axis=1)
    data_dict['date'] = da.zeros((ntime, nvel)) + da.mean(data['date'][:])
    data_dict['date'] = data_dict['date']
    cum_ic_flash = da.array(data['IC_FLASHCOUNT'][:, grid_indx, :])
    cum_cg_flash = da.array(data['CG_FLASHCOUNT'][:, grid_indx, :])
    data_dict['IC_FLASHCOUNT'] = da.repeat(cum_ic_flash[1:, :] - cum_ic_flash[:-1, :], nvel, axis=1)
    data_dict['CG_FLASHCOUNT'] = da.repeat(cum_cg_flash[1:, :] - cum_cg_flash[:-1, :], nvel, axis=1)
    e_no_lower = da.array(data['E_NO'])[1:, grid_indx, :]
    e_no_upper = da.zeros((ntime, nvel - e_no_lower.shape[1]))
    data_dict['E_NO'] = da.concatenate([e_no_lower, e_no_upper], axis=1)
    data_dict['U'] = (data['U'][1:, wind_grid_indx[0][0], :] + data['U'][1:, wind_grid_indx[0][1], :])/2
    data_dict['V'] = (data['V'][1:, wind_grid_indx[1][0], :] + data['V'][1:, wind_grid_indx[1][1], :])/2

    match_vars = ['no2', 'pres', 'temp', 'CLDFRA']
    print('Variables read directly from wrf: {}'.format(match_vars[:]))
    for var in match_vars:
        data_dict[var] = da.array(data[var])[1:, grid_indx, :]

    reduce_dim_vars = ['elev', 'W']
    print('Variables average vertically: {}'.format(reduce_dim_vars[:]))
    for var in reduce_dim_vars:
        this_value = da.array(data[var])[1:, grid_indx, :]
        data_dict[var] = (this_value[:, 1:] + this_value[:, :-1]) / 2

    add_dim_vars = ['COSZEN', 'PBLH', 'LAI', 'HGT', 'SWDOWN', 'GLW']
    print('Variables add vertical layers: {}'.format(add_dim_vars[:]))

    for var in add_dim_vars:
        this_value = da.array(data[var])[1:, grid_indx, :]
        data_dict[var] = da.repeat(this_value, nvel, axis=1)

    print('Key of dict:{}'.format(data_dict.keys()))
    save_arr = []
    for var in data_dict.keys():
        data_dict[var] = data_dict[var].flatten()
        save_arr.append(data_dict[var])
    save_arr = da.array(save_arr).compute()
    return save_arr


def const_features_for_single_grid(i_indx, j_indx):
    grid_indx = 405*j_indx+i_indx
    wind_grid_indx = find_indx_for_wind(i_indx, j_indx)
    orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus_2005*')))
    train_filenames, test_filenames = train_test_filename(orig_filenames)
    train_arr = []
    test_arr = []
    for filename in orig_filenames:
        print(filename)
        data = nc.Dataset(filename)
        this_arr = const_features_for_single_grid_single_file(grid_indx, wind_grid_indx, data)
        if filename in train_filenames:
            train_arr.append(this_arr)
        else:
            test_arr.append(this_arr)
        del data
    np.save('train', np.array(train_arr))


if __name__=='__main__':
    create_index_list_for_patch_region()
    #create_index_list_for_target_cells()
    #orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus*')))
    #additional_arr, x_arr, y_arr, x_labels, additional_features = read_orig_file_from_wrf(orig_filenames[0])




