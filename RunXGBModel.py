import dask
import xgboost as xgb
from Train_test_days_split import *
import xarray as xr
import time
import dask.dataframe as dd
from dask.utils import tmpfile
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

dask.config.set(temporary_directory='/global/home/users/qindan_zhu/myscratch/qindan_zhu/SatelliteNO2')

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

def get_slurm_dask_client(n_workers, n_cores):
    cluster = SLURMCluster(cores=n_cores,
                           memory='32GB',
                           project="co_aiolos",
                           walltime="02:00:00",
                           queue="savio2_gpu",
                           job_extra=['--gres=gpu:1','--cpus-per-task=2'])

    cluster.scale(n_workers)
    client = Client(cluster)
    return client

def get_slurm_dask_client_bigmem(n_nodes):
    cluster = SLURMCluster(cores=24,
                           memory='128GB',
                           project="co_aiolos",
                           walltime="02:00:00",
                           queue="savio2_bigmem",
                           local_directory = '/global/home/users/qindan_zhu/myscratch/qindan_zhu/SatelliteNO2',
                            job_extra=['--qos="savio_lowprio"'])

    cluster.scale(n_nodes*4)
    client = Client(cluster)
    return client

def get_slurm_dask_client_savio2(n_nodes):
    cluster = SLURMCluster(cores=12,
                           memory='64GB',
                           project="co_aiolos",
                           walltime="10:00:00",
                           queue="savio",
                           local_directory = '/global/home/users/qindan_zhu/myscratchu/qindan_zhu/SatelliteNO2',
                           job_extra=['--qos="aiolos_savio_normal"'])

    cluster.scale(n_nodes*4)
    client = Client(cluster)
    return client

def get_slurm_dask_client_savio3(n_nodes):
    cluster = SLURMCluster(cores=32,
                           memory='96GB',
                           project="co_aiolos",
                           walltime="08:00:00",
                           queue="savio3",
                           local_directory = '/global/home/users/qindan_zhu/myscratch/qindan_zhu/SatelliteNO2',
                           job_extra=['--qos="aiolos_savio3_normal"'])

    cluster.scale(n_nodes*8)
    client = Client(cluster)
    return client

nvel = 29
ngrid = 100250
ntime = 16
keep_indx = np.load('patch_cells_index.npy')
#keep_indx = np.load('target_cells_index.npy')
reader = csv.reader(open("surrouding_cells.csv", "r"))
surr_arr = []
for row in reader:
    this_arr = list(map(int, row[1].strip(']|[').split(',')))
    surr_arr.append(this_arr)
surr_arr = np.array(surr_arr)[keep_indx, :]


def read_var_from_file_xr(filename, varname):
    ds = xr.open_mfdataset(filename, parallel=True)
    this_var = da.array(ds[varname][1:, keep_indx, :]).flatten()
    del ds
    return this_var


def read_var_from_file_nc(filename, varname):
    ds = nc.Dataset(filename)
    this_var = da.array(ds[varname][1:, keep_indx, :]).flatten()
    del ds
    return this_var


def read_2d_var_from_file_xr(filename, varname):
    ds = xr.open_mfdataset(filename, parallel=True)
    this_var = da.repeat(da.array(ds[varname][1:, keep_indx, :]),nvel,axis=2).flatten()
    del ds
    return this_var

def read_hour_from_file_xr(filename):
    ds = xr.open_mfdataset(filename, parallel=True)
    this_data = da.array(ds['hour']).reshape((17, 1, 1))
    this_var = da.repeat(da.repeat(da.array(this_data[1:, :, :]), len(keep_indx), axis=1), nvel, axis=2).flatten()
    del ds
    return this_var


def read_wind_from_file_xr(filename):
    ds = xr.open_mfdataset(filename, parallel=True)
    stg_u = da.array(ds['U'])
    stg_v = da.array(ds['V'])
    stg_w = da.array(ds['W'][1:, keep_indx, :])
    u_indx_left, u_indx_right, v_indx_bot, v_indx_up = find_indx_for_wind()
    wind_u = (stg_u[1:, u_indx_left, :] + stg_u[1:, u_indx_right, :])/2
    u = wind_u[:, keep_indx, :].flatten()
    wind_v = (stg_v[1:, v_indx_up, :] + stg_v[1:, v_indx_bot, :])/2
    v = wind_v[:, keep_indx, :].flatten()
    w = (stg_w[:, :, 1:] + stg_w[:, :, :-1])/2
    w = w.flatten()
    return da.stack([u,v,w])


def read_lightning_from_file_xr(filename):
    ds = xr.open_mfdataset(filename, parallel=True)
    cum_ic_flash = da.array(ds['IC_FLASHCOUNT'][:])
    cum_cg_flash = da.array(ds['CG_FLASHCOUNT'][:])
    ic_flash = da.repeat(cum_ic_flash[1:, :, :]-cum_ic_flash[:-1, :, :], nvel, axis=2)
    cg_flash = da.repeat(cum_cg_flash[1:, :, :]-cum_cg_flash[:-1, :, :], nvel, axis=2)
    e_lightning = ic_flash + cg_flash
    lightning_dis = [0.015, 0.235, 0.32, 0.0505, 0.065, 0.0575, 0.044, 0.0465, 0.057,
                     0.0555, 0.035, 0.0115, 0.0045, 0.003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lightning_dis = da.array(lightning_dis).reshape(1, 1, nvel)
    lightning_dis = da.repeat(lightning_dis, ntime, axis=0)
    lightning_dis = da.repeat(lightning_dis, e_lightning.shape[1], axis=1)
    e_lightning = e_lightning * lightning_dis
    this_var = e_lightning[:, keep_indx, :].flatten()
    surrs = []
    surrs.append(this_var)
    for i in range(0, 24):
        surr_indx = surr_arr[:, i]
        surrs.append(e_lightning[:, surr_indx, :].flatten())
    return da.stack(surrs)


def read_anthro_emis_from_file_xr(filename):
    ds = xr.open_mfdataset(filename, parallel=True)
    e_no_lower = da.array(ds['E_NO'])[1:, :, :]
    e_no_upper = da.zeros((ntime, e_no_lower.shape[1], nvel - e_no_lower.shape[2]))
    e_no = da.concatenate([e_no_lower, e_no_upper], axis=2)
    e_no_total = []
    e_no_total.append(e_no[:, keep_indx, :].flatten())
    for i in range(0, 24):
        surr_indx = surr_arr[:, i]
        e_no_total.append(e_no[:, surr_indx, :].flatten())
    return da.stack(e_no_total)


def make_xgbmodel(client, filename):
    varnames = ['no2', 'pres', 'temp', 'CLDFRA']
    futures = []
    for varname in varnames:
        future = client.submit(read_var_from_file_xr, filename, varname)
        futures.append(future)
    dirt_feature = da.stack([future.result() for future in futures])
    client.cancel(futures)
    hour_feature = client.submit(read_wind_from_file_xr, filename)
    hour_feature = hour_feature.result()
    varnames_2d = ['COSZEN', 'PBLH', 'LAI', 'HGT', 'SWDOWN', 'GLW']
    futures_2d = []
    for varname in varnames_2d:
        future_2d = client.submit(read_2d_var_from_file_xr, filename, varname)
        futures_2d.append(future_2d)
    dirt2d_feature = da.stack([future.result() for future in futures_2d])
    client.cancel(futures_2d)
    wind_feature = client.submit(read_wind_from_file_xr, filename)
    anthroemis_future = client.submit(read_anthro_emis_from_file_xr, filename)
    lightning_future = client.submit(read_lightning_from_file_xr, filename)
    datasets = da.transpose(da.concatenate((dirt_feature, hour_feature, dirt2d_feature, wind_feature.result(),
                                            anthroemis_future.result(), lightning_future.result()), axis=0))
    client.cancel(dirt_feature)
    client.cancel(dirt2d_feature)
    del wind_feature
    del anthroemis_future
    del lightning_future
    datasets_future = client.scatter(datasets)
    return datasets_future


if __name__=='__main__':
    #make_training_patch_region_single()
    #xgbmodel_training_patch_region()
#    save_datasets_test()
#    xgbmodel_testing_continuous()
#    xgbmodel_training_continuous()
#    client = get_slurm_dask_client_bigmem(8)
#    client = get_slurm_dask_client_savio2(12)
    if False:
        client = get_slurm_dask_client_savio3(4)
        client.wait_for_workers(32)
        orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus_2005*')))
        train_filenames, test_filenames = train_test_filename(orig_filenames)
        bst = make_xgbmodel_final(client, train_filenames[0:2])
        print('Restart the client')
        client.restart()
        for i in range(2, len(train_filenames), 2):
            bst = make_xgbmodel_final_update(client, train_filenames[i:i+2], bst, i)
            print('Restart the client')
            client.restart()
