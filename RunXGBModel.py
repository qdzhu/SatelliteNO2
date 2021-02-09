import dask
import xgboost as xgb
from Train_test_days_split import *
import xarray as xr
import time

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
                           job_extra=['--qos="savio_lowprio"'])

    cluster.scale(n_nodes*4)
    client = Client(cluster)
    return client

def get_slurm_dask_client_savio2(n_nodes):
    cluster = SLURMCluster(cores=12,
                           memory='64GB',
                           project="co_aiolos",
                           walltime="12:00:00",
                           queue="savio",
                           local_directory = '/global/home/users/qindan_zhu/myscratch/qindan_zhu/SatelliteNO2',
                           job_extra=['--qos="aiolos_savio_normal"'])

    cluster.scale(n_nodes*6)
    client = Client(cluster)
    return client
def get_slurm_dask_client_savio3(n_nodes):
    cluster = SLURMCluster(cores=32,
                           memory='96GB',
                           project="co_aiolos",
                           walltime="72:00:00",
                           queue="savio3",
                           job_extra=['--qos="aiolos_savio3_normal"'])

    cluster.scale(n_nodes*16)
    client = Client(cluster)
    return client

nvel = 29
ngrid = 100250
ntime = 16
keep_indx = np.load('target_cells_index.npy')
reader = csv.reader(open("surrouding_cells.csv", "r"))
surr_arr = []
for row in reader:
    this_arr = list(map(int, row[1].strip(']|[').split(',')))
    surr_arr.append(this_arr)
surr_arr = np.array(surr_arr)[keep_indx, :]

def read_var_from_file_xr(filename, varname):
    ds = xr.open_mfdataset(filename, parallel=True)
    this_var = da.array(ds[varname][1:, keep_indx, :]).flatten()
    return this_var

def read_var_from_file_nc(filename, varname):
    ds = nc.Dataset(filename)
    this_var = da.array(ds[varname][1:, keep_indx, :]).flatten()
    return this_var

def read_2d_var_from_file_xr(filename, varname):
    ds = xr.open_mfdataset(filename, parallel=True)
    this_var = da.repeat(da.array(ds[varname][1:, keep_indx, :]),nvel,axis=2).flatten()
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
    varnames_2d = ['COSZEN', 'PBLH', 'LAI', 'HGT', 'SWDOWN', 'GLW']
    futures_2d = []
    for varname in varnames_2d:
        future_2d = client.submit(read_2d_var_from_file_xr, filename, varname)
        futures_2d.append(future_2d)
    dirt2d_feature = da.stack([future.result() for future in futures_2d])
    wind_feature = client.submit(read_wind_from_file_xr, filename)
    anthroemis_future = client.submit(read_anthro_emis_from_file_xr, filename)
    lightning_future = client.submit(read_lightning_from_file_xr, filename)
    datasets = da.transpose(da.concatenate(
        (dirt_feature, dirt2d_feature, wind_feature.result(), anthroemis_future.result(), lightning_future.result()),
        axis=0))
    X = datasets[:, 1:]
    y = datasets[:, 0]
    X, y = dask.persist(X, y)
    del datasets
    for future in futures:
        del future
    for future in futures_2d:
        del future
    del wind_feature
    del anthroemis_future
    del lightning_future
    chunksize = int(X.shape[0] / 50)
    X = X.rechunk(chunks=(chunksize, 62))
    y = y.rechunk(chunks=(chunksize, 1))
    print('Start making training datasets')
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    print('Start running xgboost model')
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             'tree_method': 'hist',
                             'objective': 'reg:squarederror',
                             'subsample': 0.1,
                             'sampling_method': 'gradient_based'
                             },
                            dtrain,
                            num_boost_round=4, evals=[(dtrain, 'train')])
    print('Training is complete')
    bst = output['booster']
    hist = output['history']
    print(hist)
#    bst.save_model('/global/home/users/qindan_zhu/PYTHON/SatelliteNO2/2005.model')
    print('Saving model')
    bst.dump_model('/global/home/users/qindan_zhu/PYTHON/SatelliteNO2/dump.raw.txt', '/global/home/users/qindan_zhu/PYTHON/SatelliteNO2/featmap.txt')

if __name__=='__main__':
    client = get_slurm_dask_client_savio2(5)
    orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus*')))
    orig_filenames = orig_filenames[0:5]
    train_filenames, test_filenames = train_test_filename(orig_filenames)
    filename = train_filenames[0]
    make_xgbmodel(client, filename)
