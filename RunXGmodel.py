from Train_test_days_split import *
from Utils import *
import time
import xgboost as xgb
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def get_slurm_dask_client(n_workers):
    cluster = SLURMCluster(cores=24,
                           memory='128GB',
                           project="co_aiolos",
                           walltime="24:00:00",
                           queue="savio2_bigmem")

    cluster.scale(n_workers)
    client = Client(cluster)
    return client

def make_train_matrix(client):
    orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus_2005*')))
    train_filenames, test_filenames = train_test_filename(orig_filenames)
    train_filenames = train_filenames[0:1]
    x_total = []
    y_total = []
    add_total = []
    starttime = time.time()
    for train_filename in train_filenames:
        print('Reading in data from {}'.format(train_filename))
        additional_arr, x_arr, y_arr, x_labels, additional_features = read_orig_file_from_wrf(train_filename)
        add_total.append(additional_arr)
        x_total.append(x_arr)
        y_total.append(y_arr)
    print("--- {:.1f} minutes to read the data ---".format((time.time() - starttime) / 60))
    starttime = time.time()
    X = da.concatenate(x_total, axis=0).rechunk((62, 100))
    y = da.concatenate(y_total, axis=0).rechunk((1, 100))
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    print("--- {:.1f} minutes to concatenate the data---".format((time.time() - starttime) / 60))
    return dtrain

def make_xgboost_model(client, dtrain):
    starttime = time.time()
    bst = xgb.dask.train(client,
                            {'verbosity': 2,
                             'tree_method': 'hist',
                             'objective': 'reg:squarederror'
                             },
                            dtrain,
                            num_boost_round=4, evals=[(dtrain, 'train')])
    print("--- {} minutes ---".format((time.time() - starttime) / 60))
    config = bst.save_config()
    print(config)
    bst.save_model('2005.model')
    bst.dump_model('dump.raw.txt', 'featmap.txt')

if __name__=='__main__':
    client = get_slurm_dask_client_savio3(2)
    dtrain = make_train_matrix(client)
    make_xgboost_model(client, dtrain)
