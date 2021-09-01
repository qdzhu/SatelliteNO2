import dask
import xgboost as xgb
from Train_test_days_split import *
from RunXGBModel import *
from Train_test_days_split import *
import dask.dataframe as dd
import argparse
from dask_jobqueue import SLURMCluster

save_filepath = '/global/home/users/qindan_zhu/myscratchu/qindan_zhu/Sat_NO2_outputs_lightning'
save_modelpath = '/global/home/users/qindan_zhu/PYTHON/SatelliteNO2/Outputs_annual'

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
    num_worker = n_nodes*4
    return client, num_worker


def prepare_inputs(client):
    orig_file_path = '/global/scratch-old/jlgrant/ML-WRF/ML-WRF/'
    orig_filenames = sorted(glob(os.path.join(orig_file_path, 'met_conus*')))
    for this_filename in orig_filenames:
        save_filename = os.path.basename(this_filename).replace('conus', 'patch')
        if not os.path.isfile(save_filename):
            print('Reading training data {}'.format(this_filename))
            this_dataset = make_xgbmodel(client, this_filename)
            total_datasets = this_dataset.result()
            df = dd.io.from_dask_array(total_datasets)
            df.to_csv(os.path.join(save_filepath, save_filename), index=False, single_file=True)
            del total_datasets
        else:
            print('File already exists {}'.format(this_filename))
    return


def read_inputs(client, years):
    create_df = True
    for year in years:
        filename = 'met_patch_{}_*'.format(year)
        if create_df:
            df = dd.read_csv(os.path.join(save_filepath, filename))
            create_df = False
        else:
            this_df = dd.read_csv(os.path.join(save_filepath, filename))
            df = dd.concat([df, this_df])
    X = df.iloc[:, 1:]
    y = np.log(df.iloc[:, 0] * 1e3)
    X, y = client.persist([X, y])
    return X, y


def train_model(client, years):
    print('Start making training datasets')
    X, y = read_inputs(client, years)
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             'tree_method': 'hist',
                             'objective': 'reg:squarederror',
                             'subsample': 1,
                             'sampling_method': 'gradient_based'
                             },
                            dtrain, num_boost_round=100, evals=[(dtrain, 'train')])
    bst = output['booster']
    bst.save_model(os.path.join(save_modelpath, 'model.model'))



def do_prediction(client):
    X, y = read_inputs(client, 2014)
    ref_no2 = y.compute()
    pres = X.iloc[:, 0].compute()
    dtest = xgb.dask.DaskDMatrix(client, X, y)

    save_data = []
    save_data.append(pres)
    save_data.append(ref_no2)

    bst_assemble = []
    filenames_assemble = sorted(glob(os.path.join(save_modelpath,
                                                  'model_v*.model')))
    print('Start making training datasets')
    for filename in filenames_assemble:
        print(filename)
        bst = xgb.Booster({'nthread': 40})
        bst.load_model(filename)
        pred_no2 = xgb.dask.predict(client, bst_assemble[0], dtest)
        save_data.append(pred_no2.compute())
    np.save(os.path.join(save_modelpath, 'model_v_res.npy'), save_data)



def _get_args():
    parser = argparse.ArgumentParser(description='Build one of the chemical mechanisms for the PECANS model',
                                     epilog='If no mechanism name is provided, a user interactive prompt is given.')
    parser.add_argument('--node', '-n', default=5, type=int, help='Number of nodes')
    parser.add_argument('--process', '-p', default='Train', type=str, help='Which process you want to run?')
    parser.add_argument('--years', '-y', nargs='+', type=int, help='The years of inputs')
    args = parser.parse_args()
    return args


def main():
    args = _get_args()
    print('Start acquire nodes')
    client, num_worker = get_slurm_dask_client_savio2(args.node)
    client.wait_for_workers(num_worker)
    print('Nodes are ready')
    if args.process == 'Prepare':
        prepare_inputs(client)
    elif args.process == 'Train':
        train_model(client, args.years)
    elif args.process == 'Predict':
        do_prediction(client)
    client.shutdown()
    

if __name__=='__main__':
    main()
