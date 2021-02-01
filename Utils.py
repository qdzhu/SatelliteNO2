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

    cluster.scale(n_nodes*6)
    client = Client(cluster)
    return client
def get_slurm_dask_client_savio2(n_nodes):
    cluster = SLURMCluster(cores=12,
                           memory='64GB',
                           project="co_aiolos",
                           walltime="12:00:00",
                           queue="savio",
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

    cluster.scale(n_nodes*32)
    client = Client(cluster)
    return client