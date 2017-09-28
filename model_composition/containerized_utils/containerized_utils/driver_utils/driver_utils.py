import json


class HeavyNodeConfig(object):
    def __init__(self,
                 name,
                 input_type,
                 model_image,
                 allocated_cpus,
                 cpus_per_replica,
                 slo=500000,
                 num_replicas=1,
                 gpus=None,
                 batch_size=1):
        self.name = name
        self.input_type = input_type
        self.model_image = model_image
        self.allocated_cpus = allocated_cpus
        self.cpus_per_replica = cpus_per_replica
        self.slo = slo
        self.num_replicas = num_replicas
        self.gpus = gpus
        self.batch_size = batch_size

    def to_json(self):
        return json.dumps(self.__dict__)


def setup_heavy_node(clipper_conn, config):
    clipper_conn.register_application(name=config.name,
                                      default_output="TIMEOUT",
                                      slo_micros=config.slo,
                                      input_type=config.input_type)

    clipper_conn.deploy_model(name=config.name,
                              version=1,
                              image=config.model_image,
                              input_type=config.input_type,
                              num_replicas=config.num_replicas,
                              batch_size=config.batch_size,
                              gpus=config.gpus,
                              allocated_cpus=config.allocated_cpus,
                              cpus_per_replica=config.cpus_per_replica)

    clipper_conn.link_model_to_app(app_name=config.name, model_name=config.name)

def save_results(configs, clipper_conn):
    """
    Parameters
    ----------
    configs : list(HeavyNodeConfig)
        The configs for any models deployed


    """
    pass

