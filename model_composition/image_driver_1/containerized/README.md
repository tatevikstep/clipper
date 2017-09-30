# Image Driver 1

## Background
First, here's a quick review of some relevant terminology:

* [**(Docker) Container**](https://www.docker.com/what-container): A self-contained virtual system for running software

* **Model**: A machine learning model implemented in one of several prominent frameworks (Tensorflow, Scikit, PyTorch, Theano, etc).

* **Replica**: A copy of a **model**. In practice, this is a single **container** running the model's code and dependencies.

* **Prediction Request**: A query sent to Clipper containing an application-specific input. For benchmarking purposes, each 
prediction request will contain an input of the type expected by the model being benchmarked (i.e. a 299x299x3 numpy array for 
the the VGG image featurization model).

* **Client**: Module of code for sending prediction requests to Clipper. Effectively, the driver file that you'll be using to benchmark models is a client.

* **Frontend**: A piece of Clipper's infrastructure that communicates with **clients** and or **replicas**. (For more information, see the [Clipper design doc](https://docs.google.com/document/d/1Ghc-CAKXzzRshSa6FlonFa5ttmtHRAqFwMg7vhuJakw/edit)).

This driver makes use of 4 heavyweight models: A VGG model for image featurization, an Inception V3 model for image featurization, 
an SVM with Kernel PCA for feature classification, and a light boosted gradient model (LGBM) for feature classification. The driver file,
[driver.py](driver.py), will help you benchmark each of this models with Clipper in isolation (one model at a time). The remaining sections of this README will get you started with the benchmarking process.

## Activate your Clipper Anaconda environment
Before proceeding, make sure to activate your Anaconda environment if it is not already activated. This can be done by running:
```sh
$ source activate clipper
```

## Pre-requisite: Setting up infrastructure
If the [high-perf-clipper branch](https://github.com/dcrankshaw/clipper/tree/high-perf-clipper) has been updated since you last
performed benchmarking, follow the subsequent instructions to build or install the latest benchmarking infrastructure components.

### Install the latest versions of Clipper's python modules:
Run the following command from the [model_composition directory](../../.)
```sh
$ ./setup_env.sh
```

### Build the latest Clipper docker images 
Run the following command from the clipper root directory to build the latest Clipper docker images 
(ZMQ Frontend, Management Frontend, etc):

```sh
$ ./bin/build_docker_images.sh
```

### Build the latest model docker images
Run the following command from the [containerized image driver 1 directory](.) 
to build docker images for models used by Image Driver 1:

```sh
$ ./containers/build_docker_images.sh
```

## Benchmarking with [driver.py](driver.py)

### The Driver API

The driver accepts the following arguments:
- **model_name**: The name of the model to benchmark. Must be one of the following: `vgg`, `inception`, `svm`, `lgbm`
  * This argument is REQUIRED
  
- **duration**: The duration for which each iteration of the benchmark should run, in seconds
  * If unspecified, this argument has a default value of 120 seconds
  
- **batch_sizes**: The batch size configurations to benchmark. Each configuration will be benchmarked separately.
  * If unspecified, the driver will benchmark a single batch size configuration of size `2`.
  
- **num_replicas**: The "number of replicas" configurations to benchmark. Each configuration will be benchmarked separately.
  * If unspecified, the driver will benchmark a single "number of replicas" configuration of size `1`
  
- **model_cpus**: The cpu cores on which model replicas should be run. Clipper will automatically handle the placement of each replica on one or more of these cpu cores (as specified by **cpus_per_replica**).

- **cpus_per_replica_nums**: The "number of cpu cores per replica" configurations to benchmark. Each configuration will be benchmarked
separately.

- **model_gpus**: The set of gpus on which to run replicas of the provided model. If you're running multiple replicas of
a model on GPUs, **each replica must have its own GPU**. Any replicas that don't have their own GPU will be executed on 
one or more of the CPU cores specified by **model_cpus**.


### Examples

1. As an example, consider the following driver command:

   ```sh
   $ python driver.py --duration_seconds 120 --model_name vgg --num_replicas 1 2 3 4 --batch_sizes 1 2 4 8 16 32 \
                      --model_cpus 20 21 22 23 24 25 26 27 --cpus_per_replica_nums 1,2
   ```

   This command specifies that:

   1. The VGG model will be benchmarked using cpus `20-27`
   
   2. `4` different "number of replicas" and `6` different "batch size" configurations will be benchmarked
   
   3. `2` different "number of cpu cores per replica" configurations will be benchmarked
      - Note that we cannot experiment with more than `max(num_replicas) / |model_cpus|` cpu cores per replica.
      - This means that, for this example, `max(cpus_per_replica_nums) = 2 cores` 
        (as reflected in the **cpus_per_replica_num** argument)

   4. Therefore, a total of `4 * 6 * 2 = 48` iterations lasting 120 seconds will occur (one for each configuration)

2. The last command benchmarked the VGG model on various CPU configurations. However, this model may perform better on a GPU.
Therefore, we might also run the following driver command:

 
   ```sh
   $ python driver.py --duration_seconds 120 --model_name vgg --num_replicas 1 2 3 4 --batch_sizes 1 2 4 8 16 32 \
                      --model_gpus 0,1 --model_cpus 20 21 22 23 --cpus_per_replica_nums 2
   ```  

   This command specifies that:
   
   1. As before, the VGG model will be benchmarked
   
   2. As before, `4` different "number of replicas" and `6` different "batch size" configurations will be benchmarked
   
   3. When the number of replicas exceeds the number of model gpus (`2`), two replicas will be run on GPUs 
      (one on GPU 0 and one on GPU 1), and the remaining replicas will be run on CPU cores `20-23`
      
   4. Each CPU-bound replica will be allocated `2` CPU cores.

### Avoiding CPU resource conflicts
When you're running replicas of a CPU-intensive model on a set of cores, `C = {c_1, ..., c_n}`, you want to avoid
starting other processes on these cores (or model performance might suffer). Therefore, you need to **make sure of two things**:

1. The set of cpu cores, `C`, allocated to the model (via **model_cpus**) does NOT intersect with the set of cpu cores allocated
to other pieces of Clipper's infrastructure. Therefore, **avoid using the following cpu cores**:

   * Core 0: The CPU core allocated to [Redis](https://redis.io/topics/introduction) (Clipper's configuration database)
   * Core 8: The CPU core allocated to Clipper's management frontend
   * Cores 1-5, 9-13: The CPU cores allocated to Clipper's ZMQ frontend
   
   **Note:** These core numbers are [configured in the driver ](https://github.com/dcrankshaw/clipper/blob/e2e292c0637327fed73df6a689df6f67677c0330/model_composition/image_driver_1/containerized/driver.py#L54-L56), and you can easily change them if necessary.
   
2. The benchmarking driver ([driver.py](driver.py)) is not executed on any CPU cores present in `C = {c_1, ..., c_n}`. 
   To make sure of this, you can use [numactl](https://linux.die.net/man/8/numactl) to control the allocation of CPU cores
   to the driver. **The driver should be allocated at least 4 cores**.
   
   For example, consider the second driver command from the previous section. We specified via **model_cpus** that
   replicas of the VGG model were allocated CPU cores 20-23. Therefore, we should have used `numactl` to run ([driver.py](driver.py))
   on a different set of cores as follows:
   
   ```sh
   $ numactl -C 24,25,26,27 python driver.py --duration_seconds 120 --model_name vgg --num_replicas 1 2 3 4 \
                                             --batch_sizes 1 2 4 8 16 32 --model_gpus 0,1 --model_cpus 20 21 22 23 \
                                             --cpus_per_replica_nums 2
   ```

## Monitoring the benchmarking process
Once you've started a benchmark, there are some useful tools and logs that you can use to monitor behavior.

### Monitoring CPU usage
If you're running replicas of a CPU-intensive model on a set of cores, `{c_1, ..., c_n}`, you can use [htop](http://hisham.hm/htop/)
to monitor for higher activity on those cores. 

### Monitoring GPU usage
If you're benchmarking the VGG or Inception models on a GPU, you can make sure that the model is using the GPU via the 
[nvidia-smi](http://developer.download.nvidia.com/compute/cuda/6_0/rel/gdk/nvidia-smi.331.38.pdf) command.

If no tasks are running on the GPU, the output of `nvidia-smi` will look similar to the following:

![Image of Unused Nvidia-Smi](imgs/nvidia-smi-unused.jpg)

In contrast, if tasks are running, you should see non-zero (hopefully high) memory and utilization, as well as one or more active processes (Note: This output was obtained from a machine with 2 GPUs):

![Image of Used Nvidia-Smi](imgs/nvidia-smi-used.jpg)

