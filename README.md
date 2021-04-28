# BayesLens

BayesLens is a purpose-built wrapper that relies on common (external) lensing codes for the galaxy cluster lensing likelihood and samples the posterior on parameters and hyperparameters. BayesLens is designed to permit an accurate description of cluster member galaxies in cluster lensing models.

## Installation

To use BayesLens you need to simply add the downloaded BayesLens directory to your system path.

```bash
export PATH=$PATH:/Users/user/.../BayesLens/CODE
```

WARNINGS:
- LensTool needs to be installed on your machine.

- To avoid bottleneck in the LensTool likelihood computation and to preserve the SSD (or the HD) of your machine, we strongly recommended to create a RAMdisk to store the tmp BayesLens files (see paper). You can find exhaustive instructions to create a RAMdisk [here](https://www.jamescoyle.net/how-to/943-create-a-ram-disk-in-linux). For MAC-OS see [here](https://blog.macsales.com/46348-how-to-create-and-use-a-ram-disk-with-your-mac-warnings-included/)


## Usage

### Use BayesLens to optimize a model

```bash
bayeslens run config_file
```

Yuo can find an example of BayesLens input_file in:
```bash
/BayesLens/EXAMPLES/TEST/test_par.dat
```

To obtain a list of available options:

```bash
bayeslens --h
```

WARNING:

- Before running BayesLens, use the following command to check if the LensTool input file has the right structure.

    ```bash
    bayeslens ltpar config_file
    ```

### Read BayesLens results
BayesLens results are stored in a .h5 file. The following commands can be used to unpack the results and save them in a readable format.

1) Save the MCMC chains for the parameters
    ```bash
    bayeslens_results rw config_file bk_file.h5 --chains 0 0
    ```
   substituting 0 0 with n1 n2, BayesLens saves only the chains for the parameters between n1 and n2. 

2) Save the best parameters values
    ```bash
    bayeslens_results rw config_file bk_file.h5 --val 0
    ```
   substituting 0 with -1, it saves the modes of the marginalized posterior distributions for the parameters
   
   substituting 0 with -2, it saves the medians of the marginalized posterior distributions for the parameters
   
   substituting 0 with n(>0), it saves n randomly extracted chain steps

    HINT 1: From these BayesLens_results files, you can create a LensTool input file with the command
    ```bash
    bayeslens_results lt config_file BayesLens_results.dat
    ```
    HINT 2: The randomly extracted chains are useful to create a series of LensTool maps with the following command:
    ```bash
    bayeslens_maps config_file BayesLens_results_random.dat
    ```

3) Save the walkers movements for parameters between n1 and n2
    ```bash
    bayeslens_results rw config_file bk_file.h5 --wm n1 n2
    ```
   

To obtain a list of available options:

```bash
bayeslens_results --h
```

### Run the test example
We have prepared a simple example of a BayesLens run. All outputs of the following commands are saved in:
```bash
/BayesLens/EXAMPLES/TEST/RESULTS
```

Go into the directory containing the test example files:
```bash
cd /BayesLens/EXAMPLES/TEST/
```

We sample the model posterior with the following command:
```bash
bayeslens run inputs_test.dat --n_walkers 100 --n_steps 5000 --ramdisk absolute_path_to_ramdisk
```
The posterior sampling is performed using 100 walkers for a total of 5000 steps each.

To plot the walkers movements for cluster-scale halo parameters (4 to 10) we use the command:
```bash
bayeslens_results rw inputs_test.dat BayesLens.h5 --wm 4 10
```
Now, we save the best-fit lens model (model that maximize the total posterior) and we generate the LensTool best.par file
```bash
bayeslens_results rw inputs_test.dat BayesLens.h5 --val 0
```
```bash
bayeslens_results lt inputs_test.dat BayesLens_best.dat
```

Finally, we save the MCMC chains in a LensTool format (a burn-in of 3000 steps is assumed)
```bash
bayeslens_results rw inputs_test.dat BayesLens.h5 --chains 0 0
```

The marginalized posterior distribution can be plotted using [corner.py](https://corner.readthedocs.io/en/latest/index.html). For the cluster-scale halo we obtain the following plot with, in red, the input values:

```bash
/BayesLens/EXAMPLES/TEST/RESULTS/degeneracy_cluster-scale_halo.pdf
```
Finally we extract 10 random walker positions from the MCMC chains (without the burn-in = 3000 steps). The output file will be used to produce cluster mass maps (bayeslens_maps is parallelized).
```bash
bayeslens_results rw inputs_test.dat BayesLens.h5 --resume support/chains_burnin_3000.npy --val 10

bayeslens_maps inputs_test.dat BayesLens_random_10.dat --ramdisk absolute_path_to_ramdisk

>>> Type LensTool line to generate the maps: mass 4 500 0.439 mass.fits
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Free license: You are completely free to use and modify the code.
