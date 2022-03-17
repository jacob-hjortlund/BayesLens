#!/usr/bin/env python
# ==========================================================
# Author: Pietro Bergamini - pietro.bergamini@phd.unipd.it
# ==========================================================


import numpy as np
import scipy.optimize as op
import multiprocessing as mp
import emcee
from pathlib import Path
import os

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util_likelihoods_priors import partial_posterior, lnposterior



def walker_init(priors_bounds_w, translation_vector_w, dim_w, n_walkers_w, working_dir, ramdisk, translation_vector_ex,
                mag_ex, lenstool_vector, header, image_file, deprojection_matrix):
    """
    Inizialize walkers positions

    :param priors_bounds_w: see *BayesLens_parser
    :param translation_vector_w: see *BayesLens_parser
    :param dim_w: see *BayesLens_parser
    :param n_walkers_w: see *BayesLens_parser
    :param working_dir: see *BayesLens_parser
    :param ramdisk: path to RAMDISK
    :param translation_vector_ex: see *BayesLens_parser
    :param mag_ex: see *BayesLens_parser
    :param lenstool_vector: see *BayesLens_parser
    :param header: see *BayesLens_parser
    :param image_file: see *BayesLens_parser
    :param deprojection_matrix: see *BayesLens_parser
    :return: array containing walker positions
    """
    ## WALKER INIZIALIZATION: WALKERS ARE INIZIALIZED INSIDE A SPHERE AROUND THE MAXIMUM OF THE PARTIAL POSTERIOR: partial_posterior ##
    print('\nWalkers inizialization...')
    init = np.zeros(len(priors_bounds_w[:, 0]))
    mask_vd = (np.asarray(translation_vector_w[:, 0], dtype=float) >= 2) & (
                np.asarray(translation_vector_w[:dim_w, 0], dtype=float) < 3)

    # STARTING GUESS FOR THE PARAMETERS FOR partial_posterior MAXIMIZATION
    init[~mask_vd] = priors_bounds_w[:, 0][~mask_vd] + (
                (priors_bounds_w[:, 1])[~mask_vd] - (priors_bounds_w[:, 0])[~mask_vd]) / 2
    init[mask_vd] = priors_bounds_w[:, 0][mask_vd]

    # partial_posterior  MAXIMIZATION
    nll = lambda *args: -partial_posterior(*args)
    results = op.minimize(nll, init, args=(priors_bounds_w, translation_vector_w))

    pos = [results["x"] + 1e-4 * np.random.randn(len(init)) for i in range(n_walkers_w)]

    mask_halo_cosmo = (
        (np.asarray(translation_vector_w[:dim_w, 0], dtype=float) < 0) |
        (
            (np.asarray(translation_vector_w[:dim_w, 0], dtype=float) >= 1) &
            (np.asarray(translation_vector_w[:dim_w, 0], dtype=float) < 2)
        )
    )

    for i in range(n_walkers_w):
        pos[i][mask_halo_cosmo] = priors_bounds_w[:dim_w, 0][mask_halo_cosmo] + (
                (priors_bounds_w[:dim_w, 1])[mask_halo_cosmo] -
                (priors_bounds_w[:dim_w, 0])[
                    mask_halo_cosmo]) * np.random.uniform(0., 1., (len((priors_bounds_w[:dim_w, 0])[mask_halo_cosmo])))

    return pos


def run_sampler(n_walkers_r, dim_r, lnposterior_r, priors_bounds_r, working_dir_r, translation_vector_r,
                lenstool_vector_r, header_r, image_file_r, ramdisk_r, deprojection_matrix_r, n_threads_r, backend_r,
                n_steps_r, translation_vector_ex_r, mag_ex_r, pos_r=None):
    """
    Run emcee

    :param n_walkers_r: Number of walkers
    :param dim_r: dimension of parameter space
    :param lnposterior_r: total posterior
    :param priors_bounds_r: see *BayesLens_parser
    :param working_dir_r: working directory
    :param translation_vector_r: see *BayesLens_parser
    :param lenstool_vector_r: see *BayesLens_parser
    :param header_r: see *BayesLens_parser
    :param image_file_r: see *BayesLens_parser
    :param ramdisk_r: path to RAMDISK
    :param deprojection_matrix_r: see *BayesLens_parser
    :param n_threads_r: number of CPU threads to be used
    :param backend_r: emcee backend
    :param n_steps_r: walkers steps
    :param translation_vector_ex_r: see *BayesLens_parser
    :param mag_ex_r: see *BayesLens_parser
    :param pos_r: array with walkers initial positions
    :return: array with results and sampler object
    """
    # START THE SAMPLER
    print('\nSampler started...')
    with mp.Pool(n_threads_r) as pool:
        sampler = emcee.EnsembleSampler(n_walkers_r, dim_r, lnposterior_r, args=(
        priors_bounds_r, working_dir_r, translation_vector_r, lenstool_vector_r, header_r, image_file_r, ramdisk_r,
        deprojection_matrix_r, translation_vector_ex_r, mag_ex_r), pool=pool, backend=backend_r)

        results = sampler.run_mcmc(pos_r, n_steps_r, progress=True)

    return results, sampler


def best_chain(dir_working, bk_c, translation_vector_c, dir_out):
    '''
    Save the best chain. IT IS NOT USE IN THE CURRENT VERSION OF BayesLens

    :param bk_c: BayesLens output file name
    :param translation_vector_c: see *BayesLens_parser
    :return: Save the best chain
    '''
    print('Extraction of the best chain: ' + bk_c)
    filename = dir_working + bk_c + '.h5'
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    chains = reader.get_chain(discard=0, flat=True)
    log_prob_samples = reader.get_log_prob(discard=0, flat=True)
    chains_tot = np.concatenate((log_prob_samples.reshape(1, len(log_prob_samples)).T, chains), axis=1)

    max_index, = np.where(log_prob_samples == max(log_prob_samples))

    comments = 'ln(Lhood)'
    for i, id in enumerate(translation_vector_c[:, 0]):
        comments += '\n' + id + '_' + translation_vector_c[i, 1]

    np.savetxt(dir_out + bk_c + '_best.dat', chains_tot[max_index, :], fmt='%.4f', delimiter='\t', header=comments)

    del bk_c, translation_vector_c


def BayesLens_emcee(priors_bounds, working_dir, translation_vector, lenstool_vector, header, image_file,
                    deprojection_matrix, translation_vector_ex, mag_ex, n_walkers=100, n_steps=500, n_threads=1,
                    ramdisk='', bk='BayesLens', mf=[1, 0]):
    """
    Run BayesLens optimization

    :param priors_bounds: see *BayesLens_parser
    :param working_dir: working directory
    :param translation_vector: see *BayesLens_parser
    :param lenstool_vector: see *BayesLens_parser
    :param header: see *BayesLens_parser
    :param image_file: see *BayesLens_parser
    :param deprojection_matrix: see *BayesLens_parser
    :param translation_vector_ex: see *BayesLens_parser
    :param mag_ex: see *BayesLens_parser
    :param n_walkers: number of walkers
    :param n_steps: number of walker steps
    :param n_threads: number of CPU threads to be used
    :param ramdisk: path to RAMDISK
    :param bk: name of output BayesLens file
    :param mf: array with two entryes. If mf[0] = N, split BayesLens outputs subsequent N files. If mf[1] = 1, removes previous saved files
    :return: Save results in .h5 files and plot the acceptance fraction.
    """
    # INIZIALIZE THE FILE WITH THE RESULTS
    filename = working_dir + bk + '.h5'
    # filename = working_dir + 'BayesLens.h5'
    my_file = Path(filename)
    backend = emcee.backends.HDFBackend(filename)

    print('Number of threads: ' + str(n_threads))

    free_par_mask = (priors_bounds[:,0] != 0) | (priors_bounds[:,1] != 0)

    for i in np.arange(mf[0]):
        print('\nRUN: ' + str(i + 1) + ' of ' + str(mf[0]))
        if i == 0:

            os.makedirs(working_dir + 'acc_frac')
            os.makedirs(working_dir + 'best_par')

            ## CONTINUE A PRECEDING RUN APPENDING THE CHAINS TO THE SAME BayesLens.h5 FILE ##
            if my_file.is_file():

                dim = backend.shape[1]
                print('\nNumber of free parameters: ', dim)

                n_walkers = backend.shape[0]
                print('\nNumber of walkers: ', n_walkers)

                results, sampler_o = run_sampler(n_walkers, dim, lnposterior, priors_bounds, working_dir,
                                                 translation_vector, lenstool_vector, header, image_file, ramdisk,
                                                 deprojection_matrix, n_threads, backend, n_steps,
                                                 translation_vector_ex, mag_ex, pos_r=None)

            ### CREATE A NEW RUN ###
            else:
                # DIMENSION OF PARAMETER-SPACE
                dim = len(priors_bounds[free_par_mask])
                print('\nNumber of free parameters: ', dim)

                # DEFINE THE NUMBER OF WALKERS n_walkers, ENSURE n_walkers >= 2*(N° OF PARAMETERS)+2
                if n_walkers <= (dim * 2) + 2:
                    n_walkers = (dim * 2) + 2
                print('\nNumber of walkers: ', n_walkers)

                backend.reset(n_walkers, dim)

                pos = walker_init(priors_bounds[free_par_mask], translation_vector[free_par_mask], dim, n_walkers, working_dir, ramdisk,
                                  translation_vector_ex, mag_ex, lenstool_vector, header, image_file,
                                  deprojection_matrix)

                results, sampler_o = run_sampler(n_walkers, dim, lnposterior, priors_bounds, working_dir,
                                                 translation_vector, lenstool_vector, header, image_file, ramdisk,
                                                 deprojection_matrix, n_threads, backend, n_steps,
                                                 translation_vector_ex, mag_ex, pos_r=pos)

        else:
            if i > 1:
                best_chain(working_dir, bk + '_' + str(i - 1), translation_vector, working_dir + 'best_par/')
                if mf[1] == 1:
                    os.remove(working_dir + bk + '_' + str(i - 1) + '.h5')
                    print('File removed: ' + bk + '_' + str(i - 1) + '.h5')

            filename = working_dir + bk + '_' + str(i) + '.h5'
            backend = emcee.backends.HDFBackend(filename)
            results, sampler_o = run_sampler(n_walkers, dim, lnposterior, priors_bounds, working_dir,
                                             translation_vector, lenstool_vector, header, image_file, ramdisk,
                                             deprojection_matrix, n_threads, backend, n_steps, translation_vector_ex,
                                             mag_ex, pos_r=results)

        frac_update = sampler_o.acceptance_fraction

        print("Mean acceptance fraction: {0:.3f}".format(np.mean(frac_update)))

        # plt.plot(frac_update[0:len(frac_update) - 1], frac_update)

        plt.clf()
        plt.plot(np.arange(len(frac_update)), frac_update)
        plt.hlines(np.mean(frac_update),0,len(frac_update),linestyles='--')
        plt.title('Acceptance Fraction (mean = %.3f)'%np.mean(frac_update))
        plt.savefig(working_dir + 'acc_frac/accfrac_' + str(i) + '.pdf')
