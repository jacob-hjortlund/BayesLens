#!/usr/bin/env python
# ==========================================================
# Author: Pietro Bergamini - pietro.bergamini@phd.unipd.it
# ==========================================================

import numpy as np
import emcee
import tqdm
from scipy import stats
import random
import os
from subprocess import Popen, PIPE
import glob
import shutil

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from util_func import randomstring
from util_lenstool import BayesLens_writer


def BayesLens_read_file(dir, translation_vector, wm, b, c, par, bk):
    """
    Reads BayesLens output files and creates useful outputs

    :param dir: Directory containing the output files
    :param translation_vector: see *BayesLens_parser
    :param wm: see bayeslens_results
    :param b: see bayeslens_results
    :param c: see bayeslens_results
    :param par: see bayeslens_results
    :param bk: BayesLens output file name
    :return: save output files
    """

    if c == 'none':
        filename = dir + bk + '.h5'
        reader = emcee.backends.HDFBackend(filename, read_only=True)

        ### TRY TO REMOVE A BURN-IN EQUAL TO TWO TIMES THE MAXIMUM AUTOCORRELATION TIME
        try:
            tau = reader.get_autocorr_time()
            print('Autocorrelation time: ', tau)
            ask = input('\nType n to quit, everything else to continue: ')
            if ask == 'n':
                quit()
            burnin = int(2 * np.max(tau))
            thin = int(1)
            chains = reader.get_chain(discard=burnin, flat=True, thin=thin)
            log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
            print("thin: {0}".format(thin))

        ### IF AUTOCORRELATION TIMES ARE NOT WELL DETERMINED YOU CAN SPECIFY A BURN-IN BY HAND
        except:
            burnin = int(input('You need more steps for an auto burnin. Specify a burnin value: '))
            chains = reader.get_chain(discard=burnin, flat=True)
            log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)

        print("burn-in: {0}".format(burnin))
        print("flat chain shape: {0}".format(chains.shape))
        print("flat log prob shape: {0}".format(log_prob_samples.shape))

        chains_tot = np.concatenate((log_prob_samples.reshape(1, len(log_prob_samples)).T, chains), axis=1)

        chains_2 = reader.get_chain()

        np.save(dir + 'support/chains_burnin_' + str(burnin) + '.npy', chains_tot)
        np.save(dir + 'support/chains.npy', chains_2)
        print('\nChains without burnin saved in support/chains_burnin_' + str(burnin) + '.npy')
        print('Use the option --c to resume that chains')

    else:
        # chains_tot = np.load(dir + 'support/' + c)
        chains_tot = np.load(c)
        chains_2 = np.load(dir + 'support/chains.npy')
        log_prob_samples = chains_tot[:, 0]

    if wm[1] != 0:
        ask_walkerplot_low = wm[0]
        ask_walkerplot_up = wm[1]
        fig, axes = plt.subplots(int(ask_walkerplot_up) - int(ask_walkerplot_low), figsize=(10, 7), sharex=True)
        for i in range(int(ask_walkerplot_up) - int(ask_walkerplot_low)):
            ax = axes[i]
            ax.plot(chains_2[:, :, i + int(ask_walkerplot_low)], "k", alpha=0.3)
            ax.set_xlim(0, len(chains_2))
            ax.set_ylabel(translation_vector[i + int(ask_walkerplot_low), 0] + '_' + translation_vector[
                i + int(ask_walkerplot_low), 1])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.axvline(x=burnin, color='red', linestyle='--')

        axes[-1].set_xlabel("step number")
        plt.savefig('walkers.pdf')
        print('\nFile saved: walkers.pdf\n')

    ### FIND PARAMETERS CORRESPONDING TO THE MAXIMUM OF THE POSTERIOR PROBABILITY AND SAVE THEM IN BayesLens_best.dat.
    ### THE LensTool FORMAT IS ADOPTED FOR THE OUTPUT FILES

    if b != -10:
        if b == 0:
            max_index, = np.where(log_prob_samples == max(log_prob_samples))

            comments = 'ln(Lhood)'
            for i, id in enumerate(translation_vector[:, 0]):
                comments += '\n' + id + '_' + translation_vector[i, 1]

            np.savetxt(dir + 'BayesLens_best.dat', chains_tot[max_index, :], fmt='%.4f', delimiter='\t',
                       header=comments)
            print('\nFile saved: BayesLens_best.dat\n')

        if b == -1:
            mode_vector = np.zeros(len(chains_tot[0, :]))
            bar = tqdm.tqdm(total=len(chains_tot[0, :]) - 1, desc='Mode vector', position=1)
            for i in np.arange(len(mode_vector) - 1):
                mode_vector[i + 1] = stats.mode(np.asarray(chains_tot[:, i + 1], dtype=float), axis=None)[0]
                bar.update(1)

            comments = 'ln(Lhood)'
            for i, id in enumerate(translation_vector[:, 0]):
                comments += '\n' + id + '_' + translation_vector[i, 1]

            np.savetxt(dir + 'BayesLens_modes.dat', [mode_vector], fmt='%.4f', delimiter='\t', header=comments)
            print('\nFile saved: BayesLens_modes.dat\n')

        if b == -2:
            median_vector = np.zeros(len(chains_tot[0, :]))
            bar = tqdm.tqdm(total=len(chains_tot[0, :]) - 1, desc='Median vector', position=1)
            for i in np.arange(len(median_vector) - 1):
                median_vector[i + 1] = np.percentile(np.asarray(chains_tot[:, i + 1], dtype=float), 50)
                bar.update(1)

            comments = 'ln(Lhood)'
            for i, id in enumerate(translation_vector[:, 0]):
                comments += '\n' + id + '_' + translation_vector[i, 1]

            np.savetxt(dir + 'BayesLens_medians.dat', [median_vector], fmt='%.4f', delimiter='\t', header=comments)
            print('\nFile saved: BayesLens_medians.dat\n')

        if b > 0:

            indexes = random.sample(range(len(chains_tot[:, 0])), b)
            comments = 'ln(Lhood)'

            for i, id in enumerate(translation_vector[:, 0]):
                comments += '\n' + id + '_' + translation_vector[i, 1]

            np.savetxt(dir + 'BayesLens_random_' + str(b) + '.dat', chains_tot[indexes, :], fmt='%.4f', delimiter='\t',
                       header=comments)
            print('\nFile saved:','BayesLens_random_' + str(b) + '.dat\n')

    if par[1] != -1:
        mask_vdg = (np.asarray(translation_vector[:, 0], dtype=float) >= 2) & (
                    np.asarray(translation_vector[:, 0], dtype=float) < 3)

        if par[1] != 0:
            custom_l = par[0]
            custom_h = par[1]

            mask_custom = np.zeros(len(mask_vdg), dtype=bool)
            mask_custom[int(custom_l):int(custom_h)] = 1

            comments_custom = 'ln(Lhood)'

            for i, (id, number) in enumerate(
                    zip(translation_vector[:, 0][mask_custom], translation_vector[:, 1][mask_custom])):
                comments_custom += '\n' + id + '_' + number
            np.savetxt(dir + 'BayesLens_chains_custom.dat', chains_tot[:, np.append(True, mask_custom)], fmt='%.6f',
                       delimiter='\t', header=comments_custom)
            print('\nFile saved: BayesLens_chains_custom.dat\n')


        else:

            mask_shear = (np.asarray(translation_vector[:, 0], dtype=float) < 0)
            mask_par = (np.asarray(translation_vector[:, 0], dtype=float) >= 0) & (
                        np.asarray(translation_vector[:, 0], dtype=float) < 1)
            mask_h = (np.asarray(translation_vector[:, 0], dtype=float) >= 1) & (
                        np.asarray(translation_vector[:, 0], dtype=float) < 2)
            mask_g = (np.asarray(translation_vector[:, 0], dtype=float) >= 3)

            try:
                comments_shear = 'ln(Lhood)'
                for i, (id, number) in enumerate(
                        zip(translation_vector[:, 0][mask_shear], translation_vector[:, 1][mask_shear])):
                    comments_shear += '\n' + id + '_' + number
                np.savetxt(dir + 'BayesLens_chains_shear.dat', chains_tot[:, np.append(True, mask_shear)], fmt='%.6f',
                           delimiter='\t', header=comments_shear)
                print('\nFile saved: BayesLens_chains_shear.dat\n')
            except:
                pass

            comments_par = 'ln(Lhood)'
            for i, (id, number) in enumerate(
                    zip(translation_vector[:, 0][mask_par], translation_vector[:, 1][mask_par])):
                comments_par += '\n' + id + '_' + number
            np.savetxt(dir + 'BayesLens_chains_par.dat', chains_tot[:, np.append(True, mask_par)], fmt='%.6f',
                       delimiter='\t', header=comments_par)
            print('\nFile saved: BayesLens_chains_par.dat\n')

            comments_h = 'ln(Lhood)'
            for i, (id, number) in enumerate(zip(translation_vector[:, 0][mask_h], translation_vector[:, 1][mask_h])):
                comments_h += '\n' + id + '_' + number
            np.savetxt(dir + 'BayesLens_chains_halos.dat', chains_tot[:, np.append(True, mask_h)], fmt='%.6f',
                       delimiter='\t', header=comments_h)
            print('\nFile saved: BayesLens_chains_halos.dat\n')

            comments_vdg = 'ln(Lhood)'
            for i, (id, number) in enumerate(
                    zip(translation_vector[:, 0][mask_vdg], translation_vector[:, 1][mask_vdg])):
                comments_vdg += '\n' + id + '_' + number
            np.savetxt(dir + 'BayesLens_chains_vdgalaxies.dat', chains_tot[:, np.append(True, mask_vdg)], fmt='%.6f',
                       delimiter='\t', header=comments_vdg)
            print('\nFile saved: BayesLens_chains_vdgalaxies.dat\n')

            comments_g = 'ln(Lhood)'
            for i, (id, number) in enumerate(zip(translation_vector[:, 0][mask_g], translation_vector[:, 1][mask_g])):
                comments_g += '\n' + id + '_' + number
            np.savetxt(dir + 'BayesLens_chains_galaxies.dat', chains_tot[:, np.append(True, mask_g)], fmt='%.6f',
                       delimiter='\t', header=comments_g)
            print('\nFile saved: BayesLens_chains_galaxies.dat\n')


def maps(theta, working_dir, translation_vector, priors_bounds, lenstool_vector, header, ramdisk, deprojection_matrix,
         translation_vector_ex, mag_ex, tot, index):
    """
    Generate LensTool maps from BayesLens results

    :param theta: array containing all parameters optimized by BayesLens
    :param working_dir: working directory
    :param translation_vector: see *BayesLens_parser
    :param lenstool_vector: see *BayesLens_parser
    :param header: see *BayesLens_parser
    :param ramdisk: path to RAMDISK
    :param deprojection_matrix: see *BayesLens_parser
    :param translation_vector_ex: see *BayesLens_parser
    :param mag_ex: see *BayesLens_parser
    :param tot: number of parameters
    :param index: an index used in parallelization
    :return: Save LensTool maps (ampli, dpl, ecc.)
    """
    random_name = randomstring()
    if not os.path.exists(ramdisk + random_name):
        os.makedirs(ramdisk + random_name)
        path = ramdisk + random_name + '/'

        # CREATE LensTool INPUT FILE FROM SAMPLER PARAMETERS
        BayesLens_writer(out_path=path + 'map_', par_vector=theta, translation_vector=translation_vector,
                         lenstool_vector=lenstool_vector, header=header, priors_bounds=priors_bounds,
                         deprojection_matrix=deprojection_matrix, translation_vector_ex_w=translation_vector_ex,
                         mag_ex_w=mag_ex)

    else:
        print('ERROR: Two directory with the same name.')

    # RUN LensTool
    lenstool = Popen(['lenstool map_lenstool_in.par -n'], stdout=PIPE, stderr=PIPE, shell=True, cwd=path)
    lenstool.wait()

    try:
        file = glob.glob(path + '*.fits')
        if len(file) == 1:
            shutil.copyfile(file[0], working_dir + 'map_' + str(index) + '.fits')
        else:
            filex = glob.glob(path + '*x.fits')
            filey = glob.glob(path + '*y.fits')
            shutil.copyfile(filex[0], working_dir + 'map_x_' + str(index) + '.fits')
            shutil.copyfile(filey[0], working_dir + 'map_y_' + str(index) + '.fits')

    except:
        print('Warning: No map, ' + ramdisk + random_name)

    shutil.rmtree(ramdisk + random_name, ignore_errors=True)

    if index == 0:
        print('0%', end='', flush=True)
    if index == round(tot / 4):
        print(' ==> ', end='', flush=True)
    if index == round(tot / 2):
        print(' ==> 50%', end='', flush=True)
    if index == round(tot * 3 / 4):
        print(' ==> ', end='', flush=True)
    if index == round(tot - 1):
        print(' ==> 100%\n')

    del random_name, path, lenstool