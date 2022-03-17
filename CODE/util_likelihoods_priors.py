#!/usr/bin/env python
# ==========================================================
# Author: Pietro Bergamini - pietro.bergamini@phd.unipd.it
# ==========================================================


from gettext import translation
import numpy as np
import os
import shutil
from subprocess import Popen, PIPE


from util_func import randomstring
from util_lenstool import BayesLens_writer


def prior_creator(vector, priors_lowbounds, priors_highbounds):
    """
    Generates flat priors between *priors_lowbounds and *priors_highbounds for parameters in *vector

    :param vector: array containing parameters optimized within flat priors
    :param priors_lowbounds: array containing lower bound of flat priors
    :param priors_highbounds: array containing higher bound of flat priors
    :return: selection. selection = True if all *vector entries are within their flat prior. Otherwise selection = False
    """
    selection = True

    for i, entry in enumerate(vector):
        if entry > priors_lowbounds[i] and entry < priors_highbounds[i]:
            selection = selection * True
        else:
            selection = selection * False

    return selection


def ln_gauss(vector, mu, std):
    """
    Generates Gaussian priors with mean *mu and standard deviation *std for parameters in *vector

    :param vector: array containing parameters with Gaussian priors
    :param mu: array containing mean of Gaussian priors
    :param std: array containing standard deviation of Gaussian priors
    :return: array with log values of Gaussian priors
    """
    lg = -0.5 * (np.sum(np.log(2 * np.pi * std ** 2) + ((vector - mu) ** 2) / std ** 2))
    return lg


def priors_scaling_relations(theta, priors_bounds, translation_vector):
    """
    Flat priors on scaling relation parameters (see eq.9 Bergamini et al. 2020)

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: -inf if scaling relation parameters are outside their flat priors. Otherwise -ln(scatter). scatter = scatter of galaxies around fitted scaling relation
    """

    mask_sr = (
        (np.asarray(translation_vector[:,0], dtype=float) >= 0) & 
        (np.asarray(translation_vector[:,0], dtype=float) < 1)
    )
    selection_vd = prior_creator(theta[mask_sr], priors_bounds[mask_sr, 0], priors_bounds[mask_sr, 1])

    if selection_vd:
        lnprior_vdgalaxies = -np.log(theta[mask_sr][2])
    else:
        lnprior_vdgalaxies = -np.inf

    del mask_sr, selection_vd

    return lnprior_vdgalaxies


def lnlike_vdgalaxies(theta, priors_bounds, translation_vector):
    """
    Likelihood on sigma-mag scaling relation parameters using measured velocity dispersions (see eq.10 Bergamini et al. 2020)

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: summed log likelihood for measured galaxies
    """

    mask_sr = (
        (np.asarray(translation_vector[:,0], dtype=float) >= 0) & 
        (np.asarray(translation_vector[:,0], dtype=float) < 1)
    )
    vdslope, vdq, vdscatter = np.asarray(theta[mask_sr][0:3], dtype='float')

    mask_vdgalaxies = (np.asarray(translation_vector[:, 0], dtype=float) >= 2) & (
            np.asarray(translation_vector[:, 0], dtype=float) < 3) & (translation_vector[:, 1] == 'v_disp')

    mag = np.asarray(priors_bounds[:, 2][mask_vdgalaxies], dtype='float')
    sigma = np.asarray(priors_bounds[:, 0][mask_vdgalaxies], dtype='float')
    dsigma = np.asarray(priors_bounds[:, 1][mask_vdgalaxies], dtype='float')

    mag_ref = float(priors_bounds[mask_sr][1, 2])

    model = vdq * 10 ** ((vdslope / 2.5) * (mag_ref - mag))
    inc2 = (dsigma ** 2 + vdscatter ** 2)
    lnlike_vdgalaxies = -0.5 * (np.sum(np.log(2 * np.pi * inc2) + ((sigma - model) ** 2) / inc2))

    del mask_sr, vdslope, vdq, vdscatter, mask_vdgalaxies, mag, sigma, dsigma, mag_ref, model, inc2

    return lnlike_vdgalaxies


def prior_vdgalaxies(theta, priors_bounds, translation_vector):
    """
    Gaussian priors on measured galaxies velocity dispersions. Priors are centered on measured values and have a std = 5 * dsigma (see eq.12 Bergamini et al. 2020)

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: summed log value of gaussian priors
    """

    mask_vdgalaxies = (np.asarray(translation_vector[:, 0], dtype=float) >= 2) & (
                np.asarray(translation_vector[:, 0], dtype=float) < 3) & (translation_vector[:, 1] == 'v_disp')

    sigma_mea = np.asarray(theta[mask_vdgalaxies], dtype='float')
    sigma = np.asarray(priors_bounds[:, 0][mask_vdgalaxies], dtype='float')
    dsigma = np.asarray(priors_bounds[:, 1][mask_vdgalaxies], dtype='float')
    ln_prior_vdgalaxies = ln_gauss(sigma_mea, sigma, 5 * dsigma)

    del sigma_mea, sigma, dsigma, mask_vdgalaxies

    return ln_prior_vdgalaxies


def priors_galaxies(theta, priors_bounds, translation_vector):
    """
    Large flat priors, around the proposed sigma-mag scaling relation, for galaxies without a measure velocity dispersion. IT IS NOT USE IN THE CURRENT VERSION OF BayesLens

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: -inf if scaling relation parameters are outside their flat priors. Otherwise 0.
    """

    mask_galaxies_vd = (np.asarray(translation_vector[:, 0], dtype=float) >= 3) & (translation_vector[:, 1] == 'v_disp')

    sigmas = np.asarray(theta[mask_galaxies_vd], dtype='float')
    down_prior = np.asarray(priors_bounds[:, 0][mask_galaxies_vd], dtype='float')
    up_prior = np.asarray(priors_bounds[:, 1][mask_galaxies_vd], dtype='float')

    selection_gal = prior_creator(sigmas, down_prior, up_prior)

    if selection_gal:
        lnprior_gal = 0.
    else:
        lnprior_gal = -np.inf

    del mask_galaxies_vd, sigmas, selection_gal, down_prior, up_prior

    return lnprior_gal


def lnlike_galaxies(theta, priors_bounds, translation_vector):
    """
    Gaussian priors on unmeasured galaxies velocity dispersions. Priors are centered on the proposed sigma-mag scaling relation and have std = scatter (see eq.13 Bergamini et al. 2020)

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: summed log value of gaussian priors
    """

    mask_sr = (
        (np.asarray(translation_vector[:,0], dtype=float) >= 0) & 
        (np.asarray(translation_vector[:,0], dtype=float) < 1)
    )
    vdslope, vdq, vdscatter = theta[mask_sr][0:3]

    mask_galaxies_vd = (np.asarray(translation_vector[:, 0], dtype=float) >= 3) & (translation_vector[:, 1] == 'v_disp')

    sigmas = np.asarray(theta[mask_galaxies_vd], dtype='float')
    mags = np.asarray(priors_bounds[:, 2][mask_galaxies_vd], dtype='float')
    mag_ref_vd = float(priors_bounds[mask_sr][1, 2])

    vdmodel = vdq * (10.0 ** (0.4 * vdslope * (mag_ref_vd - mags)))
    vdinc2s = (vdscatter ** 2)
    lnlike_vdscaling = -0.5 * (np.sum(np.log(2 * np.pi * vdinc2s) + ((sigmas - vdmodel) ** 2) / vdinc2s))

    del mask_sr, vdslope, vdq, vdscatter, mask_galaxies_vd, sigmas, mags, mag_ref_vd

    return lnlike_vdscaling


def priors_halos(theta, priors_bounds, translation_vector):
    """
    Flat priors on DM halo parameters (see eq.14 Bergamini et al. 2020)


    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: -inf if DM halo parameters are outside their flat priors. Otherwise 0
    """

    mask_halos = ((np.asarray(translation_vector[:, 0], dtype=float) >= 1) & (
                np.asarray(translation_vector[:, 0], dtype=float) < 2)) | (
                             np.asarray(translation_vector[:, 0], dtype=float) < 0)

    halos = np.asarray(theta[mask_halos], dtype='float')
    priors_lowbounds_halos = np.asarray(priors_bounds[:, 0][mask_halos], dtype='float')
    priors_highbounds_halos = np.asarray(priors_bounds[:, 1][mask_halos], dtype='float')

    selection_halos = prior_creator(halos, priors_lowbounds_halos, priors_highbounds_halos)

    if selection_halos:
        lnprior_halos = 0.
    else:
        lnprior_halos = -np.inf

    del mask_halos, halos, priors_lowbounds_halos, priors_highbounds_halos, selection_halos

    return lnprior_halos


def lnlike_halos(theta, working_dir, translation_vector, priors_bounds, lenstool_vector, header, image_file, ramdisk,
                 deprojection_matrix, translation_vector_ex, mag_ex):
    """
    This function silently call LensTool to compute the likelihood of multiple images (see eq.15 Bergamini et al. 2020)

    :param theta: array containing all parameters optimized by BayesLens
    :param working_dir: working directory
    :param translation_vector: see *BayesLens_parser
    :param lenstool_vector: see *BayesLens_parser
    :param header: see *BayesLens_parser
    :param image_file: see *BayesLens_parser
    :param ramdisk: path to RAMDISK
    :param deprojection_matrix: see *BayesLens_parser
    :param translation_vector_ex: see *BayesLens_parser
    :param mag_ex: see *BayesLens_parser
    :return: likelihood of multiple image positions
    """
    # AVOID THE POSSIBILITY OF NEGATIVE VELOCITY DISPERSIONS
    mask_negatives = ((translation_vector[:, 1] == 'kappa') | (translation_vector[:, 1] == 'gamma') | (
                translation_vector[:, 1] == 'cut_radius') | (translation_vector[:, 1] == 'core_radius') | (
                                  translation_vector[:, 1] == 'ellipticite') | (
                                  translation_vector[:, 1] == 'v_disp') | (
                                  (np.asarray(translation_vector[:, 0], dtype=float) < 1) & (
                                      np.asarray(translation_vector[:, 0], dtype=float) >= 0))) & (theta < 0)

    if len(theta[mask_negatives]) > 0:
        lnLH = -np.inf
        random_name = ''
        path = ''
        chires = ''
        chires_file = ''
        lenstool = ''

    else:
        # CREATE A DIRECTORY IN THE RAM-DISK WITH A RANDOM NAME
        random_name = randomstring()
        if not os.path.exists(ramdisk + random_name):
            os.makedirs(ramdisk + random_name)
            path = ramdisk + random_name + '/'

            # COPY THE MULTIPLE IMAGES LensTool INPUT FILE IN THE RAM-DISK DIRECTORY
            shutil.copyfile(working_dir + image_file, path + image_file)

            # CREATE LensTool INPUT FILE FROM SAMPLER PARAMETERS
            BayesLens_writer(out_path=path, par_vector=theta, translation_vector=translation_vector,
                             lenstool_vector=lenstool_vector, header=header, priors_bounds=priors_bounds,
                             deprojection_matrix=deprojection_matrix, translation_vector_ex_w=translation_vector_ex,
                             mag_ex_w=mag_ex)
        else:
            print('ERROR: Two directory with the same name.')
            quit()

        # RUN LensTool
        lenstool = Popen(['bayesChires lenstool_in.par'], stdout=PIPE, stderr=PIPE, shell=True, cwd=path)
        lenstool.wait()

        try:
            chires_file = open(path + '/chires/chires0.dat', 'r')
            chires = chires_file.read()
            lnLH = float(chires.split('log(Likelihood)')[-1])
            chires_file.close()
        except:
            print('Warning: No chires, ' + ramdisk + random_name)
            lnLH = -np.inf
            chires = ''
            chires_file = ''

        shutil.rmtree(ramdisk + random_name, ignore_errors=True)

    del random_name, path, chires, chires_file, lenstool, mask_negatives

    return lnLH

def priors_cosmo(theta, priors_bounds, translation_vector):
    """
    Flat priors on cosmological parameters

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: -inf if cosmological parameters are outside their flat priors. Otherwise blah
    """

    selection_cosmo = prior_creator(theta, priors_bounds[:, 0], priors_bounds[:, 1])

    if selection_cosmo:
        value = np.sum(1 / (priors_bounds[:, 1] - priors_bounds[:, 0]))
        lnprior_cosmo = -np.log(value)
    else:
        lnprior_cosmo = -np.inf

    del selection_cosmo

    return lnprior_cosmo

def partial_posterior(theta, priors_bounds, translation_vector):
    """
    Compute flat priors on scaling relation parameters, Gaussian priors on measured galaxy velocity dispersion, priors for galaxies without a measured velocity dispersion and the likelihood on scaling relation parameters

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :return: sum of eq.9, 10, 12 and 13 of Bergamini et al. 2020
    """

    # SCALING RELATIONS
    priors_scaling = priors_scaling_relations(theta, priors_bounds, translation_vector)
    priors_vdgal = prior_vdgalaxies(theta, priors_bounds, translation_vector)

    prior_tot = priors_scaling + priors_vdgal

    if not np.isfinite(prior_tot):
        return -np.inf

    like_scaling = lnlike_vdgalaxies(theta, priors_bounds, translation_vector)
    like_galaxies = lnlike_galaxies(theta, priors_bounds, translation_vector)

    likelihood_tot = like_scaling + like_galaxies

    del priors_scaling, priors_vdgal, like_scaling, like_galaxies

    return prior_tot + likelihood_tot


def lnposterior(theta, priors_bounds, working_dir, translation_vector, lenstool_vector, header, image_file, ramdisk,
                deprojection_matrix, translation_vector_ex, mag_ex):
    """
    Compute flat priors on DM halo parameters, multiple image likelihood and sum the result to *partial_posterior

    :param theta: array containing all parameters optimized by BayesLens
    :param priors_bounds: see *BayesLens_parser
    :param working_dir: working directory
    :param translation_vector: see *BayesLens_parser
    :param lenstool_vector: see *BayesLens_parser
    :param header: see *BayesLens_parser
    :param image_file: see *BayesLens_parser
    :param ramdisk: path to RAMDISK
    :param deprojection_matrix: see *BayesLens_parser
    :param translation_vector_ex: see *BayesLens_parser
    :param mag_ex: see *BayesLens_parser
    :return: sum of eq.14, 15 of Bergamini et al. 2020 and the result from *partial_posterior
    """
    
    # COSMOLOGICAL PRIOR
    mask_free_par = (priors_bounds[:,0] != 0) | (priors_bounds[:,1] != 0)
    mask_cosmo = np.asarray(translation_vector[mask_free_par, 0], dtype=float) < 0
    if np.count_nonzero(mask_cosmo) != 0:
        priors_c = priors_cosmo(
            theta[mask_cosmo], priors_bounds[mask_free_par][mask_cosmo],
            translation_vector[mask_free_par][mask_cosmo]
        )
    else:
        priors_c = 0
    
    # HALOS PRIOR
    priors_h = priors_halos(theta, priors_bounds, translation_vector)

    if not np.isfinite(priors_h):
        return -np.inf

    part = partial_posterior(theta, priors_bounds, translation_vector)

    post_1 = part + priors_h + priors_c

    # LENSTOOL LIKELIHOOD
    if not np.isfinite(post_1):
        return -np.inf

    like_h = lnlike_halos(theta, working_dir, translation_vector, priors_bounds, lenstool_vector, header, image_file,
                          ramdisk, deprojection_matrix, translation_vector_ex, mag_ex)

    del priors_h, part

    return post_1 + like_h