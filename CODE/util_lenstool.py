#!/usr/bin/env python
# ==========================================================
# Author: Pietro Bergamini - pietro.bergamini@phd.unipd.it
# ==========================================================

import numpy as np
import copy

from util_func import find_nearest, scaling_func

def BayesLens_writer(out_path=None, par_vector=None, translation_vector=None, lenstool_vector=None, header=None,
                     priors_bounds=None, deprojection_matrix=None, translation_vector_ex_w=None, mag_ex_w=None):
    """
    Generates a LensTool input file from a BayesLens input file.

    :param out_path: directory where to save the LensTool input file
    :param par_vector: see *BayesLens_parser
    :param translation_vector: see *BayesLens_parser
    :param lenstool_vector: see *BayesLens_parser
    :param header: see *BayesLens_parser
    :param priors_bounds: see *BayesLens_parser
    :param deprojection_matrix: see *BayesLens_parser
    :param translation_vector_ex_w: see *BayesLens_parser
    :param mag_ex_w: see *BayesLens_parser
    :return: save a LensTool input file in *out_path
    """

    par_vector_ex_w = scaling_func(mag_ex_w, par_vector[1], par_vector[0], priors_bounds[1][2])

    par_vector = np.append(par_vector, par_vector_ex_w)

    translation_vector = np.vstack((translation_vector, translation_vector_ex_w))

    par_vector = np.round(par_vector, 4)

    r_core = np.zeros(len(par_vector))
    r_cut = np.zeros(len(par_vector))

    mask_mem = (np.asarray(translation_vector[:, 0], dtype=float) >= 2.)

    r_core[mask_mem] = np.round(
        scaling_func(np.append(priors_bounds[:, 2], mag_ex_w)[mask_mem], priors_bounds[3][2], 0.5, priors_bounds[1][2]),
        4)
    r_cut[mask_mem] = np.round(scaling_func(np.append(priors_bounds[:, 2], mag_ex_w)[mask_mem], float(par_vector[3]),
                                            priors_bounds[2][2] - 2 * float(par_vector[0]) + 1, priors_bounds[1][2]), 4)

    # HERE THE LensTool FIDUCIAL VELOCITY DISPERSION IS DERIVED FROM THE MEASURED VELOCITIES WITHIN AN APERTURE
    value = r_cut[mask_mem]
    index = find_nearest(deprojection_matrix[:, :, 0], value)
    d_matrix = deprojection_matrix[:, :, 1]
    vd_lenstool = np.round((par_vector[mask_mem] / d_matrix[np.arange(len(d_matrix[:, 0])), index]), 4)

    halos = ''

    translation_vector_writer = copy.copy(translation_vector)

    par_vector[mask_mem] = vd_lenstool

    par_vector = np.asarray([par_vector], dtype='str')
    translation_vector_writer[:, 1] = np.core.defchararray.add(
        np.full(translation_vector_writer.shape[0], '\t', dtype='str'), translation_vector_writer[:, 1])

    par_matrix = np.concatenate((translation_vector_writer, par_vector.T), axis=1)

    index_name_first, ind = np.unique(lenstool_vector[:, 0], return_index=True)
    index_name = index_name_first[np.argsort(ind)]

    for n, id in enumerate(index_name):
        lenstool_mask_id = lenstool_vector[:, 0] == str(id)
        lenstool_mask_type = (lenstool_vector[:, 2] == '1')
        lenstool_mask_limit = lenstool_mask_id * lenstool_mask_type
        par_mask_id = (par_matrix[:, 0] == str(id))

        lenstool_vector_masked = lenstool_vector[lenstool_mask_id]
        r_core_masked = r_core[par_mask_id]
        r_cut_masked = r_cut[par_mask_id]
        par_matrix_masked = par_matrix[par_mask_id]
        lenstool_vector_limit_masked = lenstool_vector[lenstool_mask_limit]

        for j in np.arange(lenstool_vector_masked.shape[0]):
            if j == 0:
                halos += lenstool_vector_masked[j][1] + ' ' + str(n + 1) + ' ' + lenstool_vector_masked[j][3] + '\n'
            else:
                if (lenstool_vector_masked[j][6] != '0.01'):
                    halos += lenstool_vector_masked[j][1] + '  ' + lenstool_vector_masked[j][3] + '\n'

        for k in np.arange(par_matrix_masked.shape[0]):
            halos += par_matrix_masked[k][1] + '  ' + par_matrix_masked[k][2] + '\n'
            if par_matrix_masked[k][1] == '\tv_disp' and float(par_matrix_masked[k][0]) >= 2:
                halos += '\tcore_radius ' + str(r_core_masked[k]) + '\n'
                halos += '\tcut_radius ' + str(r_cut_masked[k]) + '\n'
        halos += '\tend\n'

        if lenstool_vector_limit_masked.shape[0] != 0:
            halos += 'limit ' + str(n + 1) + '\n'
            for l in np.arange(lenstool_vector_limit_masked.shape[0]):
                if lenstool_vector_limit_masked[l][2] == '1':
                    halos += lenstool_vector_limit_masked[l][1] + '  ' + lenstool_vector_limit_masked[l][2] + '  ' + \
                             lenstool_vector_limit_masked[l][4] + '  ' + lenstool_vector_limit_masked[l][5] + '  ' + \
                             lenstool_vector_limit_masked[l][6] + '\n'
                else:
                    halos += lenstool_vector_limit_masked[l][1] + '  ' + lenstool_vector_limit_masked[l][2] + '  ' + \
                             lenstool_vector_limit_masked[l][3] + '  ' + lenstool_vector_limit_masked[l][4] + '  ' + \
                             lenstool_vector_limit_masked[l][6] + '\n'
            halos += '\tend\n'

    halos += '\nfini\n'

    file = open(out_path + 'lenstool_in.par', 'w')
    file.write(header + halos)
    file.close()

    mask_bayes = (np.asarray(lenstool_vector[:, 0], dtype=float) < 2)

    bayes_line = '#Nsample\n#ln(Lhood)\n#O' + str(
        len(np.unique(lenstool_vector[:, 0][mask_bayes])) + 1) + ' : sigma (km/s)\n#Chi2\n1 0 ' + str(
        vd_lenstool[0]) + ' 0\n'

    bayes_file = open(out_path + 'bayes.dat', "w")
    bayes_file.write(bayes_line)
    bayes_file.close()

    del translation_vector_writer, halos, index, index_name, index_name_first, r_core, r_cut, par_vector, mask_mem, translation_vector_ex_w, par_vector_ex_w, mag_ex_w  # mask_lenstool, mask_lens_vector
