#!/usr/bin/env python
# ==========================================================
# Author: Pietro Bergamini - pietro.bergamini@phd.unipd.it
# ==========================================================

import numpy as np
from astropy.table import Table, Column

from util_func import deprojection_lenstool, scaling_func


def BayesLens_parser(par_file=None, dir=None):
    """
    Generates arrays (see below) from the BayesLens input file. This function is executed only once at the beginning of BayesLens run

    :param par_file: BayesLens input file
    :param dir: directory to store the deprojection matrix (see below). If deprojection matrix already exists in this directory it will be used
    :return: par_vectors: 1D array containing the N free parameter values of the lens model (scaling relation parameters (0-3), halo parameters, cluster member velocity dispersions)

             priors_bounds: (N X 4) array containing prior boundaries. The two columns [:,0] and [:,1] contain flat prior boundaries for each parameter (or the mean and std of the goussian prior only for optimized cluster members). Entry [1,2], [2,2] and [3,2] contain the reference mag of SR, the M/L slope and the reference core radius respectively. Column [:,2] and [:,3] contain CM magnitudes and aperture radii

             translation_vector: (N X 2) array. [:,0] contains the IDs of the parameters (0. = scaling relations, 1. = DM halos, 2. = measured galaxies, 3. = unmeasured galaxies). [:,0] contains parameter types

             lenstool_vector: (N X 7) string array used to create the LensTool input file

             header: string array containing the header of the LensTool input file

             image_file: string variable containing the name of multiple image file

             deprojection_matrix: (N, 100, 2) matrix containing deprojection coefficients (1/cp) corresponding to different rcut value (aperture radii and rcore are fixed)

             translation_vector_ex: same as *translation_vector but for galaxies that are not optimized

             mag_ex: magnitudes for galaxies that are not optimized
    """
    str_file = open(par_file, 'r').readlines()

    # THIS STRING WILL CONTAIN SEVERAL PARAMETERS OF CONFIGURATION USED IN LensTool INPUT FILE
    header = ''

    scaling_vector = np.zeros(4)
    j = 0

    # THIS MATRIX WILL CONTAIN: THE BOUNDARIES, THE MEANS AND STDs, USED FOR CREATE THE PRIORS; THE REFERENCE LUMINOSITY AND CORE RADIUS; AND THE LIST OF GALAXY MAGNITUDES
    priors_bounds = np.zeros((4, 4), dtype='float')

    ### READ INPUT FILE PARAMETERS
    for i, item in enumerate(str_file):
        if item == '\n' or "#" in item:
            j = j + 1
        if "REFERENCE" in item:
            ra_dec_ref = item.split(': ')[1].split(' ')
            ra_ref = float(ra_dec_ref[0])
            dec_ref = float(ra_dec_ref[1])
            header += 'runmode\n\treference 3 ' + item.split(': ')[1]
            header += '\tinverse 3 1 1\n\tend\n'
        elif "MULTFILE" in item:
            header += '\nimage\n\tmultfile 1 ' + item.split(': ')[1]
            image_file = item.split(': ')[1][:-1]
            header += '\tforme -10\n\tend\n'

            header += '\ncosmology\n'
            header += '\ngrille\n\tnlens 1000\n'
            header += '\tnlens_opt 0\n'
            header += '\tnombre 256\n\tend\n'
        elif "XMIN" in item:
            header += '\nchamp\n\txmin ' + item.split(': ')[1]
        elif "XMAX" in item:
            header += '\txmax ' + item.split(': ')[1]
        elif "YMIN" in item:
            header += '\tymin ' + item.split(': ')[1]
        elif "YMAX" in item:
            header += '\tymax ' + item.split(': ')[1] + '\tend\n'
        elif "Z" in item:
            z = item.split(': ')[1]
        elif "N_GAL" in item:
            n_gal = item.split(': ')[1]
        elif "M/L_SLOPE" in item:
            priors_bounds[2, 2] = float(item.split(': ')[1])
        elif "REF_RCORE" in item:
            priors_bounds[3, 2] = float(item.split(': ')[1])
        elif "VD_SLOPE_SC" in item:
            vdslope_v = item.split(': ')[1].split(',')
            scaling_vector[0] = np.asarray(vdslope_v[0], dtype='float')
            priors_bounds[0, 0] = float(vdslope_v[0]) - float(vdslope_v[1])
            priors_bounds[0, 1] = float(vdslope_v[0]) + float(vdslope_v[1])
        elif "VD_Q_SC" in item:
            vbdq_v = item.split(': ')[1].split(',')
            scaling_vector[1] = np.asarray(vbdq_v[0], dtype='float')
            priors_bounds[1, 0] = float(vbdq_v[0]) - float(vbdq_v[1])
            priors_bounds[1, 1] = float(vbdq_v[0]) + float(vbdq_v[1])
            priors_bounds[1, 2] = float(vbdq_v[2])
            reference_mag = float(vbdq_v[2])
        elif "VD_SCATTER_SC" in item:
            vdscatter_v = item.split(': ')[1].split(',')
            scaling_vector[2] = np.asarray(vdscatter_v[0], dtype='float')
            priors_bounds[2, 0] = float(vdscatter_v[0]) - float(vdscatter_v[1])
            priors_bounds[2, 1] = float(vdscatter_v[0]) + float(vdscatter_v[1])
        elif "CUT_Q_SC" in item:
            cutq_v = item.split(': ')[1].split(',')
            scaling_vector[3] = np.asarray(cutq_v[0], dtype='float')
            priors_bounds[3, 0] = float(cutq_v[0]) - float(cutq_v[1])
            priors_bounds[3, 1] = float(cutq_v[0]) + float(cutq_v[1])
        elif "HALOS" in item:
            halos_line = i - j + 1
        elif "SHEAR" in item:
            shear_line = i - j + 1
        elif "VD_GALAXIES" in item:
            vd_line = i - j + 1
        elif "GALAXIES" in item:
            gal_line = i - j + 1
        elif "END" in item:
            last_line = i - j + 1

    ### READ LensTool_plus_SP INPUT FILE PARAMETERS: SHEAR TABLE
    try:
        table_shear = Table.read(par_file, format='ascii', delimiter=' ', header_start=shear_line,
                                 data_start=shear_line + 1, data_end=vd_line)

        n_shear = vd_line - (shear_line + 1)

        index_shear = ['-1.'] * n_shear
        label_shear = np.zeros(n_shear)
        number_shear = table_shear['ID']

        for j, index in enumerate(index_shear):
            label_shear[j] = float(index + str(number_shear[j]))
        column_shear = Column(label_shear, name='label')
        table_shear.add_column(column_shear)
    except:
        shear_line = vd_line

    ### READ LensTool_plus_SP INPUT FILE PARAMETERS: HALOS TABLE

    table_halos = Table.read(par_file, format='ascii', delimiter=' ', header_start=halos_line,
                             data_start=halos_line + 1, data_end=shear_line)

    n_halos = shear_line - (halos_line + 1)

    index_halos = ['1.'] * n_halos
    label_halos = np.zeros(n_halos)
    number_halos = table_halos['ID']

    for j, index in enumerate(index_halos):
        label_halos[j] = float(index + str(number_halos[j]))
    column_halos = Column(label_halos, name='label')
    table_halos.add_column(column_halos)

    ### READ LensTool_plus_SP INPUT FILE PARAMETERS: GALAXIES WITH MASURED VELOCITY DISPERSION TABLE

    table_vdgalaxies = Table.read(par_file, format='ascii', delimiter=' ', header_start=vd_line,
                                  data_start=vd_line + 1, data_end=gal_line)

    n_vd_galaxies = gal_line - (vd_line + 1)

    index_vdgalaxies = ['2.'] * n_vd_galaxies
    label_vdgalaxies = np.zeros(n_vd_galaxies)
    number_vdgalaxies = table_vdgalaxies['ID']

    for j, index in enumerate(index_vdgalaxies):
        label_vdgalaxies[j] = float(index + str(number_vdgalaxies[j]))
    column_galaxies = Column(label_vdgalaxies, name='label')
    table_vdgalaxies.add_column(column_galaxies)

    ### READ LensTool_plus_SP INPUT FILE PARAMETERS: GALAXIES WITHOUT MASURED VELOCITY DISPERSION TABLE

    table_galaxies = Table.read(par_file, format='ascii', delimiter=' ', header_start=gal_line,
                                data_start=gal_line + 1,
                                data_end=last_line)

    n_galaxies = last_line - (gal_line + 1)

    index_galaxies = ['3.'] * n_galaxies
    label_galaxies = np.zeros(n_galaxies)
    number_galaxies = table_galaxies['ID']

    for j, index in enumerate(index_galaxies):
        label_galaxies[j] = float(index + str(number_galaxies[j]))
    column_galaxies = Column(label_galaxies, name='label')
    table_galaxies.add_column(column_galaxies)

    table_halos['RA_m'] = np.round(-(table_halos['RA_m'] - ra_ref) * np.cos(dec_ref / 180.0 * np.pi) * 3600.0, 4)
    table_halos['DEC_m'] = np.round((table_halos['DEC_m'] - dec_ref) * 3600.0, 4)

    table_vdgalaxies['RA'] = np.round(-(table_vdgalaxies['RA'] - ra_ref) * np.cos(dec_ref / 180.0 * np.pi) * 3600.0, 4)
    table_vdgalaxies['DEC'] = np.round((table_vdgalaxies['DEC'] - dec_ref) * 3600.0, 4)

    table_galaxies['RA'] = np.round(-(table_galaxies['RA'] - ra_ref) * np.cos(dec_ref / 180.0 * np.pi) * 3600.0, 4)
    table_galaxies['DEC'] = np.round((table_galaxies['DEC'] - dec_ref) * 3600.0, 4)

    # THIS ARRAY CONTAINS THE PARAMETERS TO SAMPLE
    par_vectors = scaling_vector

    # THIS MATRIX MAP par_vectors DESCRIBING THE TYPE OF PARAMETER FOR EACH ENTRY: 0.x REFER TO SCALING RELATION; 1.x TO HALOS; 2.x TO MEASURED GALAXIES; 3.x TO GALAXIES WITHOUT MEASURED VELOCITY
    translation_vector = np.array([['0.0', 'vd_slope'], ['0.1', 'vd_q'], ['0.2', 'vd_scatter'], ['0.3', 'cut_q']])

    # THIS MATRIX WILL BE USED TO CREATE LensTool INPUT FILE
    lenstool_vector = np.full((1, 7), '')

    names_in = ['RA', 'DEC', 'E', 'A', 'VD', 'Rc', 'RC']
    names_ls = ['x_centre', 'y_centre', 'ellipticite', 'angle_pos', 'v_disp', 'core_radius', 'cut_radius']

    for h in np.arange(len(table_halos)):

        lenstool_vector = np.vstack((lenstool_vector,
                                     [table_halos[h]['label'], '\npotentiel', '', '# ' + str(table_halos[h]['ID']), '',
                                      '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_halos[h]['label'], '\tprofil', '', table_halos[h]['PF'], '', '', '']))

        for p in np.arange(len(names_in)):
            if table_halos[h][names_in[p] + '_t'] == 1:
                par_vectors = np.append(par_vectors, table_halos[h][names_in[p] + '_m'])
                translation_vector = np.vstack((translation_vector, [table_halos[h]['label'], names_ls[p]]))

                priors_bounds = np.vstack((priors_bounds,
                                           [table_halos[h][names_in[p] + '_m'] - table_halos[h][names_in[p] + '_s'],
                                            table_halos[h][names_in[p] + '_m'] + table_halos[h][names_in[p] + '_s'],
                                            0.0, 0.0]))
            else:
                if table_halos[h][names_in[p] + '_t'] == 0:
                    lenstool_vector = np.vstack((lenstool_vector, [table_halos[h]['label'], '\t' + names_ls[p], '0',
                                                                   table_halos[h][names_in[p] + '_m'], '', '', '']))

        lenstool_vector = np.vstack(
            (lenstool_vector, [table_halos[h]['label'], '\tz_lens', '', str(float(z)), '', '', '']))

    try:
        names_in_shear = ['GAMMA', 'KAPPA', 'ANGLE']
        names_ls_shear = ['gamma', 'kappa', 'angle_pos']

        for h in np.arange(len(table_shear)):

            lenstool_vector = np.vstack((lenstool_vector,
                                         [table_shear[h]['label'], '\npotentiel', '', '# ' + str(table_shear[h]['ID']),
                                          '', '', '']))
            lenstool_vector = np.vstack((lenstool_vector, [table_shear[h]['label'], '\tprofil', '', '14', '', '', '']))

            for s in np.arange(len(names_in_shear)):
                if table_shear[h][names_in_shear[s] + '_t'] == 1:
                    par_vectors = np.append(par_vectors, table_shear[h][names_in_shear[s] + '_m'])
                    translation_vector = np.vstack((translation_vector, [table_shear[h]['label'], names_ls_shear[s]]))
                    priors_bounds = np.vstack((priors_bounds,
                                               [table_shear[h][names_in_shear[s] + '_m'] - table_shear[h][
                                                   names_in_shear[s] + '_s'],
                                                table_shear[h][names_in_shear[s] + '_m'] + table_shear[h][
                                                    names_in_shear[s] + '_s'],
                                                0.0, 0.0]))
                else:
                    if table_shear[h][names_in_shear[s] + '_t'] == 0:
                        lenstool_vector = np.vstack(
                            (lenstool_vector, [table_shear[h]['label'], '\t' + names_ls_shear[s], '0',
                                               table_shear[h][names_in_shear[s] + '_m'], '', '', '']))

            lenstool_vector = np.vstack(
                (lenstool_vector, [table_shear[h]['label'], '\tz_lens', '', str(float(z)), '', '', '']))
    except:
        pass

    for h in np.arange(len(table_vdgalaxies)):
        lenstool_vector = np.vstack((lenstool_vector, [table_vdgalaxies[h]['label'], '\npotentiel', '',
                                                       '# ' + str(table_vdgalaxies[h]['ID']), '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_vdgalaxies[h]['label'], '\tprofil', '', table_vdgalaxies[h]['PF'], '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_vdgalaxies[h]['label'], '\tx_centre', '', table_vdgalaxies[h]['RA'], '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_vdgalaxies[h]['label'], '\ty_centre', '', table_vdgalaxies[h]['DEC'], '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_vdgalaxies[h]['label'], '\tellipticite', '', '0', '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_vdgalaxies[h]['label'], '\tangle_pos', '', '0', '', '', '']))
        par_vectors = np.append(par_vectors, table_vdgalaxies[h]['VD'])
        translation_vector = np.vstack((translation_vector, [table_vdgalaxies[h]['label'], 'v_disp']))

        priors_bounds = np.vstack(
            (priors_bounds, [table_vdgalaxies[h]['VD'], table_vdgalaxies[h]['VD_err'], table_vdgalaxies[h]['MAG'],
                             table_vdgalaxies[h]['R']]))

        lenstool_vector = np.vstack(
            (lenstool_vector, [table_vdgalaxies[h]['label'], '\tz_lens', '', str(float(z)), '', '', '']))

    translation_vector_ex = np.zeros((1, 2), dtype=str)
    mag_ex = []
    if int(n_gal) == -1:
        n_gal = len(table_galaxies)

    for h in np.arange(len(table_galaxies)):
        lenstool_vector = np.vstack((lenstool_vector, [table_galaxies[h]['label'], '\npotentiel', '',
                                                       '# ' + str(table_galaxies[h]['ID']), '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_galaxies[h]['label'], '\tprofil', '', table_galaxies[h]['PF'], '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_galaxies[h]['label'], '\tx_centre', '', table_galaxies[h]['RA'], '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_galaxies[h]['label'], '\ty_centre', '', table_galaxies[h]['DEC'], '', '', '']))
        lenstool_vector = np.vstack(
            (lenstool_vector, [table_galaxies[h]['label'], '\tellipticite', '', '0', '', '', '']))
        lenstool_vector = np.vstack((lenstool_vector, [table_galaxies[h]['label'], '\tangle_pos', '', '0', '', '', '']))

        if h < int(n_gal):
            par_vectors = np.append(par_vectors,
                                    scaling_func(table_galaxies[h]['MAG'], scaling_vector[1], scaling_vector[0],
                                                 reference_mag))

            translation_vector = np.vstack((translation_vector, [table_galaxies[h]['label'], 'v_disp']))

            R_mean = np.mean(table_vdgalaxies['R'])

            priors_bounds = np.vstack((priors_bounds, [
                scaling_func(table_galaxies[h]['MAG'], scaling_vector[1] - 150, scaling_vector[0], reference_mag),
                scaling_func(table_galaxies[h]['MAG'], scaling_vector[1] + 150, scaling_vector[0], reference_mag),
                table_galaxies[h]['MAG'], R_mean]))

            lenstool_vector = np.vstack(
                (lenstool_vector, [table_galaxies[h]['label'], '\tz_lens', '', str(float(z)), '', '', '']))

        else:
            translation_vector_ex = np.vstack((translation_vector_ex, [table_galaxies[h]['label'], 'v_disp']))
            lenstool_vector = np.vstack(
                (lenstool_vector, [table_galaxies[h]['label'], '\tz_lens', '', str(float(z)), '', '', '']))
            mag_ex = np.append(mag_ex, table_galaxies[h]['MAG'])

    lenstool_vector = lenstool_vector[1:]
    translation_vector_ex = translation_vector_ex[1:]

    try:
        deprojection_matrix = np.load(dir + 'support/deprojection_matrix.npy')

        print('\n--------------------------------')
        print('\nDeprojection matrix from support...\n')

    except:
        # HERE WE CREATE A DEPROJECTION MATRIX TO TRANSLATE MEASURED VELOCITY DISPERSION TO LensTool FIDUCIAL VELOCITY DISPERSION
        # GIVEN GALAXIES APERTURES (IN THE INPUT FILE) AND DIFFERENT POSSIBILITIES OF TRUNCATION RADII (see paper).

        # r_core = np.zeros(len(par_vectors))
        #
        # mask_mem = (np.asarray(translation_vector[:,0], dtype=float) >= 2.)
        # r_core[mask_mem] = scaling_func(priors_bounds[:,2][mask_mem],priors_bounds[3][2],0.5,priors_bounds[1][2])
        #
        # deprojection_matrix_resolution = 2
        #
        # deprojection_matrix = np.zeros((len(r_core[mask_mem]),deprojection_matrix_resolution,2))

        r_core = np.zeros(len(np.append(par_vectors, mag_ex)))

        translation_vector_matrix = np.vstack((translation_vector, translation_vector_ex))

        mask_mem = (np.asarray(translation_vector_matrix[:, 0], dtype=float) >= 2.)
        r_core[mask_mem] = scaling_func(np.append(priors_bounds[:, 2], mag_ex)[mask_mem], priors_bounds[3][2], 0.5,
                                        priors_bounds[1][2])

        deprojection_matrix_resolution = 100

        deprojection_matrix = np.zeros((len(r_core[mask_mem]), deprojection_matrix_resolution, 2))

        R_values = np.append(priors_bounds[:, 3], np.full(len(mag_ex), R_mean))

        print('--------------------------------')
        print('\nPreparing deprojection matrix...\n')

        for i in np.arange(len(r_core[mask_mem])):
            if i == 0:
                print('0%', end='', flush=True)
            if i == round(len(r_core[mask_mem]) / 4):
                print(' ==> 25%', end='', flush=True)
            if i == round(len(r_core[mask_mem]) / 2):
                print(' ==> 50%', end='', flush=True)
            if i == round(len(r_core[mask_mem]) * 3 / 4):
                print(' ==> 75%', end='', flush=True)
            if i == round(len(r_core[mask_mem]) - 1):
                print(' ==> 100%\n')

            deprojection_matrix[i, :, 0] = np.linspace(r_core[mask_mem][i] * 10, priors_bounds[3][1],
                                                       deprojection_matrix_resolution)

            deprojection_matrix[i, :, 1] = deprojection_lenstool(
                R=np.full(deprojection_matrix_resolution, R_values[mask_mem][i]),
                r_core=np.full(deprojection_matrix_resolution, r_core[mask_mem][i]), r_cut=deprojection_matrix[i, :, 0])

        np.save(dir + 'support/deprojection_matrix.npy', deprojection_matrix)

    return par_vectors, priors_bounds, translation_vector, lenstool_vector, header, image_file, deprojection_matrix, translation_vector_ex, mag_ex