import os
import tables

import MDAnalysis as mda
from MDAnalysis import contacts, distances


def contact_maps_from_traj(xyz_file, traj_file, cm_table_file, contact_cutoff=8.0, savepath=None, filename=None):
    """
    Get contact map from trajectory.
    """
    if savepath and not filename:
        raise ValueError('Please specify a filename to save contact maps.')

    if filename and not savepath:
        raise ValueError('Please specify a path to save contact maps.')

    if savepath and filename:
        outpath = os.path.join(savepath, filename)
        outfile = tables.open_file(outpath, 'w')
        atom = tables.Float64Atom()
        cm_table = outfile.create_earray(outfile.root, 'contact_maps', atom, shape=(contact_size, 0))

    mda_traj = mda.Universe(xyz_file, traj_file)
    ca = mda_traj.select_atoms('name CA')
    num_ca = ca.n_atoms
    contact_size = int((num_ca - 1) / 2)

    contact_matrices = []
    for frame in mda_traj.trajectory:
        thresholded_matrx = contacts.contact_matrix(distances.self_distance_array(ca.positions), radius=contact_cutoff) * 1.0
        reshaped_matrix = thresholded_matrix.reshape(contact_size, 1)
        contact_matrices.append(reshaped_matrix)

        if savepath and filename:
            cm_table.append(reshaped_matrix)
            outfile.close()

    return contact_matrices
