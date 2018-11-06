import simtk.openmm.app as app
import simtk.openmm as omm
import simtk.unit as u 

import numpy as np 
import h5py 

from MDAnalysis.analysis import distances, contacts

class ContactMapReporter(object): 
    def __init__(self, file, reportInterval):
        self._file = h5py.File(file, 'w') 
        self._out = self._file.create_group('contact_maps')
        self._reportInterval = reportInterval 
    
    def __del__(self):
        self._file.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)
    
    def report(self, simulation, state):
        ca_indices = []
        for atom in simulation.topology.atoms(): 
            if atom.name == 'CA': 
                ca_indices.append(atom.index)
        positions = np.array(state.getPositions().value_in_unit(u.angstrom)) 
        time = int(np.round(state.getTime().value_in_unit(u.picosecond)))
        positions_ca = positions[ca_indices].astype(np.float32)
        distance_matrix = distances.self_distance_array(positions_ca) 
        contact_map = contacts.contact_matrix(distance_matrix, radius=8.0) * 1.0 
        self._out.create_dataset(str(time), data=contact_map)