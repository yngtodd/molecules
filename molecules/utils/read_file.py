import tables 

def read_h5_contact_maps(filename): 
    data = tables.open_file(filename, 'r')
    contact_maps = data.root.contact_maps.read()
    data.close()
    return contact_maps

def read_h5_RMSD(filename):
    data = tables.open_file(filename, 'r')
    RMSD = data.root.RMSD.read()
    data.close()
    return RMSD
