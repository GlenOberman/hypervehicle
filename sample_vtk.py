# NOTE: This is currently only confirmed to work for Wedge. Problems known for Waverider, work ongoing to fix.

from hypervehicle.hangar import ParametricWedge

# This will be fed into the dvdp for the generator, which will handle the introduction of FloatWithSens values
parameters = {'wingspan': 1.0, 'chord': 1.0, 'thickness': 0.1}

wedge_generator = ParametricWedge()           # First create the generator
wedge_generator.dvdp(parameters)              # Then tell it you want sensitivities (temporary naming at the moment, to be similar to SensitivityStudy)
wedge = wedge_generator.create_instance()     # Only after you set dvdp should you create_instance?
wedge.generate()                              # Then generate
wedge.to_vtk("wedge",merge=True,binary=True)  # This should automate the process and produce the vtk files.
                                              # To get meshes without saving, you would use wedge_mesh = wedge.pv_mesh

# This is the VTK reader (note: if you request a .stl file, it will read it just as well)
reader = pv.get_reader("wedge-merged.vtk")
mesh = reader.read()

# This is the plotter for the mesh - if you were to use wedge_mesh=wedge.pv_mesh, change add_mesh(mesh... to add_mesh(wedge_mesh...
# if using scalars='dvd...', it appears to colour based on magnitude of dvd... vector.
p = pv.Plotter()
p.add_mesh(mesh, color='lightblue', opacity=0.5, scalars='dvdwingspan') # Naming convention for sensitivities is same as in SensitivityStudy, but with dv for vector rather than separate dx, dy, dz
p.show()