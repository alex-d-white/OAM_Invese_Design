# OAM_Invese_Design

This repo contains sample code to inverse design waveguide based orbital angular momentum emitters using SPINS-B/goos. 

First, fork the respository https://github.com/stanfordnqp/spins-b

To run this code, we must first add the ability for SPINS to import custom overlaps.
To do this follow the instructions below before installing spins into your enviornment.


-------------------------------------------------------------------------------------

1. Add the following code to spins/invdes/problem_graph/optplan/schema_em.py

```
@optplan.register_node_type()
class ImportOverlap(optplan.EmOverlap):
    """Represents a imported overlap vector.

    Attributes:
        file_name: .mat file containing the overlap vector.
        center: the center coordinate of the overlap, allows for translation
            of the overlap to the specified center.
    """
    type = schema_utils.polymorphic_model_type("overlap.import_field_vector")
    file_name = types.StringType()
    center = optplan.vec3d()
```

---------------------------------------------------------------------------------

2. Add the following code to spins/invdes/problem_graph/creator_em.py:

```
@optplan.register_node(optplan.ImportOverlap)
class ImportOverlap:

    def __init__(self,
                 params: optplan.ImportOverlap,
                 work: workspace.Workspace = None) -> None:
        """Creates a new waveguide mode overlap.

        Args:
            params: Waveguide mode parameters.
            work: Unused.
        """
        self._params = params

    def __call__(self, simspace: SimulationSpace, wlen: float = None,
                 **kwargs) -> fdfd_tools.VecField:
        matpath = os.path.join(simspace._filepath, self._params.file_name)
        overlap = sio.loadmat(matpath)

        # Use reference_grid to get coords which the overlap fields are defined on.
        reference_grid = simspace(wlen).eps_bg
        overlap_grid = np.zeros(reference_grid.grids.shape, dtype=np.complex_)

        xyz = reference_grid.xyz
        dxyz = reference_grid.dxyz
        shifts = reference_grid.shifts

        overlap_comp = ["Ex", "Ey", "Ez"]
        overlap_center = self._params.center

        overlap_coords = [
            overlap["x"][0] + overlap_center[0],
            overlap["y"][0] + overlap_center[1],
            overlap["z"][0] + overlap_center[2]
        ]

        # The interpolation done below only works on three-dimensional grids with each dimension containing
        # more than a single grid point (i.e. no two-dimensional grids). Therefore, if a dimension has a
        # singleton grid point, we duplicate along that axis to create a pseudo-3D grid.
        coord_dims = np.array([
            overlap_coords[0].size, overlap_coords[1].size,
            overlap_coords[2].size
        ])
        singleton_dims = np.where(coord_dims == 1)[0]
        if not singleton_dims.size == 0:
            for axis in singleton_dims:
                # The dx from the SPINS simulation grid is borrowed for the replication.
                dx = dxyz[axis][0]
                coord = overlap_coords[axis][0]
                overlap_coords[axis] = np.insert(overlap_coords[axis], 0,
                                                 coord - dx / 2)
                overlap_coords[axis] = np.append(overlap_coords[axis],
                                                 coord + dx / 2)
                # Repeat the overlap fields along the extended axis
                for comp in overlap_comp:
                    overlap[comp] = np.repeat(overlap[comp],
                                              overlap_coords[axis].size, axis)

        for i in range(0, 3):

            # Interpolate the user-specified overlap fields for use on the simulation grids
            overlap_interp_function = RegularGridInterpolator(
                (overlap_coords[0], overlap_coords[1], overlap_coords[2]),
                overlap[overlap_comp[i]],
                bounds_error=False,
                fill_value=0.0)

            # Grid coordinates for each component of Electric field. Shifts due to Yee lattice offsets.
            # See documentation of ``Grid" class for more detailed explanation.
            xs = xyz[0] + dxyz[0] * shifts[i, 0]
            ys = xyz[1] + dxyz[1] * shifts[i, 1]
            zs = xyz[2] + dxyz[2] * shifts[i, 2]

            # Evaluate the interpolated overlap fields on simulationg rids
            eval_coord_grid = np.meshgrid(xs, ys, zs, indexing='ij')
            eval_coord_points = np.reshape(eval_coord_grid, (3, -1),
                                           order='C').T
            interp_overlap = overlap_interp_function(eval_coord_points)
            overlap_grid[i] = np.reshape(interp_overlap,
                                         (len(xs), len(ys), len(zs)),
                                         order='C')

        return overlap_grid
```

-------------------------------------------------------------------------------------

3. Add the following code to spins/goos_sim/maxwell/simulate.py

```
import scipy.io as sio

@goos.polymorphic_model()
class ImportOverlap(SimOutput):
    """Represents a waveguide mode.

    The waveguide is assumed to be axis-aligned.

    Attributes:
        center: Waveguide center.
        wavelength: Wavelength at which to evaluate overlap.
        extents: Width and height of waveguide mode region.
        normal: Normal direction of the waveguide. Note that this is also the
            mode propagation direction.
        mode_num: Mode number. The mode with largest propagation constant is
            mode 0, the mode with second largest propagation constant is mode 1,
            etc.
        power: The transmission power of the mode.
        normalize: If `True`, normalize the overlap by the square of the total
            power emitted at the `wavelength`.
    """
    type = goos.ModelNameType("overlap.waveguide_mode")
    center = goos.Vec3d()
    mode_num = goos.types.StringType()


@maxwell.register(ImportOverlap, output_type=goos.Function)
class ImportOverlapOverlapImpl(SimOutputImpl):

    def __init__(self, overlap: ImportOverlap) -> None:
        self._overlap = overlap
        self._wg_overlap = None

    def before_sim(self, sim: FdfdSimProp) -> None:
        # Calculate the eigenmode if we have not already.
        if self._wg_overlap is None:
            matpath = os.path.join(simspace._filepath, self._overlap.file_name)
            overlap = sio.loadmat(matpath)

            # Use reference_grid to get coords which the overlap fields are defined on.
            reference_grid = simspace(wlen).eps_bg
            overlap_grid = np.zeros(reference_grid.grids.shape, dtype=np.complex_)

            xyz = reference_grid.xyz
            dxyz = reference_grid.dxyz
            shifts = reference_grid.shifts

            overlap_comp = ["Ex", "Ey", "Ez"]
            overlap_center = self._overlap.center

            overlap_coords = [
                overlap["x"][0] + overlap_center[0],
                overlap["y"][0] + overlap_center[1],
                overlap["z"][0] + overlap_center[2]
            ]

            # The interpolation done below only works on three-dimensional grids with each dimension containing
            # more than a single grid point (i.e. no two-dimensional grids). Therefore, if a dimension has a
            # singleton grid point, we duplicate along that axis to create a pseudo-3D grid.
            coord_dims = np.array([
                overlap_coords[0].size, overlap_coords[1].size,
                overlap_coords[2].size
            ])
            singleton_dims = np.where(coord_dims == 1)[0]
            if not singleton_dims.size == 0:
                for axis in singleton_dims:
                    # The dx from the SPINS simulation grid is borrowed for the replication.
                    dx = dxyz[axis][0]
                    coord = overlap_coords[axis][0]
                    overlap_coords[axis] = np.insert(overlap_coords[axis], 0,
                                                     coord - dx / 2)
                    overlap_coords[axis] = np.append(overlap_coords[axis],
                                                     coord + dx / 2)
                    # Repeat the overlap fields along the extended axis
                    for comp in overlap_comp:
                        overlap[comp] = np.repeat(overlap[comp],
                                                  overlap_coords[axis].size, axis)

            for i in range(0, 3):

                # Interpolate the user-specified overlap fields for use on the simulation grids
                overlap_interp_function = RegularGridInterpolator(
                    (overlap_coords[0], overlap_coords[1], overlap_coords[2]),
                    overlap[overlap_comp[i]],
                    bounds_error=False,
                    fill_value=0.0)

                # Grid coordinates for each component of Electric field. Shifts due to Yee lattice offsets.
                # See documentation of ``Grid" class for more detailed explanation.
                xs = xyz[0] + dxyz[0] * shifts[i, 0]
                ys = xyz[1] + dxyz[1] * shifts[i, 1]
                zs = xyz[2] + dxyz[2] * shifts[i, 2]

                # Evaluate the interpolated overlap fields on simulationg rids
                eval_coord_grid = np.meshgrid(xs, ys, zs, indexing='ij')
                eval_coord_points = np.reshape(eval_coord_grid, (3, -1),
                                               order='C').T
                interp_overlap = overlap_interp_function(eval_coord_points)
                overlap_grid[i] = np.reshape(interp_overlap,
                                             (len(xs), len(ys), len(zs)),
                                             order='C')

            self._wg_overlap = overlap_grid

    def eval(self, sim: FdfdSimProp) -> goos.NumericFlow:
        return goos.NumericFlow(np.sum(self._wg_overlap * sim.fields))

    def before_adjoint_sim(self, adjoint_sim: FdfdSimProp,
                           grad_val: goos.NumericFlow.Grad) -> None:
        adjoint_sim.source += np.conj(grad_val.array_grad * self._wg_overlap)
```
-------------------------------------------------------------------------------------
