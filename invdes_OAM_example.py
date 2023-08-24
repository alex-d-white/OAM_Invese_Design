'''
READ ME

To run this code, we must first add the ability for SPINS to import custom overlaps.
To do this follow the instructions below before installing spins into your VENV.


-------------------------------------------------------------------------------------

1. Add the following code to spins/invdes/problem_graph/optplan/schema_em.py

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

---------------------------------------------------------------------------------

2. Add the following code to spins/invdes/problem_graph/creator_em.py:

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

-------------------------------------------------------------------------------------

3. Add the following code to spins/goos_sim/maxwell/simulate.py

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

-------------------------------------------------------------------------------------

'''


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos_sim import maxwell
from spins.invdes.problem_graph import optplan
import compat_levelset.py


def main(save_folder: str,
         min_feature: float = 140,
         use_cubic: bool = False,
         sim_3d: bool = True,
         visualize: bool = False) -> None:
    goos.util.setup_logging(save_folder)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        wg_in = goos.Cuboid(pos=goos.Constant([-2500, 0, 0]),
                            extents=goos.Constant([4000, 1500, 220]),
                            material=goos.material.Material(index=3.45))

        def initializer(size):
            # Set the seed immediately before calling `random` to ensure
            # reproducibility.
            np.random.seed(247)
            return np.random.random(size) * 0.2 + 0.5

         #Continuous optimization.
        if use_cubic:
           var, design = goos.cubic_param_shape(
                 initializer=initializer,
                 pos=goos.Constant([0, 0, 0]),
                 extents=[3000, 3000, 220],
                 pixel_spacing=40,
                 control_point_spacing=1.5 * min_feature,
                 material=goos.material.Material(index=1),
                 material2=goos.material.Material(index=3.45),
                 var_name="var_cont")
        else:
           var, design = goos.pixelated_cont_shape(
                 initializer=initializer,
                 pos=goos.Constant([0, 0, 0]),
                 extents=[3000, 3000, 220],
                 material=goos.material.Material(index=1.444),
                 material2=goos.material.Material(index=3.45),
                 pixel_size=[40, 40, 220],
                 var_name="var_cont")
           var_top, design_top = goos.pixelated_cont_shape(
                 initializer=initializer,
                 pos=goos.Constant([0, 0, 110 + 250 + 200]),
                 extents=[3000, 3000, 400],
                 material=goos.material.Material(index=1.444),
                 material2=goos.material.Material(index=2.0),
                 pixel_size=[40, 40, 400],
                 var_name="var_top_cont")

        sigmoid_factor = goos.Variable(4, parameter=True, name="discr_factor")
        design = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design - 1)),
                           goos.Shape)
        design_top = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design_top - 1)),
                           goos.Shape)
        obj, sim = make_objective(goos.GroupShape([wg_in, design, design_top]),
                                  "cont",
                                  sim_3d=sim_3d)

        for factor in [4, 6, 8]:
            sigmoid_factor.set(factor)
            goos.opt.scipy_minimize(
                obj,
                "L-BFGS-B",
                monitor_list=[sim["eps"], sim["field"], sim["overlap"], obj],
                max_iters=20,
                name="opt_cont")

        if visualize:
            eps = maxwell.RenderShape(
                design,
                region=goos.Box3d(center=[0, 0, 0], extents=[4000, 4000, 0]),
                mesh=maxwell.UniformMesh(dx=40),
                wavelength=1550,
            )
            goos.util.visualize_eps(eps.get().array[2])

        # Run discretization.
        var.freeze()
        var_top.freeze()

        region = goos.Box3d(center=[0, 0, 0], extents=[4000, 4000, 0])
        var_disc, design_disc, obj_fabcon = spins.special.discretize_to_hermite_levelset(
            goos.GroupShape([wg_in, design]),
            render_region=region,
            device_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[3000 + 4 * min_feature, 3000 + 4 * min_feature, 220]),
            pixel_spacing=40,
            control_point_spacing=1.2 * min_feature,
            min_feature=min_feature,
            material=goos.material.Material(index=1.444),
            material2=goos.material.Material(index=3.45),
            wavelength=1550,
            var_name="var_disc",
        )

        regionTop = goos.Box3d(center=[0, 0, 110 + 250 + 200], extents=[4000, 4000, 0])
        var_top_disc, design_top_disc, obj_fabcon_top = spins.special.discretize_to_hermite_levelset(
            goos.GroupShape([wg_in, design_top]),
            render_region=regionTop,
            device_region=goos.Box3d(
                center=[0, 0, 110 + 250 + 200],
                extents=[3000 + 4 * min_feature, 3000 + 4 * min_feature, 400]),
            pixel_spacing=40,
            control_point_spacing=1.2 * min_feature,
            min_feature=300,
            material=goos.material.Material(index=1.444),
            material2=goos.material.Material(index=2),
            wavelength=1550,
            var_name="var_top_disc",
        )

        obj_fabcon = goos.rename(obj_fabcon, name="obj_fabcon")

        # Discrete optimization.
        obj_disc, sim_disc = make_objective(goos.GroupShape(
            [wg_in, design_disc, design_top_disc]),
                                            "disc",
                                            sim_3d=sim_3d)

        if visualize:
            eps = maxwell.RenderShape(
                design_disc,
                region=goos.Box3d(center=[0, 0, 0], extents=[4000, 4000, 0]),
                mesh=maxwell.UniformMesh(dx=40),
                wavelength=1550,
            )
            goos.util.visualize_eps(eps.get().array[2])

        em_weight = goos.Variable(1, parameter=True)
        em_weight.set(1 / goos.abs(obj_disc))
        fab_weight = goos.Variable(1, parameter=True)
        fab_weight.set(1 / obj_fabcon)
        mu = goos.Variable(2, parameter=True, name="mu")
        obj = goos.rename(em_weight * obj_disc + mu *
                          (fab_weight * obj_fabcon)**2,
                          name="obj_disc_weighted")
        for i in range(5):
            ftol = max(1 / 100**(i + 1), 1e-9)
            goos.log_print(mu)
            goos.opt.scipy_minimize(obj,
                                    method="L-BFGS-B",
                                    monitor_list=[
                                        sim_disc["eps"],
                                        sim_disc["field"],
                                        sim_disc["overlap"],
                                        obj,
                                        obj_disc,
                                        obj_fabcon,
                                    ],
                                    max_iters=20,
                                    ftol=ftol,
                                    name="disc_cycle{}".format(i))
            mu.set(mu**1.5)

        plan.save()
        plan.run()


def make_objective(eps: goos.Shape, stage: str, sim_3d: bool):
    if sim_3d:
        sim_z_extent = 3000
        solver = "maxwell_cg"
    else:
        sim_z_extent = 40
        solver = "local_direct"

    sim = maxwell.fdfd_simulation(
        name="sim_{}".format(stage),
        wavelength=1550,
        eps=eps,
        solver=solver,
        sources=[
            maxwell.WaveguideModeSource(center=[-1900, 0, 0],
                                        extents=[0, 2500, 1000],
                                        normal=[1, 0, 0],
                                        mode_num=0,
                                        power=1)
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=40),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[5000, 5000, sim_z_extent],
            ),
            pml_thickness=[400, 400, 400, 400, 400, 400]),
        background=goos.material.Material(index=1.444),
        outputs=[
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.ImportOverlap(name="overlap",
                                         file_name = "OAM1_GC_1p0.mat",
                                         center=[0, 0, 1000],
                                         ),
        ],
    )

    obj = goos.rename(-goos.abs(sim["overlap"]), name="obj_{}".format(stage))
    return obj, sim


def visualize(folder: str, step: int):
    """Visualizes result of the optimization.

    This is a quick visualization tool to plot the permittivity and electric
    field distribution at a particular save step. The function automatically
    determines whether the optimization is in continuous or discrete and
    plot the appropriate data.

    Args:
       folder: Save folder location.
       step: Save file step to load.
    """
    if step is None:
        step = goos.util.get_latest_log_step(folder)

    with open(os.path.join(folder, "step{0}.pkl".format(step)), "rb") as fp:
        data = pickle.load(fp)

    # Determine whether we have continuous epsilon or discrete.
    if "sim_cont.eps" in data["monitor_data"].keys():
        stage = "cont"
    else:
        stage = "disc"

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.abs(data["monitor_data"]["sim_{}.eps".format(stage)][1].squeeze()))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.abs(data["monitor_data"]["sim_{}.field".format(stage)][1].squeeze()))
    plt.colorbar()
    plt.show()

def gen_gds(folder: str, step: int):
    """Generates a GDS file of the device.

    Args:
        save_folder: Location where log files are saved. It is assumed that
            the optimization plan is also saved there.
        step: Optimization step of which to generate gds of (must be discrete)
    """
    if step < 0:
        step = goos.util.get_latest_log_step(folder)
    pkl_file = "step" + str(step) + ".pkl"
    # Param node that corresponds to HermiteLevelSet parameterization
    param_node = "goos.compat.spins0_2_0.shape.param.1"

    with goos.OptimizationPlan() as plan:
        plan.load(folder)
        plan.read_checkpoint(os.path.join(folder, pkl_file))
        dx = plan.get_node(param_node)._goos_schema.pixel_size[0]
        param = plan.get_node(param_node).create_param()
        var = plan.get_node("var_disc")
        param.from_vector(var.get().array.flatten(order="F"))
        polygons = param.generate_polygons(dx)

        spins.gds.gen_gds(polygons, os.path.join(folder, "output1.gds"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "view", "gen_gds"))
    parser.add_argument("save_folder")
    parser.add_argument("--step")

    args = parser.parse_args()
    if args.action == "run":
        main(args.save_folder, visualize=False)
    elif args.action == "view":
        visualize(args.save_folder, int(args.step))
    elif args.action == "gen_gds":
        gen_gds(args.save_folder, int(args.step))
