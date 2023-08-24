from typing import Callable, List, Tuple, Optional, Union

import copy

import numpy as np
import numpy.random as rand

from spins import goos
from spins.goos import compat
from spins.goos import flows
from spins.goos_sim import maxwell
from spins.invdes import parametrization
from spins.invdes.problem_graph import optplan
from spins.invdes.problem_graph import workspace


def hermite_levelset_shape(
        initializer: Callable,
        extents: np.ndarray,
        pixel_spacing: float,
        control_point_spacing: float,
        pos: Union[np.ndarray, goos.Function],
        var_name: Optional[str] = None,
        reflection_symmetry: List[bool] = None,
        periods: List[int] = None,
        **kwargs,
) -> Tuple[goos.Variable, goos.Shape]:
    """Creates a new levelset parametrization with Hermite interpolation.

    The values of the parametrization are governed by Hermite interpolation
    on certain control points. Control points are defined with spacing
    given by `control_point_spacing`.

    Args:
        initializer: A callable to initialize values for the shape. This should
            accept a single argument `size` and return an array of values with
            shape `size`.
        extents: Extents of the shape.
        pixel_spacing: The pixel size will be given by
            `(pixel_spacing, pixel_spacing, extents[2])`.
        control_point_spacing: Spacing between two control points.
        var_name: Name to give the variable.
        pos: Position of the shape.
        **kwargs: Additional argument to pass to shape constructor.

    Returns:
        A tuple `(var, shape)` where `var` is the variable containing the values
        and `shape` is the newly created shape.
    """
    # TODO(vcruysse): Stop using the old parametrization implementation.
    from spins.goos import compat
    from spins.invdes.problem_graph import optplan

    if not isinstance(pos, goos.Function):
        pos = goos.Constant(pos)

    return compat.compat_param(
        param=optplan.HermiteLevelSetParametrization(
            undersample=control_point_spacing / pixel_spacing,
            reflection_symmetry=reflection_symmetry,
            periods=periods),
        initializer=initializer,
        extents=extents,
        pixel_size=[pixel_spacing, pixel_spacing, extents[2]],
        pos=pos,
        var_name=var_name,
        **kwargs)


def bicubic_levelset_shape(
        initializer: Callable,
        extents: np.ndarray,
        pixel_spacing: float,
        control_point_spacing: float,
        pos: Union[np.ndarray, goos.Function],
        var_name: Optional[str] = None,
        reflection_symmetry: List[bool] = None,
        periods: List[int] = None,
        **kwargs,
) -> Tuple[goos.Variable, goos.Shape]:
    """Creates a new levelset parametrization with bicubic interpolation.

    The values of the parametrization are governed by bicubic interpolation
    on certain control points. Control points are defined with spacing
    given by `control_point_spacing`.

    Args:
        initializer: A callable to initialize values for the shape. This should
            accept a single argument `size` and return an array of values with
            shape `size`.
        extents: Extents of the shape.
        pixel_spacing: The pixel size will be given by
            `(pixel_spacing, pixel_spacing, extents[2])`.
        control_point_spacing: Spacing between two control points.
        var_name: Name to give the variable.
        pos: Position of the shape.
        **kwargs: Additional argument to pass to shape constructor.

    Returns:
        A tuple `(var, shape)` where `var` is the variable containing the values
        and `shape` is the newly created shape.
    """
    # TODO(vcruysse): Stop using the old parametrization implementation.
    from spins.goos import compat
    from spins.invdes.problem_graph import optplan

    if not isinstance(pos, goos.Function):
        pos = goos.Constant(pos)

    return compat.compat_param(
        param=optplan.BicubicLevelSetParametrization(
            undersample=control_point_spacing / pixel_spacing,
            periods=periods,
            reflection_symmetry=reflection_symmetry),
        initializer=initializer,
        extents=extents,
        pixel_size=[pixel_spacing, pixel_spacing, extents[2]],
        pos=pos,
        var_name=var_name,
        **kwargs)


class CubicToHermiteLevelsetThresholding(goos.Action):
    node_type = "goos.compat.spins0_2_0.cubic_to_hermite_threshold"

    def __init__(self,
                 var_cont: goos.Variable,
                 var_disc: goos.Variable,
                 cont_shape: goos.Shape,
                 disc_shape: goos.Shape,
                 threshold: float = 0.5):
        super().__init__([var_cont, var_disc])

        self._cont_param = cont_shape.create_param()
        self._disc_param = disc_shape.create_param()

        self._var_cont = var_cont
        self._var_disc = var_disc
        self._thresh = threshold

    def run(self, plan: goos.OptimizationPlan):
        # TODO(vcruysse): [symmetry] I believe this only works for
        # no symmetry and no periodicity (logansu).
        self._cont_param.decode(
            plan.get_var_value(self._var_cont).flatten(order="F"))

        vec = self._cont_param.geometry_matrix @ self._cont_param.encode()
        p = (self._disc_param.reverse_geometry_matrix
             @ self._disc_param.derivative_matrix @ vec)
        p[:len(p) // 4] -= self._thresh

        self._disc_param.decode(p)
        plan.set_var_value(self._var_disc, self._disc_param.encode())


def cubic_to_hermite_levelset(*args,
                              **kwargs) -> CubicToHermiteLevelsetThresholding:
    thresh = CubicToHermiteLevelsetThresholding(*args, **kwargs)
    goos.get_default_plan().add_action(thresh)
    return thresh


class LevelsetFixBorder(goos.Action):
    node_type = "goos.compact.spins0_2_0.levelset_fix_border"

    def __init__(
        self,
        var: goos.Variable,
        shape: goos.Shape,
        num_border: List[int],
    ):
        super().__init__(var)

        self._var = var
        self._num_border = num_border

        self._disc_param = shape.create_param()

    def run(self, plan: goos.OptimizationPlan) -> None:
        value = plan.get_var_value(self._var)
        self._disc_param.decode(value.flatten(order="F"))
        self._disc_param.fix_borders(self._num_border[0], self._num_border[1],
                                     self._num_border[2], self._num_border[3])

        lower_bounds = self._disc_param.lower_bound
        lower_bounds = [x if x is not None else -np.inf for x in lower_bounds]
        lower_bounds = np.reshape(lower_bounds, value.shape, order="F")

        upper_bounds = self._disc_param.upper_bound
        upper_bounds = [x if x is not None else np.inf for x in upper_bounds]
        upper_bounds = np.reshape(upper_bounds, value.shape, order="F")

        plan.set_var_bounds(self._var, [lower_bounds, upper_bounds])


def levelset_fix_border(*args, **kwargs) -> LevelsetFixBorder:
    action = LevelsetFixBorder(*args, **kwargs)
    goos.get_default_plan().add_action(action)
    return action


class HermiteFabConstraint(goos.Function):
    node_type = "goos.compact.spins0_2_0.hermite_fab_constraint"

    def __init__(self, var: goos.Variable, min_gap: float,
                 min_curv_diameter: float, region: goos.Box3d,
                 pixel_spacing: float, control_point_spacing: float) -> None:
        super().__init__(var)

        self._d_gap = np.pi / (min_curv_diameter / pixel_spacing) / 1.2
        self._d_curv = np.pi / (min_gap / pixel_spacing) / 1.1

        from spins.invdes.problem_graph import optplan
        undersample = control_point_spacing / pixel_spacing
        pixel_size = [pixel_spacing, pixel_spacing, region.extents[2]]
        self._disc_param = compat.create_hermite_levelset_param(
            optplan.HermiteLevelSetParametrization(undersample=undersample),
            region.extents, pixel_size)[0]

    def eval(self, input_vals: List[goos.NumericFlow]) -> goos.NumericFlow:
        self._disc_param.decode(input_vals[0].array)
        curv = self._disc_param.calculate_curv_penalty(self._d_curv)
        gap = self._disc_param.calculate_gap_penalty(self._d_gap)
        return goos.NumericFlow(curv + gap)

    def grad(self, input_vals: List[goos.NumericFlow],
             grad_val: goos.NumericFlow.Grad) -> List[goos.NumericFlow.Grad]:
        self._disc_param.decode(input_vals[0].array)
        curv = self._disc_param.calculate_curv_penalty_gradient(self._d_curv)
        gap = self._disc_param.calculate_gap_penalty_gradient(self._d_gap)
        return [goos.NumericFlow.Grad((curv + gap) * grad_val.array_grad)]


def discretize_to_hermite_levelset(
    shape_cont: goos.Shape,
    render_region: goos.Box3d,
    device_region: goos.Box3d,
    pixel_spacing: float,
    control_point_spacing: float,
    material: goos.material.Material,
    material2: goos.material.Material,
    wavelength: float,
    min_feature: float = None,
    param_symmetry: List[bool] = None,
    periods: List[int] = None,
    simulation_symmetry: List[int] = None,
    fix_border: List[int] = None,
    var_name: str = None,
    stage_name: str = "discretize_to_hermite_levelset"):
    """Discretizes a permittivity distribution into a Hermite levelset.

    Args:
        shape_cont: Continuous permittivity to match.
        render_region: Region over which to match the permittivity distribution.
        device_region: Region to define the levelset parametrization.
        pixel_spacing: Pixel spacing of the Hermite levelset.
        control_point_spacing: Control point spacing of the Hermite levelset.
        min_feature: Minimum feature size of the Hermite levelset. Specifically,
            The minimum gap size becomes `min_feature` and the minimum curvature
            diameter becomes `min_feature * 100/120`.
        material: Background material of the discrete shape.
        material2: Foreground material of the discrete shape.
        wavelength: Wavelength to match the permittivity distribution.

    Returns:
        A tuple `(var_disc, shape_disc, obj_weighted_fabcon)` where `var_disc`
        is the variable for the discrete shape, `shape_disc` is the discrete
        shape, and `obj_weighted_fabcon` is the weighted fabrication constraint
        objective.
    """
    if simulation_symmetry is None:
        simulation_symmetry = [0, 0, 0]
    if fix_border is None:
        fix_border = [2, 2, 2, 2]

    # Step 1.
    # Create an intermediate continuous parametrization with the same grid
    # as the discrete parametrization.
    var_cont_fine, shape_cont_fine = goos.cubic_param_shape(
        initializer=np.random.random,
        pos=device_region.center,
        extents=device_region.extents,
        pixel_spacing=pixel_spacing,
        control_point_spacing=control_point_spacing,
        material=material,
        material2=material2,
        periods=periods,
        reflection_symmetry=param_symmetry)

    eps_cont = maxwell.RenderShape(shape_cont,
                                   region=render_region,
                                   mesh=maxwell.UniformMesh(dx=pixel_spacing),
                                   wavelength=wavelength)
    eps_cont_fine = maxwell.RenderShape(
        shape_cont_fine,
        region=render_region,
        mesh=maxwell.UniformMesh(dx=pixel_spacing),
        wavelength=wavelength)
    goos.opt.scipy_minimize(goos.Norm(eps_cont - eps_cont_fine),
                            "L-BFGS-B",
                            max_iters=20)

    var_disc, shape_disc = hermite_levelset_shape(
        initializer=np.zeros,
        pos=device_region.center,
        extents=device_region.extents,
        pixel_spacing=pixel_spacing,
        control_point_spacing=control_point_spacing,
        material=material,
        material2=material2,
        var_name=var_name,
        periods=periods,
        reflection_symmetry=param_symmetry)
    eps_disc = maxwell.RenderShape(shape_disc,
                                   region=render_region,
                                   mesh=maxwell.UniformMesh(dx=pixel_spacing),
                                   wavelength=wavelength)
    # Step 2.
    # Threshold the continuous structure as an initialization for discrete
    # parametrization.
    cubic_to_hermite_levelset(var_cont_fine, var_disc, shape_cont_fine,
                              shape_disc)

    # Step 3.
    # Fit the continuous parametrization to a discrete
    # parametrization.
    obj_cont_to_disc = goos.Norm(eps_disc - eps_cont)
    goos.opt.scipy_minimize(obj_cont_to_disc,
                            "L-BFGS-B",
                            monitor_list=[obj_cont_to_disc],
                            max_iters=20)

    # Step 4. Fix the borders.
    levelset_fix_border(var_disc, shape_disc, num_border=fix_border)

    # Step 5.
    # Set up fabrication constraint.
    fabcon_weight = goos.Variable(1, parameter=True)
    obj_fabcon = HermiteFabConstraint(
        var_disc,
        min_curv_diameter=min_feature / 120 * 100,
        min_gap=min_feature,
        region=device_region,
        pixel_spacing=pixel_spacing,
        control_point_spacing=control_point_spacing)
    obj_weighted_fabcon = fabcon_weight * obj_fabcon

    # Step 6.
    # Refit with fabrication constraints.
    fit_weight = goos.Variable(1, parameter=True)
    obj_weighted_eps_fit = fit_weight * obj_cont_to_disc

    fit_weight.set(1 / obj_cont_to_disc)
    fabcon_weight.set(1 / obj_fabcon)

    mu = goos.Variable(2, parameter=True)
    obj = obj_weighted_eps_fit + mu * obj_weighted_fabcon**2
    for i in range(5):
        ftol = max(1 / 100**(i + 1), 1e-9)
        goos.opt.scipy_minimize(obj,
                                "L-BFGS-B",
                                monitor_list=[obj],
                                max_iters=20,
                                ftol=ftol,
                                name="cont_to_disc_fit_cycle{}".format(i))
        mu.set(mu**1.5)

    return var_disc, shape_disc, obj_fabcon


def discretize_to_bicubic_levelset(
    shape_cont: goos.Shape,
    render_region: goos.Box3d,
    device_region: goos.Box3d,
    pixel_spacing: float,
    control_point_spacing: float,
    material: goos.material.Material,
    material2: goos.material.Material,
    wavelength: float,
    min_feature: float = None,
    param_symmetry: List[bool] = None,
    periods: List[int] = None,
    simulation_symmetry: List[int] = None,
    fix_border: List[int] = None,
    var_name: str = None,
    stage_name: str = "discretize_to_bicubic_levelset"):
    """Discretizes a permittivity distribution into a bicubic levelset.

    Args:
        shape_cont: Continuous permittivity to match.
        render_region: Region over which to match the permittivity distribution.
        device_region: Region to define the levelset parametrization.
        pixel_spacing: Pixel spacing of the Hermite levelset.
        control_point_spacing: Control point spacing of the Hermite levelset.
        min_feature: Minimum feature size of the Hermite levelset. Specifically,
            The minimum gap size becomes `min_feature` and the minimum curvature
            diameter becomes `min_feature * 100/120`.
        material: Background material of the discrete shape.
        material2: Foreground material of the discrete shape.
        wavelength: Wavelength to match the permittivity distribution.

    Returns:
        A tuple `(var_disc, shape_disc)` where `var_disc`
        is the variable for the discrete shape and `shape_disc` is the discrete
        shape.
    """
    if simulation_symmetry is None:
        simulation_symmetry = [0, 0, 0]
    if fix_border is None:
        fix_border = [2, 2, 2, 2]

    # Step 1.
    # Create an intermediate continuous parametrization with the same grid
    # as the discrete parametrization.
    var_cont_fine, shape_cont_fine = goos.cubic_param_shape(
        initializer=np.random.random,
        pos=device_region.center,
        extents=device_region.extents,
        pixel_spacing=pixel_spacing,
        control_point_spacing=control_point_spacing,
        material=material,
        material2=material2,
        periods=periods,
        reflection_symmetry=param_symmetry)

    eps_cont = maxwell.RenderShape(shape_cont,
                                   region=render_region,
                                   simulation_symmetry=simulation_symmetry,
                                   mesh=maxwell.UniformMesh(dx=pixel_spacing),
                                   wavelength=wavelength)
    eps_cont_fine = maxwell.RenderShape(
        shape_cont_fine,
        region=render_region,
        simulation_symmetry=simulation_symmetry,
        mesh=maxwell.UniformMesh(dx=pixel_spacing),
        wavelength=wavelength)
    goos.opt.scipy_minimize(goos.Norm(eps_cont - eps_cont_fine),
                            "L-BFGS-B",
                            max_iters=20,
                            name=stage_name)

    var_disc, shape_disc = bicubic_levelset_shape(
        initializer=np.zeros,
        pos=device_region.center,
        extents=device_region.extents,
        pixel_spacing=pixel_spacing,
        control_point_spacing=control_point_spacing,
        material=material,
        material2=material2,
        periods=periods,
        reflection_symmetry=param_symmetry,
        var_name=var_name)
    eps_disc = maxwell.RenderShape(shape_disc,
                                   region=render_region,
                                   simulation_symmetry=simulation_symmetry,
                                   mesh=maxwell.UniformMesh(dx=pixel_spacing),
                                   wavelength=wavelength)
    # Step 2.
    # Threshold the continuous structure as an initialization for discrete
    # parametrization.
    var_disc.set(var_cont_fine - 0.5)

    # Step 3.
    # Fit the continuous parametrization to a discrete
    # parametrization.
    obj_cont_to_disc = goos.Norm(eps_disc - eps_cont)**2
    goos.opt.scipy_minimize(obj_cont_to_disc,
                            "L-BFGS-B",
                            monitor_list=[obj_cont_to_disc],
                            max_iters=20)
    # Step 4. Fix the borders.
    levelset_fix_border(var_disc, shape_disc, num_border=fix_border)

    return var_disc, shape_disc
