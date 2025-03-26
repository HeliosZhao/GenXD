import enum
import numpy as np
import scipy
from internal import stepfun
import torch
import ipdb
from typing import Optional, Tuple, List

def colmap_w2c_3x4_to_nerf(w2c_3x4):
    '''
    w2c 3x4, numpy
    '''
    w2c = pad_poses(w2c_3x4)
    c2w_3x4 = np.linalg.inv(w2c)[:, :3]
    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = c2w_3x4 @ np.diag([1, -1, -1, 1])
    return poses

def nerf_c2w_3x4_to_colmap(c2w_3x4):
    '''
    c2w 3x4, numpy
    '''
    # Switch from NeRF (right, up, back) to COLMAP (right, down, fwd) frame.
    c2w_3x4 = c2w_3x4 @ np.diag([1, -1, -1, 1])
    c2w = pad_poses(c2w_3x4)
    w2c_3x4 = np.linalg.inv(c2w)[:, :3]
    return w2c_3x4

def convert_to_ndc(origins,
                   directions,
                   pixtocam,
                   near: float = 1.):
    """Converts a set of rays to normalized device coordinates (NDC).

  Args:
    origins: ndarray(float32), [..., 3], world space ray origins.
    directions: ndarray(float32), [..., 3], world space ray directions.
    pixtocam: ndarray(float32), [3, 3], inverse intrinsic matrix.
    near: float, near plane along the negative z axis.

  Returns:
    origins_ndc: ndarray(float32), [..., 3].
    directions_ndc: ndarray(float32), [..., 3].

  This function assumes input rays should be mapped into the NDC space for a
  perspective projection pinhole camera, with identity extrinsic matrix (pose)
  and intrinsic parameters defined by inputs focal, width, and height.

  The near value specifies the near plane of the frustum, and the far plane is
  assumed to be infinity.

  The ray bundle for the identity pose camera will be remapped to parallel rays
  within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
  world space can be remapped as long as it has dz < 0 (ray direction has a
  negative z-coord); this allows us to share a common NDC space for "forward
  facing" scenes.

  Note that
      projection(origins + t * directions)
  will NOT be equal to
      origins_ndc + t * directions_ndc
  and that the directions_ndc are not unit length. Rather, directions_ndc is
  defined such that the valid near and far planes in NDC will be 0 and 1.

  See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
  """

    # Shift ray origins to near plane, such that oz = -near.
    # This makes the new near bound equal to 0.
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = np.moveaxis(directions, -1, 0)
    ox, oy, oz = np.moveaxis(origins, -1, 0)

    xmult = 1. / pixtocam[0, 2]  # Equal to -2. * focal / cx
    ymult = 1. / pixtocam[1, 2]  # Equal to -2. * focal / cy

    # Perspective projection into NDC for the t = 0 near points
    #     origins + 0 * directions
    origins_ndc = np.stack([xmult * ox / oz, ymult * oy / oz,
                            -np.ones_like(oz)], axis=-1)

    # Perspective projection into NDC for the t = infinity far points
    #     origins + infinity * directions
    infinity_ndc = np.stack([xmult * dx / dz, ymult * dy / dz,
                             np.ones_like(oz)],
                            axis=-1)

    # directions_ndc points from origins_ndc to infinity_ndc
    directions_ndc = infinity_ndc - origins_ndc

    return origins_ndc, directions_ndc


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses, diff_poses=None):
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    if diff_poses is not None and diff_poses.size > 0:
        diff_poses = transform @ pad_poses(diff_poses)
        poses = np.concatenate([poses, diff_poses], axis=0)
    return unpad_poses(poses), transform


def average_pose(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


# Constants for generate_spiral_path():
NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.


def generate_spiral_path(poses, 
                         bounds, 
                         n_frames=120, 
                         n_rots=3, 
                         zrate=.5,
                         perc=90,
                         shift_z: Optional[List[float] ] = None, 
                         multiscale: Optional[List[float] ] = None,
                         return_center=False,
                         shift_zy: Optional[List[float] ] = False):
    """Calculates a forward facing spiral path for rendering."""

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    # radii = np.percentile(np.abs(positions), 90, 0)
    # use the max instead of percentile
    # radii = np.max(np.abs(positions), 0)
    radii = np.percentile(np.abs(positions), perc, 0)
    radii = np.concatenate([radii, [1.]])

    # get the scale of x, y, z shift
    trans_low = np.percentile(positions, 10, axis=0)
    trans_high = np.percentile(positions, 90, axis=0)
    trans_scale = (trans_high - trans_low) / 2
    # ipdb.set_trace()
    # Generate poses for spiral path.
    # ipdb.set_trace()
    cam2world = average_pose(poses) # 3,4
    
    up = poses[:, :3, 1].mean(0) # 3
    theta = np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False) # N
    t = radii[None] * np.stack([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), np.ones_like(theta)], axis=-1) # N,4
    center_point = (cam2world @ np.zeros((4,1))).T 
    pos_trans = (cam2world @ t.T).T # N,3
    # ipdb.set_trace()
    if bounds is None:
        center = focus_point_fn(poses)
        lookat = center
    else:
        # Find a reasonable 'focus depth' for this dataset as a weighted average
        # of conservative near and far bounds in disparity space.
        near_bound = bounds.min() * NEAR_STRETCH
        far_bound = bounds.max() * FAR_STRETCH
        # All cameras will point towards the world space point (0, 0, -focal).
        focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))
        lookat = cam2world @ [0, 0, -focal, 1.] # 3


    new_poses = np.array([viewmatrix(p-lookat, up, p) for p in pos_trans])
    center_pose = np.array(viewmatrix(center_point[0]-lookat, up, center_point[0]))[None]
    pose_sets = []
    
    multiscale = [1] if multiscale is None else [1] + multiscale
    shift_z = [0] if shift_z is None else [0] + shift_z
    ## scale before add shift
    for ms in multiscale:
        for sz in shift_z:
            shift_value = np.zeros_like(new_poses)
            # ipdb.set_trace()
            # Apply shifts only to the translation vector (last column)
            if shift_zy:
                shift_value[:, :3, 3] = 0, sz * trans_scale[1], 0
            else:
                shift_value[:, :3, 3] = 0, 0, sz * trans_scale[2]
            cur_pos = new_poses.copy()
            cur_pos[:, :, 3] += shift_value[:, :, 3]  # Apply the shift
            cur_pos[:, :3, 3] *= ms
            pose_sets.append(cur_pos)
   
    if return_center:
       return np.concatenate(pose_sets, axis=0), center_pose
   
    return np.concatenate(pose_sets, axis=0)


def transform_poses_pca(poses, diff_poses=None):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    # diff pose
    if diff_poses is not None and diff_poses.size > 0:
        diff_poses_recentered = unpad_poses(transform @ pad_poses(diff_poses))
        
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        if diff_poses is not None and diff_poses.size > 0:
            diff_poses_recentered = np.diag(np.array([1, -1, -1])) @ diff_poses_recentered
            
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    if diff_poses is not None and diff_poses.size > 0:
        diff_poses_recentered[:, :3, 3] *= scale_factor
        poses_recentered = np.concatenate([poses_recentered, diff_poses_recentered], axis=0)

    return poses_recentered, transform


def generate_ellipse_path(poses, 
                          n_frames=120, 
                          const_speed=True, 
                          z_variation=0., 
                          z_phase=0.,
                          shift_z: Optional[List[float] ] = None, 
                          multiscale: Optional[List[float] ] = None,):

    """Generate an elliptical render path based on the given poses."""
    # ipdb.set_trace()
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)
    # print(f"Center: {center}, Offset: {offset}, Low: {low}, High: {high}, Z_low: {z_low}, Z_high: {z_high}")
    trans_scale = (z_high - z_low) / 2
    
    
    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        # NOTE here is changed, z should be interpolated between z_low and z_high
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
             z_low[2] + (z_high - z_low)[2] *
                           ( z_variation * np.cos(theta + 2 * np.pi * z_phase) * .5 + .5),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = stepfun.sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]
    # shift_value = np.random.uniform(-shift_scale_range, shift_scale_range, size=(positions.shape[0],3))
    # ipdb.set_trace()

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    
    new_poses = np.stack([viewmatrix(p - center, up, p) for p in positions])
    
    pose_sets = []
    
    multiscale = [1] if multiscale is None else [1] + multiscale
    shift_z = [0] if shift_z is None else [0] + shift_z
    ## scale before add shift
    for ms in multiscale:
        for sz in shift_z:
            shift_value = np.zeros_like(new_poses)
            # ipdb.set_trace()
            # Apply shifts only to the translation vector (last column)
            shift_value[:, :3, 3] = 0, 0, sz * trans_scale[2]
            # shift_value[:, :3, 3] = 0, sz * trans_scale[1], 0
            # shift_value[:, :3, 3] = sz * trans_scale[0], 0, 0
            cur_pos = new_poses.copy()
            cur_pos[:, :, 3] += shift_value[:, :, 3]  # Apply the shift
            cur_pos[:, :3, 3] *= ms
            pose_sets.append(cur_pos)
   
   
    return np.concatenate(pose_sets, axis=0)

    
    return new_poses


def generate_interpolated_path(poses, 
                               n_frames, 
                               spline_degree=5, 
                               smoothness=.03, 
                               rot_weight=.1, 
                               shift_x: Optional[List[float] ] = None,
                               shift_z: Optional[List[float] ] = None, 
                               multiscale: Optional[List[float] ] = None):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    n_frames: returned path will have n_frames total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.
    shift_scale: scale of random shift along x, y for each interpolated pose. for re10k
    multiscale: scale the trajectory by a factor

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    
  Example:
      render_poses = generate_interpolated_path(
        keyframes,
        n_interp=config.render_spline_n_interp,
        spline_degree=config.render_spline_degree,
        smoothness=config.render_spline_smoothness,
        rot_weight=.1)
  """
    
    # get the scale of x, y, z shift
    trans_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    trans_high = np.percentile((poses[:, :3, 3]), 90, axis=0)
    trans_scale = (trans_high - trans_low) / 2
    # trans_scale = np.ones(3)
    
    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        
        # lookat = lookat + shift_value
        # up = up + shift_value
        # shift_value = np.random.uniform(-shift_scale_range, shift_scale_range, size=pos.shape)
        # pos = pos + shift_value
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(pos, lookat, up):
        """Converts from (position, lookat, up) format to pose matrices."""
        # ipdb.set_trace()
        return np.array([viewmatrix(p - l, u - p, p) for (p, l, u) in zip(pos, lookat, up)])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_frames,
                        k=spline_degree,
                        s=smoothness)
    ## shift poses
    pos, lookat, up = new_points[:, 0], new_points[:, 1], new_points[:, 2]
    # shift_value = np.random.uniform(-shift_scale_range, shift_scale_range, size=pos.shape)
    # pos += shift_value
    new_poses = points_to_poses(pos, lookat, up)
    # ipdb.set_trace()
    pose_sets = []
    if multiscale is None:
        ## shift translation without changing look at and up
        shift_x = [0] if shift_x is None else [0] + shift_x
        shift_z = [0] if shift_z is None else [0] + shift_z
        
        for sx in shift_x:
            for sz in shift_z:
                shift_value = np.zeros_like(new_poses)
                # ipdb.set_trace()
                # Apply shifts only to the translation vector (last column)
                shift_value[:, :3, 3] = sx * trans_scale[0], 0, sz * trans_scale[2]
                cur_pos = new_poses.copy()
                cur_pos[:, :, 3] += shift_value[:, :, 3]  # Apply the shift
                pose_sets.append(cur_pos)

        return np.concatenate(pose_sets, axis=0)
            
    else:
        multiscale = [1] + multiscale 
        for ms in multiscale:
            scale_poses = new_poses.copy()
            scale_poses[:, :3, 3] *= ms
            pose_sets.append(scale_poses)
            
        return np.concatenate(pose_sets, axis=0)
        




def generate_interpolated_path_old(poses, 
                               n_frames, 
                               spline_degree=5, 
                               smoothness=.03, 
                               rot_weight=.1, 
                               shift_scale=0, 
                               multiscale: Optional[List[float] ] = None):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    n_frames: returned path will have n_frames total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.
    shift_scale: scale of random shift along x, y for each interpolated pose. for re10k
    multiscale: scale the trajectory by a factor

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    
  Example:
      render_poses = generate_interpolated_path(
        keyframes,
        n_interp=config.render_spline_n_interp,
        spline_degree=config.render_spline_degree,
        smoothness=config.render_spline_smoothness,
        rot_weight=.1)
  """
    
    # get the scale of x, y, z shift
    trans_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    trans_high = np.percentile((poses[:, :3, 3]), 90, axis=0)
    shift_scale_range = (trans_high - trans_low) * shift_scale # 3
    # shift along x,y
    shift_scale_range[2] = 0 # z shift should be 0

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        
        # lookat = lookat + shift_value
        # up = up + shift_value
        # shift_value = np.random.uniform(-shift_scale_range, shift_scale_range, size=pos.shape)
        # pos = pos + shift_value
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(pos, lookat, up):
        """Converts from (position, lookat, up) format to pose matrices."""
        # ipdb.set_trace()
        return np.array([viewmatrix(p - l, u - p, p) for (p, l, u) in zip(pos, lookat, up)])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_frames,
                        k=spline_degree,
                        s=smoothness)
    ## shift poses
    pos, lookat, up = new_points[:, 0], new_points[:, 1], new_points[:, 2]
    shift_value = np.random.uniform(-shift_scale_range, shift_scale_range, size=pos.shape)
    pos += shift_value
    new_poses = points_to_poses(pos, lookat, up)
    if multiscale is not None:
        pose_sets = [new_poses]
        for ms in multiscale:
            scale_poses = poses
            scale_poses[:, :3, 3] *= ms
            scale_points = poses_to_points(scale_poses, dist=rot_weight)
            scale_points = interp(scale_points,
                                  n_frames,
                                  k=spline_degree,
                                  s=smoothness)
            pos, lookat, up = scale_points[:, 0], scale_points[:, 1], scale_points[:, 2]
            shift_value = np.random.uniform(-shift_scale_range, shift_scale_range, size=pos.shape)
            pos += shift_value
            pose_sets.append(points_to_poses(pos, lookat, up))
        return np.concatenate(pose_sets, axis=0)
        
        
    return new_poses


def interpolate_1d(x, n_interp, spline_degree, smoothness):
    """Interpolate 1d signal x (by a factor of n_interp times)."""
    t = np.linspace(0, 1, len(x), endpoint=True)
    tck = scipy.interpolate.splrep(t, x, s=smoothness, k=spline_degree)
    n = n_interp * (len(x) - 1)
    u = np.linspace(0, 1, n, endpoint=False)
    return scipy.interpolate.splev(u, tck)

def intrinsic_matrix(fx, fy, cx, cy):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.],
    ])


def get_pixtocam(focal, width, height):
    """Inverse intrinsic matrix for a perfect pinhole camera."""
    camtopix = intrinsic_matrix(focal, focal, width * .5, height * .5)
    return np.linalg.inv(camtopix)


def pixel_coordinates(width, height):
    """Tuple of the x and y integer coordinates for a grid of pixels."""
    return np.meshgrid(np.arange(width), np.arange(height), indexing='xy')


def _compute_residual_and_jacobian(x, y, xd, yd,
                                   k1=0.0, k2=0.0, k3=0.0,
                                   k4=0.0, p1=0.0, p2=0.0, ):
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = (k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4)))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(xd, yd, k1=0, k2=0,
                                     k3=0, k4=0, p1=0,
                                     p2=0, eps=1e-9, max_iterations=10):
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = np.copy(xd)
    y = np.copy(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps, x_numerator / denominator,
            np.zeros_like(denominator))
        step_y = np.where(
            np.abs(denominator) > eps, y_numerator / denominator,
            np.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return x, y


class ProjectionType(enum.Enum):
    """Camera projection type (standard perspective pinhole or fisheye model)."""
    PERSPECTIVE = 'perspective'
    FISHEYE = 'fisheye'


def pixels_to_rays(pix_x_int, pix_y_int, pixtocams,
                   camtoworlds,
                   distortion_params=None,
                   pixtocam_ndc=None,
                   camtype=ProjectionType.PERSPECTIVE):
    """Calculates rays given pixel coordinates, intrinisics, and extrinsics.

  Given 2D pixel coordinates pix_x_int, pix_y_int for cameras with
  inverse intrinsics pixtocams and extrinsics camtoworlds (and optional
  distortion coefficients distortion_params and NDC space projection matrix
  pixtocam_ndc), computes the corresponding 3D camera rays.

  Vectorized over the leading dimensions of the first four arguments.

  Args:
    pix_x_int: int array, shape SH, x coordinates of image pixels.
    pix_y_int: int array, shape SH, y coordinates of image pixels.
    pixtocams: float array, broadcastable to SH + [3, 3], inverse intrinsics.
    camtoworlds: float array, broadcastable to SH + [3, 4], camera extrinsics.
    distortion_params: dict of floats, optional camera distortion parameters.
    pixtocam_ndc: float array, [3, 3], optional inverse intrinsics for NDC.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.

  Returns:
    origins: float array, shape SH + [3], ray origin points.
    directions: float array, shape SH + [3], ray direction vectors.
    viewdirs: float array, shape SH + [3], normalized ray direction vectors.
    radii: float array, shape SH + [1], ray differential radii.
    imageplane: float array, shape SH + [2], xy coordinates on the image plane.
      If the image plane is at world space distance 1 from the pinhole, then
      imageplane will be the xy coordinates of a pixel in that space (so the
      camera ray direction at the origin would be (x, y, -1) in OpenGL coords).
  """

    # Must add half pixel offset to shoot rays through pixel centers.
    def pix_to_dir(x, y):
        return np.stack([x + .5, y + .5, np.ones_like(x)], axis=-1)

    # We need the dx and dy rays to calculate ray radii for mip-NeRF cones.
    pixel_dirs_stacked = np.stack([
        pix_to_dir(pix_x_int, pix_y_int),
        pix_to_dir(pix_x_int + 1, pix_y_int),
        pix_to_dir(pix_x_int, pix_y_int + 1)
    ], axis=0)

    matmul = np.matmul
    mat_vec_mul = lambda A, b: matmul(A, b[..., None])[..., 0]

    # Apply inverse intrinsic matrices.
    camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

    if distortion_params is not None:
        # Correct for distortion.
        x, y = _radial_and_tangential_undistort(
            camera_dirs_stacked[..., 0],
            camera_dirs_stacked[..., 1],
            **distortion_params)
        camera_dirs_stacked = np.stack([x, y, np.ones_like(x)], -1)

    if camtype == ProjectionType.FISHEYE:
        theta = np.sqrt(np.sum(np.square(camera_dirs_stacked[..., :2]), axis=-1))
        theta = np.minimum(np.pi, theta)

        sin_theta_over_theta = np.sin(theta) / theta
        camera_dirs_stacked = np.stack([
            camera_dirs_stacked[..., 0] * sin_theta_over_theta,
            camera_dirs_stacked[..., 1] * sin_theta_over_theta,
            np.cos(theta),
        ], axis=-1)

    # Flip from OpenCV to OpenGL coordinate system.
    camera_dirs_stacked = matmul(camera_dirs_stacked,
                                 np.diag(np.array([1., -1., -1.])))

    # Extract 2D image plane (x, y) coordinates.
    imageplane = camera_dirs_stacked[0, ..., :2]

    # Apply camera rotation matrices.
    directions_stacked = mat_vec_mul(camtoworlds[..., :3, :3],
                                     camera_dirs_stacked)
    # Extract the offset rays.
    directions, dx, dy = directions_stacked

    origins = np.broadcast_to(camtoworlds[..., :3, -1], directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    if pixtocam_ndc is None:
        # Distance from each unit-norm direction vector to its neighbors.
        dx_norm = np.linalg.norm(dx - directions, axis=-1)
        dy_norm = np.linalg.norm(dy - directions, axis=-1)

    else:
        # Convert ray origins and directions into projective NDC space.
        origins_dx, _ = convert_to_ndc(origins, dx, pixtocam_ndc)
        origins_dy, _ = convert_to_ndc(origins, dy, pixtocam_ndc)
        origins, directions = convert_to_ndc(origins, directions, pixtocam_ndc)

        # In NDC space, we use the offset between origins instead of directions.
        dx_norm = np.linalg.norm(origins_dx - origins, axis=-1)
        dy_norm = np.linalg.norm(origins_dy - origins, axis=-1)

    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see the original mipnerf paper).
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / np.sqrt(12)
    return origins, directions, viewdirs, radii, imageplane


def cast_ray_batch(cameras, pixels, camtype):
    """Maps from input cameras and Pixel batch to output Ray batch.

  `cameras` is a Tuple of four sets of camera parameters.
    pixtocams: 1 or N stacked [3, 3] inverse intrinsic matrices.
    camtoworlds: 1 or N stacked [3, 4] extrinsic pose matrices.
    distortion_params: optional, dict[str, float] containing pinhole model
      distortion parameters.
    pixtocam_ndc: optional, [3, 3] inverse intrinsic matrix for mapping to NDC.

  Args:
    cameras: described above.
    pixels: integer pixel coordinates and camera indices, plus ray metadata.
      These fields can be an arbitrary batch shape.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.

  Returns:
    rays: Rays dataclass with computed 3D world space ray data.
  """
    pixtocams, camtoworlds, distortion_params, pixtocam_ndc = cameras

    # pixels.cam_idx has shape [..., 1], remove this hanging dimension.
    cam_idx = pixels['cam_idx'][..., 0]
    batch_index = lambda arr: arr if arr.ndim == 2 else arr[cam_idx]

    # Compute rays from pixel coordinates.
    origins, directions, viewdirs, radii, imageplane = pixels_to_rays(
        pixels['pix_x_int'],
        pixels['pix_y_int'],
        batch_index(pixtocams),
        batch_index(camtoworlds),
        distortion_params=distortion_params,
        pixtocam_ndc=pixtocam_ndc,
        camtype=camtype)

    # Create Rays data structure.
    return dict(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        imageplane=imageplane,
        lossmult=pixels.get('lossmult'),
        near=pixels.get('near'),
        far=pixels.get('far'),
        cam_idx=pixels.get('cam_idx'),
        exposure_idx=pixels.get('exposure_idx'),
        exposure_values=pixels.get('exposure_values'),
    )


def cast_pinhole_rays(camtoworld, height, width, focal, near, far):
    """Wrapper for generating a pinhole camera ray batch (w/o distortion)."""

    pix_x_int, pix_y_int = pixel_coordinates(width, height)
    pixtocam = get_pixtocam(focal, width, height)

    origins, directions, viewdirs, radii, imageplane = pixels_to_rays(pix_x_int, pix_y_int, pixtocam, camtoworld)

    broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.),
        'near': broadcast_scalar(near),
        'far': broadcast_scalar(far),
        'cam_idx': broadcast_scalar(0),
    }

    return dict(origins=origins,
                directions=directions,
                viewdirs=viewdirs,
                radii=radii,
                imageplane=imageplane,
                **ray_kwargs)


def cast_spherical_rays(camtoworld, height, width, near, far):
    """Generates a spherical camera ray batch."""

    theta_vals = np.linspace(0, 2 * np.pi, width + 1)
    phi_vals = np.linspace(0, np.pi, height + 1)
    theta, phi = np.meshgrid(theta_vals, phi_vals, indexing='xy')

    # Spherical coordinates in camera reference frame (y is up).
    directions = np.stack([
        -np.sin(phi) * np.sin(theta),
        np.cos(phi),
        np.sin(phi) * np.cos(theta),
    ], axis=-1)

    matmul = np.matmul
    directions = matmul(camtoworld[:3, :3], directions[..., None])[..., 0]

    dy = np.diff(directions[:, :-1], axis=0)
    dx = np.diff(directions[:-1, :], axis=1)
    directions = directions[:-1, :-1]
    viewdirs = directions

    origins = np.broadcast_to(camtoworld[:3, -1], directions.shape)

    dx_norm = np.linalg.norm(dx, axis=-1)
    dy_norm = np.linalg.norm(dy, axis=-1)
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / np.sqrt(12)

    imageplane = np.zeros_like(directions[..., :2])

    broadcast_scalar = lambda x: np.broadcast_to(x, radii.shape[:-1])[..., None]
    ray_kwargs = {
        'lossmult': broadcast_scalar(1.),
        'near': broadcast_scalar(near),
        'far': broadcast_scalar(far),
        'cam_idx': broadcast_scalar(0),
    }

    return dict(origins=origins,
                directions=directions,
                viewdirs=viewdirs,
                radii=radii,
                imageplane=imageplane,
                **ray_kwargs)
