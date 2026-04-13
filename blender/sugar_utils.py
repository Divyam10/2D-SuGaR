import json
import torch
import numpy as np
from sugar_scene.cameras import CamerasWrapper, GSCamera, focal2fov, fov2focal
from sugar_scene.sugar_model import SuGaR, load_refined_model, convert_refined_sugar_into_gaussians
import open3d as o3d
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    quaternion_invert,
    quaternion_multiply,
    Transform3d
)


def load_blender_package(package_path, device):
    # Load package
    package = json.load(open(package_path))
    # Convert lists into tensors
    for key, object in package.items():
        if type(object) is dict:
            for sub_key, sub_object in object.items():
                if type(sub_object) is list:
                    object[sub_key] = torch.tensor(sub_object)
        elif type(object) is list:
            for element in object:
                if element:
                    for sub_key, sub_object in element.items():
                        if type(sub_object) is list:
                            element[sub_key] = torch.tensor(sub_object)
                            
    # Process bones
    bone_to_vertices = []
    bone_to_vertex_weights = []
    for i_mesh, mesh_dict in enumerate(package['bones']):
        if mesh_dict:
            vertex_dict = mesh_dict['vertex']
            armature_dict = mesh_dict['armature']
            
            # Per vertex info
            vertex_dict['matrix_world'] = torch.Tensor(vertex_dict['matrix_world']).to(device)
            vertex_dict['tpose_points'] = torch.Tensor(vertex_dict['tpose_points']).to(device)
            # vertex_dict['groups'] = np.array(vertex_dict['groups'])
            # vertex_dict['weights'] = torch.tensor(vertex_dict['weights']).to(device)
            
            # Per bone info
            armature_dict['matrix_world'] = torch.Tensor(armature_dict['matrix_world']).to(device)
            for key, val in armature_dict['rest_bones'].items():
                armature_dict['rest_bones'][key] = torch.Tensor(val).to(device)
            for key, val in armature_dict['pose_bones'].items():
                armature_dict['pose_bones'][key] = torch.Tensor(val).to(device)
                
            # Build mapping from bone name to corresponding vertices
            vertex_groups_idx = {}
            vertex_groups_weights = {}
            
            # > For each bone of the current armature, we initialize an empty list
            for bone_name in armature_dict['rest_bones']:
                vertex_groups_idx[bone_name] = []
                vertex_groups_weights[bone_name] = []
                
            # > For each vertex, we add the vertex index to the corresponding bone lists
            for i in range(len(vertex_dict['groups'])):
                # groups_in_which_vertex_appears = vertex_dict['groups'][i]
                # weights_of_the_vertex_in_those_groups = vertex_dict['weights'][i]
                groups_in_which_vertex_appears = []
                weights_of_the_vertex_in_those_groups = []

                # We start by filtering out the groups that are not part of the current armature.
                # This is necessary for accurately normalizing the weights.
                for j_group, group in enumerate(vertex_dict['groups'][i]):
                    if group in vertex_groups_idx:
                        groups_in_which_vertex_appears.append(group)
                        weights_of_the_vertex_in_those_groups.append(vertex_dict['weights'][i][j_group])
                
                # We normalize the weights
                normalize_weights = True
                if normalize_weights:
                    sum_of_weights = np.sum(weights_of_the_vertex_in_those_groups)
                    weights_of_the_vertex_in_those_groups = [w / sum_of_weights for w in weights_of_the_vertex_in_those_groups]
                
                # We add the vertex index and the associated weight to the corresponding bone lists
                for j_group, group in enumerate(groups_in_which_vertex_appears):
                    # For safety, we check that the group belongs to the current armature, used for rendering.
                    # Indeed, for editing purposes, one might want to use multiple armatures in the Blender scene, 
                    # but only one (as expected) for the final rendering.
                    if group in vertex_groups_idx:
                        vertex_groups_idx[group].append(i)
                        vertex_groups_weights[group].append(weights_of_the_vertex_in_those_groups[j_group])

            # > Convert the lists to tensors
            for bone_name in vertex_groups_idx:
                if len(vertex_groups_idx[bone_name]) > 0:
                    vertex_groups_idx[bone_name] = torch.tensor(vertex_groups_idx[bone_name], dtype=torch.long, device=device)
                    vertex_groups_weights[bone_name] = torch.tensor(vertex_groups_weights[bone_name], device=device)

            bone_to_vertices.append(vertex_groups_idx)
            bone_to_vertex_weights.append(vertex_groups_weights)
        else:
            bone_to_vertices.append(None)
            bone_to_vertex_weights.append(None)
    package['bone_to_vertices'] = bone_to_vertices
    package['bone_to_vertex_weights'] = bone_to_vertex_weights
    
    return package


def load_sugar_models_from_blender_package(package, device):
    sugar_models = {}
    scene_paths = []

    for mesh in package['meshes']:
        scene_path = mesh['checkpoint_name']
        if not scene_path in scene_paths:
            scene_paths.append(scene_path)

    for i_scene, scene_path in enumerate(scene_paths):        
        print(f'\nLoading Gaussians to bind: {scene_path}')
        sugar_fine = load_refined_model(scene_path, nerfmodel=None, device=device)
        sugar_models[scene_path] = sugar_fine
        
    return sugar_models, scene_paths


def get_sugar_sh_rotations(sugar_comp):
    sh_rotations = quaternion_to_matrix(
        quaternion_multiply(
            sugar_comp.quaternions,
            quaternion_invert(sugar_comp.original_quaternions),
        )
    )
    sugar_comp.sh_rotations = sh_rotations
    return sh_rotations


def build_composite_scene(sugar_models, scene_paths, package):
    device = sugar_models[scene_paths[0]].device
    sh_levels = sugar_models[scene_paths[0]].sh_levels
    n_gaussians_per_surface_triangle = sugar_models[scene_paths[0]].n_gaussians_per_surface_triangle

    comp_scales = torch.zeros(0, 2, dtype=torch.float32, device=device)
    comp_complex = torch.zeros(0, 2, dtype=torch.float32, device=device)
    comp_opacities = torch.zeros(0, 1, dtype=torch.float32, device=device)
    comp_sh_dc = torch.zeros(0, 1, 3, dtype=torch.float32, device=device)
    comp_sh_rest = torch.zeros(0, 15, 3, dtype=torch.float32, device=device)
    
    original_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)

    mesh_list = []

    # Build the composited surface mesh
    with torch.no_grad():
        for i, mesh_dict in enumerate(package['meshes']):
            scene_name = mesh_dict['checkpoint_name']
            rc_fine = sugar_models[scene_name]
            
            # Full mesh from the original checkpoint        
            original_mesh = rc_fine.surface_mesh
            original_verts = original_mesh.verts_list()[0]
            original_faces = original_mesh.faces_list()[0]
            original_normals = original_mesh.faces_normals_list()[0]
            
            with torch.no_grad():
                all_faces_indices = torch.arange(0, rc_fine.surface_mesh.faces_list()[0].shape[0], dtype=torch.long, device=device)
                new_mesh = rc_fine.surface_mesh.submeshes([[all_faces_indices]])

            filtered_verts_to_verts_idx = - torch.ones(new_mesh.verts_list()[0].shape[0], dtype=torch.long, device=device)
            filtered_verts_to_verts_idx[new_mesh.faces_list()[0]] = rc_fine.surface_mesh.faces_list()[0]
            
            # Segmented reference mesh
            vert_idx = filtered_verts_to_verts_idx[mesh_dict['idx']]
            keep_verts_mask = torch.zeros(original_verts.shape[0], dtype=torch.bool, device=device)
            keep_verts_mask[vert_idx] = True
            keep_faces_mask = keep_verts_mask[original_faces].all(dim=1)
            keep_faces_indices = keep_faces_mask.nonzero()[..., 0]

            old_verts_to_new_verts_match = -torch.ones(original_verts.shape[0], dtype=torch.long, device=device)
            old_verts_to_new_verts_match[vert_idx] = torch.arange(vert_idx.shape[0], device=device)

            reference_verts = original_verts[vert_idx]
            reference_faces = original_faces[keep_faces_indices]
            reference_faces = old_verts_to_new_verts_match[reference_faces]
                    
            mesh_scales = rc_fine._scales.reshape(len(rc_fine._surface_mesh_faces), rc_fine.n_gaussians_per_surface_triangle, 2)[keep_faces_indices].view(-1, 2)
            mesh_quaternions = rc_fine._quaternions.reshape(len(rc_fine._surface_mesh_faces), rc_fine.n_gaussians_per_surface_triangle, 2)[keep_faces_indices].view(-1, 2)
            mesh_opacities = rc_fine.all_densities.reshape(len(rc_fine._surface_mesh_faces), rc_fine.n_gaussians_per_surface_triangle, 1)[keep_faces_indices].view(-1, 1)
            mesh_sh_dc = rc_fine._sh_coordinates_dc.reshape(len(rc_fine._surface_mesh_faces), rc_fine.n_gaussians_per_surface_triangle, 1, 3)[keep_faces_indices].view(-1, 1, 3)
            mesh_sh_rest = rc_fine._sh_coordinates_rest.reshape(len(rc_fine._surface_mesh_faces), rc_fine.n_gaussians_per_surface_triangle, 15, 3)[keep_faces_indices].view(-1, 15, 3)
            mesh_original_quaternions = rc_fine.quaternions.reshape(len(rc_fine._surface_mesh_faces), rc_fine.n_gaussians_per_surface_triangle, 4)[keep_faces_indices].view(-1, 4)
            
            comp_scales = torch.cat([comp_scales, mesh_scales], dim=0)
            comp_complex = torch.cat([comp_complex, mesh_quaternions], dim=0)
            comp_opacities = torch.cat([comp_opacities, mesh_opacities], dim=0)
            comp_sh_dc = torch.cat([comp_sh_dc, mesh_sh_dc], dim=0)
            comp_sh_rest = torch.cat([comp_sh_rest, mesh_sh_rest], dim=0)
            original_quaternions = torch.cat([original_quaternions, mesh_original_quaternions], dim=0)
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(reference_verts.cpu().numpy())
            o3d_mesh.triangles = o3d.utility.Vector3iVector(reference_faces.cpu().numpy())
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(reference_verts).cpu().numpy())
            
            mesh_list.append(o3d_mesh)
                
    full_mesh = mesh_list[0]
    for i in range(1, len(mesh_list)):
        full_mesh = full_mesh + mesh_list[i]

    # Build the composited SuGaR model
    sugar_comp = SuGaR(
        nerfmodel=None,
        points=None,
        colors=None,
        initialize=False,
        sh_levels=sh_levels,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=full_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        device=device,
    )

    # Update parameters
    with torch.no_grad():
        sugar_comp._scales[...] = comp_scales
        sugar_comp._quaternions[...] = comp_complex
        sugar_comp.all_densities[...] = comp_opacities
        sugar_comp._sh_coordinates_dc[...] = comp_sh_dc
        sugar_comp._sh_coordinates_rest[...] = comp_sh_rest
    sugar_comp.make_editable()  # TODO: make_editable should be called before modifying the vertices? Check.

    # Update mesh vertices
    start_idx = 0
    end_idx = 0
    ref_indices = []
    for i, mesh_dict in enumerate(package['meshes']):
        ref_indices.append(start_idx)
        edited_verts = mesh_dict['xyz']
        matrix_world = mesh_dict['matrix_world'].to(device).transpose(-1, -2)
        world_transform = Transform3d(matrix=matrix_world, device=device)
        end_idx += len(edited_verts)
        with torch.no_grad():
            sugar_comp._points[start_idx:end_idx] = world_transform.transform_points(edited_verts.to(device))
        start_idx = end_idx + 0
    ref_indices.append(len(sugar_comp._points))
    sugar_comp.ref_indices = ref_indices
    
    # Computing rotation matrices for fixing sh coordinates
    with torch.no_grad():
        sugar_comp.original_quaternions = original_quaternions
        sugar_comp.sh_rotations = get_sugar_sh_rotations(sugar_comp)
    return sugar_comp


def load_cameras_from_blender_package(package, device):
    matrix_world = package['camera']['matrix_world'].to(device)
    angle = package['camera']['angle']
    znear = package['camera']['clip_start']
    zfar = package['camera']['clip_end']
    
    if not 'image_height' in package['camera']:
        print('[WARNING] Image size not found in the package. Using default value 1920 x 1080.')
        height, width = 1080, 1920
    else:
        height, width = package['camera']['image_height'], package['camera']['image_width']

    gs_cameras = []
    for i_cam in range(len(angle)):
        c2w = matrix_world[i_cam]
        c2w[:3, 1:3] *= -1  # Blender to COLMAP convention
        w2c = c2w.inverse()
        R, T = w2c[:3, :3].transpose(-1, -2), w2c[:3, 3]  # R is stored transposed due to 'glm' in CUDA code
        
        fov = angle[i_cam].item()
        
        if width > height:
            fov_x = fov
            fov_y = focal2fov(fov2focal(fov_x, width), height)
        else:
            fov_y = fov
            fov_x = focal2fov(fov2focal(fov_y, height), width)
        
        gs_camera = GSCamera(
            colmap_id=str(i_cam), 
            R=R.cpu().numpy(), 
            T=T.cpu().numpy(), 
            FoVx=fov_x, 
            FoVy=fov_y,
            image=None, 
            gt_alpha_mask=None,
            image_name=f"frame_{i_cam}", 
            uid=i_cam,
            data_device=device,
            image_height=height,
            image_width=width,
        )
        gs_cameras.append(gs_camera)
    
    return CamerasWrapper(gs_cameras)


def _get_edited_points_deformation_mask(sugar:SuGaR, threshold:float=2.):
    _reference_verts = sugar._reference_points[sugar._surface_mesh_faces]
    ref_dists = (_reference_verts - _reference_verts.mean(dim=-2, keepdim=True)).norm(dim=-1)
    
    _new_verts = sugar._points[sugar._surface_mesh_faces]  # n_faces, 3, 3
    new_dists = (_new_verts - _new_verts.mean(dim=-2, keepdim=True)).norm(dim=-1)  # n_faces, 3
    
    ratios = (new_dists / ref_dists).max(dim=-1)[0]  # n_faces

    render_mask = ratios <= threshold
    render_mask = render_mask.unsqueeze(-1).repeat(1, sugar.n_gaussians_per_surface_triangle).reshape(-1)
    
    return render_mask


def apply_poses_to_scene(sugar_comp, i_frame, package, use_sh=True):
    n_frames = len(package['camera']['lens'])
    bone_to_vertices = package['bone_to_vertices']
    bone_to_vertex_weights = package['bone_to_vertex_weights']

    with torch.no_grad():
        for i_mesh, mesh_dict in enumerate(package['bones']):
            if mesh_dict:
                start_idx, end_idx = sugar_comp.ref_indices[i_mesh], sugar_comp.ref_indices[i_mesh+1]
                vertex_groups_idx = bone_to_vertices[i_mesh]
                vertex_groups_weights = bone_to_vertex_weights[i_mesh]
                
                tpose_points = mesh_dict['vertex']['tpose_points']

                use_weighting = True
                # TODO: Use weight formula for vertex with multiple groups. Just add the weighted transforms, and normalize at the end.
                if use_weighting:
                    new_points = torch.zeros_like(tpose_points)
                else:
                    new_points = tpose_points.clone().to(sugar_comp.device)
                            
                for vertex_group, vertex_group_idx in vertex_groups_idx.items():
                    if len(vertex_group_idx) > 0:
                        # Build bone transform
                        bone_transform = Transform3d(matrix=mesh_dict['armature']['pose_bones'][vertex_group][i_frame % n_frames].transpose(-1, -2))
                        reset_transform = Transform3d(matrix=mesh_dict['armature']['rest_bones'][vertex_group].transpose(-1, -2)).inverse()
                        
                        # Transform points
                        if use_weighting:
                            # weights = torch.tensor(vertex_groups_weights[vertex_group], device=frosting_comp.device)
                            weights = vertex_groups_weights[vertex_group]
                            new_points[vertex_group_idx] += weights[..., None] * bone_transform.transform_points(reset_transform.transform_points(tpose_points[vertex_group_idx]))
                        else:
                            new_points[vertex_group_idx] = bone_transform.transform_points(reset_transform.transform_points(tpose_points[vertex_group_idx]))

                sugar_comp.surface_mesh.verts_list()[0][start_idx:end_idx] = new_points

        # Rotate spherical harmonics
        if use_sh:
            sugar_comp.sh_rotations = get_sugar_sh_rotations(sugar_comp)
            
            
def render_composited_image(
    package:dict,
    sugar:SuGaR, 
    render_cameras:CamerasWrapper, 
    i_frame:int,
    sh_degree:int=None,
    deformation_threshold:float=2.,
    return_GS_model:bool=False,
    ):
    
    use_sh = (sh_degree is None) or sh_degree > 0
    
    # Change pose of the meshes
    apply_poses_to_scene(sugar, i_frame, package, use_sh)
    
    if sugar.editable:
        render_mask = _get_edited_points_deformation_mask(sugar, deformation_threshold)
        render_opacities = sugar.strengths.view(-1, 1)
        render_opacities[~render_mask] = 0.
    else:
        render_opacities = None
    
    # Render the scene
    if return_GS_model:
        return convert_refined_sugar_into_gaussians(sugar, opacities=render_opacities)
    else:
        rgb_render = sugar.render_image_gaussian_rasterizer(
            nerf_cameras=render_cameras, 
            camera_indices=i_frame,
            sh_deg=sh_degree,
            compute_color_in_rasterizer=not use_sh,
            sh_rotations=None if not use_sh else sugar.sh_rotations,
            point_opacities=render_opacities,
        ).nan_to_num().clamp(min=0, max=1)
        
        return rgb_render
