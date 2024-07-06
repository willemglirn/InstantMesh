import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from InstantMesh.src.utils.train_util import instantiate_from_config
from InstantMesh.src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from InstantMesh.src.utils.mesh_util import save_obj, save_glb, save_obj_with_mtl
from InstantMesh.src.utils.infer_util import remove_background, resize_foreground, save_video


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames

def inference(img_path, dest_path):
    ###############################################################################
    # Arguments.
    ###############################################################################
    args = {
        'config':'InstantMesh/configs/instant-mesh-large.yaml',
        'input_path':img_path,
        'output_path':dest_path,
        'model_save_format':'glb',
        'diffusion_steps':75,
        'seed':42,
        'scale':1.0,
        'distance':4.5,
        'view':6, # choices=[4, 6] Number of input views.
        'no_rembg':False,
        'export_texmap':False,
        'save_video':False,
    }
    media_files_made = {}
    seed_everything(args['seed'])

    ###############################################################################
    # Stage 0: Configuration.
    ###############################################################################

    config = OmegaConf.load(args['config'])
    config_name = os.path.basename(args['config']).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config

    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

    device = torch.device('cpu')

    # load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="./InstantMesh/zero123plus/",
        #torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    print('Loading post download ...')
    state_dict = torch.load(unet_ckpt_path, map_location=device)
    print('Setting state dict...')
    pipeline.unet.load_state_dict(state_dict, strict=False)

    print('Pipline to device ...')
    pipeline = pipeline.to(device)

    # load reconstruction model
    print('Loading reconstruction model ...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    # make output directories
    image_path = os.path.join(args['output_path'], config_name, 'images')
    mesh_path = os.path.join(args['output_path'], config_name, 'meshes')
    video_path = os.path.join(args['output_path'], config_name, 'videos')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # process input files
    if os.path.isdir(args['input_path']):
        input_files = [
            os.path.join(args['input_path'], file) 
            for file in os.listdir(args['input_path']) 
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
        ]
    else:
        input_files = [args['input_path']]
    print(f'Total number of input images: {len(input_files)}')


    ###############################################################################
    # Stage 1: Multiview generation.
    ###############################################################################

    rembg_session = None if args['no_rembg'] else rembg.new_session()

    outputs = []
    for idx, image_file in enumerate(input_files):
        name = os.path.basename(image_file).split('.')[0]
        print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

        # remove background optionally
        input_image = Image.open(image_file)
        if not args['no_rembg']:
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)
        
        # sampling
        output_image = pipeline(
            input_image, 
            num_inference_steps=args['diffusion_steps'], 
        ).images[0]

        output_image.save(os.path.join(image_path, f'{name}.png'))
        print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

        outputs.append({'name': name, 'images': images})

    # delete pipeline to save memory
    del pipeline

    ###############################################################################
    # Stage 2: Reconstruction.
    ###############################################################################

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args['scale']).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1

    for idx, sample in enumerate(outputs):
        name = sample['name']
        print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

        images = sample['images'].unsqueeze(0).to(device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        if args['view'] == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(device)
            images = images[:, indices]
            input_cameras = input_cameras[:, indices]

        with torch.no_grad():
            # get triplane
            planes = model.forward_planes(images, input_cameras)

            # get mesh
            mesh_path_idx = os.path.join(mesh_path, f'{name}.{args['model_save_format']}')

            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args['export_texmap'],
                **infer_config,
            )
            if args['export_texmap'] and (args['model_save_format'] == 'obj'):
                vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    vertices.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_path_idx,
                )
                media_files_made[mesh_path_idx] = 'OBJ'
            else:
                vertices, faces, vertex_colors = mesh_out
                if args['model_save_format'] == 'glb':
                    save_glb(vertices, faces, vertex_colors, mesh_path_idx)
                    media_files_made[mesh_path_idx] = 'GLB'
                else:
                    save_obj(vertices, faces, vertex_colors, mesh_path_idx)
                    media_files_made[mesh_path_idx] = 'OBJ'
            print(f"Mesh saved to {mesh_path_idx}")
    return media_files_made

