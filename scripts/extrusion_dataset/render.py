import math
import random
import time
from pathlib import Path
from typing import Iterable, Tuple, Union


import moderngl
import numpy as np
from pyrr import Matrix44
import PIL.Image
from tqdm import tqdm

from scripts.extrusion_dataset.mesh import TriangleMesh


def render_meshes(
        meshes: Iterable[TriangleMesh],
        size: int,
        multisamples: Union[str, int] = "max",
) -> PIL.Image:

    all_vertices = []
    all_normals = []
    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        all_normals.append(mesh.normals())

    all_vertices = np.concatenate(all_vertices).reshape(-1)
    all_normals = np.concatenate(all_normals).reshape(-1)

    ctx = moderngl.create_context(require=330, standalone=True)
    ctx.gc_mode = "context_gc"

    program = ctx.program(
        vertex_shader="""
            #version 330
            
            in vec3 in_pos;
            in vec3 in_normal;
            
            // out vec4 v_pos;
            out vec3 v_normal;
            
            void main() {
                gl_Position = vec4(in_pos * vec3(1, 1, -1), 1);
                v_normal = in_normal;
            }
        """,
        fragment_shader="""
            #version 330
            
            // in vec4 v_pos;
            in vec3 v_normal;
            out vec4 fragColor;
            uniform float u_quality;
              
            void main() {
                vec3 col = vec3(1, 1, 1);
                vec3 col_quality = (v_normal * .5 + .5);
                
                col = mix(col, col_quality, u_quality);
                
                fragColor = vec4(col, 1);
            }
        """,
    )

    vao = ctx.vertex_array(
        program,
        [
            (ctx.buffer(all_vertices), '3f', 'in_pos'),
            (ctx.buffer(all_normals), '3f', 'in_normal'),
        ],
    )
    if multisamples == "max":
        multisamples = ctx.max_samples

    fbo_multi = ctx.framebuffer(
        color_attachments=[ctx.renderbuffer((size, size), 4, samples=multisamples)],
        depth_attachment=ctx.depth_renderbuffer((size, size), samples=multisamples),
    )
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((size, size), 4)],
    )

    fbo_multi.use()
    ctx.enable(ctx.DEPTH_TEST)
    ctx.enable(ctx.CULL_FACE)

    images = []
    for quality in (0., 1.):
        ctx.clear()
        program["u_quality"] = quality
        vao.render(moderngl.TRIANGLES)

        ctx.copy_framebuffer(fbo, fbo_multi)

        data = fbo.read(components=3)
        image = PIL.Image.frombytes("RGB", fbo.size, data)
        image = image.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)
        images.append(image)

    return images


def get_light_map(normals: np.ndarray, light_pos: Tuple[float, float, float]):
    """
    Use a normal map and project a light on it.

    :param normals: np.ndarray of shape (3, H, W) and range [-1, 1]
    :param light_pos: tuple of three floats
    :return:
    """
    C, H, W = normals.shape

    image_pos = np.concatenate([
        np.mgrid[:H, :W] / np.array([[[H]], [[W]]]) * 2. - 1.,
        np.zeros((1, H, W)),
    ])

    image_to_light = np.array(light_pos).reshape(3, 1, 1) - image_pos
    light_norm = image_to_light / np.linalg.norm(image_to_light, axis=0, keepdims=True)

    dots = (light_norm * normals).sum(0)
    dots *= normals[2] > 0

    return dots.clip(0, 1)
