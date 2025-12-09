from pathlib import Path

import moderngl
import torch
import numpy as np
import PIL.Image



def render_shader(
        distance_image: torch.Tensor,
        multisamples: str|int = "max",
) -> PIL.Image:

    ctx = moderngl.create_context(require=330, standalone=True)
    ctx.gc_mode = "context_gc"

    program = ctx.program(
        vertex_shader="""
            #version 330
            
            in vec2 in_pos;
            
            out vec2 v_pos;
            out vec2 v_uv;
            
            void main() {
                gl_Position = vec4(in_pos, 0, 1);
                v_pos = in_pos;
                v_uv = in_pos * .5 + .5;
            }
        """,
        fragment_shader=(Path(__file__).resolve().parent / "fragment.glsl").read_text(),
    )

    vertices = np.array([
        -1, -1,     1, -1,  1, 1,
        -1, -1,     1, 1,   -1, 1,
    ])

    vao = ctx.vertex_array(
        program,
        [
            (ctx.buffer(vertices.astype("f4").tobytes()), "2f", "in_pos"),
        ],
    )
    if multisamples == "max":
        multisamples = ctx.max_samples

    size = (distance_image.shape[-1], distance_image.shape[-2])
    fbo_multi = ctx.framebuffer(
        color_attachments=[ctx.renderbuffer(size, 3, samples=multisamples)],
    )
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture(size, 3)],
    )

    fbo_multi.use()
    #ctx.enable(ctx.DEPTH_TEST)
    #ctx.enable(ctx.CULL_FACE)

    ctx.clear()
    try:
        program["u_texture"] = 0
    except KeyError:
        pass
    try:
        program["u_resolution"] = size
    except KeyError:
        pass

    data = distance_image.permute(1, 2, 0).numpy()[::-1]
    texture = ctx.texture(size=size, components=4, data=data.tobytes(), dtype="f4")
    sampler = ctx.sampler(texture=texture)
    sampler.filter = (ctx.LINEAR, ctx.LINEAR)
    sampler.use(0)
    vao.render(moderngl.TRIANGLES)

    ctx.copy_framebuffer(fbo, fbo_multi)

    data = fbo.read(components=3)
    image = PIL.Image.frombytes("RGB", fbo.size, data)
    image = image.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)

    ctx.release()

    return image

