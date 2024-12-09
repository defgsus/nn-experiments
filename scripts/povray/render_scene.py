import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import euclid


def create_script(
        seed: int,
        quality: bool = False,
        num_small: int = 300,
        num_large: int = 300,
):
    script = """
camera
{
    orthographic
    location <0, 0, -3>
    look_at <0, 0, 0>
    right x
}

#local L = .8;
light_source { <-6,4,-1> * 1000 rgb <.3, 1, 1> * .6 }
light_source { <-3,2,-3> * 1000 rgb L }
light_source { <-1,3,-5> * 1000 rgb L }
light_source { <-7,1,-10> * 1000 rgb L }
light_source { <4,1,-20> * 1000 rgb <.2, .7, 1> * .6 }
light_source { <2,-5,-3> * 1000 rgb <.5, 1, .2> * .5 }
light_source { <-2,-5,-2> * 1000 rgb <1, 1, .2> * .5 }

#declare TEXTURE = texture
{
	pigment { rgb 1. }
	finish { 
	    ambient 1 diffuse 0 brilliance 0
	    phong 0 
    }
}    

#declare TEXTURE_QUALITY = texture
{
	pigment { rgb .5 }
	/* normal { 
	    average 
	    normal_map { 
            #local c = 0;
            #while (c < 10)
	            [1. bumps .4 scale 0.001 translate c]
	            #local c = c + 1; 
	        #end
        } 
    } */
	finish {
		ambient .2 diffuse .1 brilliance 0.8
		phong .5 phong_size 10.
		specular .2 roughness 0.001
    }
}    
    """

    rng = random.Random(seed)
    rng2 = random.Random((seed + 117) * 331)
    objects = []

    def _cylinder(p1: euclid.Point2, p2: euclid.Point2, radius: float):
        objects.append(f"""
        union {{ 
            cylinder {{ <{p1.x}, {p1.y}, 0>, <{p2.x}, {p2.y}, 0>, {radius} open }}
            sphere {{ <{p1.x}, {p1.y}, 0>, {radius} }}
            sphere {{ <{p2.x}, {p2.y}, 0>, {radius} }}
        }}
""")

    def _stroke(p1: euclid.Point2, length: float, radius: float, bending: float = 1.):
        count = max(2, int(length * 200 * (1 + bending)))
        delta = euclid.Point3(rng.uniform(-1, 1), rng.uniform(-1, 1)) * length / count
        r = rng.uniform(-1, 1) / count * bending if rng.uniform(0, 1) < .5 else 0
        r2 = rng.uniform(0, bending) if rng.uniform(0, 1) < .7 else 0
        quat = euclid.Quaternion.new_rotate_axis(r, euclid.Vector3(0, 0, 1))
        last_p = p1
        base_radius = radius
        radius_freq = rng2.uniform(.01, 1) ** 2 * 20
        radius_offset = rng2.uniform(0, math.pi * 2)
        do_spikes = rng2.random() < .5
        spike_dist = rng2.uniform(.005, .010)
        for j in range(count):
            radius = max(0.001, (math.sin(j/count * radius_freq + radius_offset) * .3 + .7) ** 1.7 * base_radius)
            p2 = p1 + delta.xy
            delta = quat * delta
            quat.rotate_axis(rng.uniform(-1, 1) / count * r2, euclid.Vector3(0, 0, 1))
            _cylinder(p1, p2, radius)
            p1 = p2
            # spikes
            pd = (p1 - last_p)
            if do_spikes and pd.magnitude() > spike_dist:
                last_p = p1
                #if rng.random() < .2:
                if not quality:
                    objects.append(f"""
                        sphere {{ <{p1.x}, {p1.y}, 0>, {max(0.001, radius * 1.4)} }}
                    """)
                else:
                    pd.normalize()
                    for r in (-1, 1):
                        quat2 = quat.new_rotate_axis(math.pi / 2 * r, euclid.Vector3(0, 0, 1))
                        p3 = p1 + (quat2 * euclid.Vector3(pd.x, pd.y, 0) * (radius + .005)).xy
                        objects.append(f"""
                            cone {{ <{p1.x}, {p1.y}, 0>, {max(0.001, radius)}, <{p3.x}, {p3.y}, 0>, 0   }}
                        """)

    radius = 0.002 if quality else 0.001

    for i in range(num_large):
        p1 = euclid.Point2(rng.uniform(-1, 1), rng.uniform(-1, 1)) * .5
        r = radius * rng2.uniform(.5, 5)
        _stroke(p1, length=rng.uniform(.01, .5), bending=rng.uniform(0, 10), radius=r)

    for i in range(num_small):
        p1 = euclid.Point2(rng.uniform(-1, 1), rng.uniform(-1, 1)) * .5
        r = radius * rng2.uniform(.5, 2)
        _stroke(p1, length=rng.uniform(.02, .1), bending=rng.uniform(0, 1), radius=r)

    objects = "\n".join(objects)
    return script + f"union {{ {objects} texture {{ TEXTURE{'_QUALITY' if quality else ''} }} }}"


def render_scene(scene: str, filename: Path, width: int = 1024, height: int = 1024, aa: float = 0.001):
    with tempfile.TemporaryDirectory() as path:
        pov_filename = Path(path) / "scene.pov"
        pov_filename.write_text(scene)
        subprocess.check_call(
            ["povray", str(pov_filename), f"+W{width}", f"+H{height}", f"+a{aa}"]
        )
        image_filename = Path(path) / "scene.png"

        if image_filename.exists():
            if filename.exists():
                os.remove(filename)
            os.makedirs(filename.parent, exist_ok=True)
            shutil.move(str(image_filename), str(filename))


def render_test():
    output_path = Path(__file__).resolve().parent

    for q in (False, True):
        scene = create_script(23, quality=q)
        render_scene(scene, output_path / f"scene-{'high' if q else 'low'}.png")


def render_dataset():
    output_path = Path(__file__).resolve().parent.parent.parent / "datasets" / "shiny-tubes3"

    for split_path, seeds in (
            ("train", (23, 42, 66, 101)),
            ("validation", (7, )),
    ):
        idx = 0
        for seed in seeds:
            for kwargs in (
                    dict(
                        num_small=1000,
                        num_large=0,
                    ),
                    dict(
                        num_small=30,
                        num_large=600,
                    ),
                    dict(
                        num_small=300,
                        num_large=300,
                    ),
            ):
                idx += 1
                for path, quality in (
                        ("source", False),
                        ("target", True),
                ):
                    scene = create_script(seed, **kwargs, quality=quality)
                    render_scene(scene, output_path / split_path / path / f"tubes-{idx:02}.png")

                seed += 12345


if __name__ == "__main__":
    #render_test()
    render_dataset()

