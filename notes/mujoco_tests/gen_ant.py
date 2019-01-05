import os
import json
from xmltodict import parse, unparse
from elements import Mujoco


def gen_world():
    world = {
        "mujoco": {
            "worldbody": {
                "light": {
                    "@cutoff": "100",
                    "@diffuse": "1 1 1",
                    "@dir": "-0 0 -1.3",
                    "@directional": "true",
                    "@exponent": "1",
                    "@pos": "0 0 1.3",
                    "@specular": ".1 .1 .1"
                },
                "geom": {
                    "@conaffinity": "1",
                    "@condim": "3",
                    "@material": "MatPlane",
                    "@name": "floor",
                    "@pos": "0 0 0",
                    "@rgba": "0.8 0.9 0.8 1",
                    "@size": "20 20 .125",
                    "@type": "plane"
                },
                "body": {
                    "@name": "ball",
                    "@pos": "0.0 0.0 1.0",
                    "joint": [
                        {
                            "@name": "ball_joint",
                            "@pos": "0.0 0.0 0.2",
                            "@type": "ball",
                            "@damping": "2.0"
                        },
                        {
                            "@type": "slide",
                            "@axis": "0 0 1",
                            "@damping": "2.0"
                        },
                        {
                            "@type": "slide",
                            "@axis": "0 1 0",
                            "@damping": "2.0"
                        },
                        {
                            "@type": "slide",
                            "@axis": "1 0 0",
                            "@damping": "2.0"
                        }
                    ],
                    "geom": {
                        "@pos": "0.0 0.0 0.0",
                        "@size": "0.2",
                        "@type": "sphere"
                    }
                }
            },
            "asset": {
                "texture": [
                    {
                        "@builtin": "gradient",
                        "@height": "100",
                        "@rgb1": ".4 .5 .6",
                        "@rgb2": "0 0 0",
                        "@type": "skybox",
                        "@width": "100"
                    },
                    {
                        "@builtin": "checker",
                        "@height": "100",
                        "@name": "texplane",
                        "@rgb1": "0 0 0",
                        "@rgb2": "0.8 0.8 0.8",
                        "@type": "2d",
                        "@width": "100"
                    }
                ],
                "material": {
                    "@name": "MatPlane",
                    "@reflectance": "0.5",
                    "@shininess": "1",
                    "@specular": "1",
                    "@texrepeat": "60 60",
                    "@texture": "texplane"
                }
            }
        }
    }

    return world


def main():
    filepath = os.path.join('assets', 'minimal.xml')
    with open(filepath, 'r') as fh:
        raw = fh.read()
    markup = parse(raw)
    jason = json.dumps(markup, indent=4)

    world_dict = gen_world()
    world_markup = unparse(world_dict, pretty=True)

    # Output
    outpath = os.path.join('assets', 'minimal-gen.xml')
    with open(outpath, 'w') as fh:
        fh.write(world_markup)

    # Class based output
    world = Mujoco()


if __name__ == '__main__':
    main()
