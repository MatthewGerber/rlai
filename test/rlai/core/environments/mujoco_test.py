import os

import mujoco


def test_cube():
    """
    Test MuJoCo with a cube mesh.
    """

    XML = r"""
    <mujoco>
      <asset>
        <mesh file="gizmo.stl"/>
      </asset>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="mesh" name="gizmo" mesh="gizmo"/>
        </body>
      </worldbody>
    </mujoco>
    """

    ASSETS = dict()
    with open(f'{os.path.dirname(__file__)}/fixtures/Cube_3d_printing_sample.stl', 'rb') as f:
        ASSETS['gizmo.stl'] = f.read()

    model = mujoco.MjModel.from_xml_string(xml=XML, assets=ASSETS)  # type: ignore
    data = mujoco.MjData(model)
    while data.time < 1:
        mujoco.mj_step(model, data)
        print(data.geom_xpos)
