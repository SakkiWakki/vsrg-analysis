"""Milkshape ASCII model loader (model_txt): the Model-actor tier's
geometry source. A synthetic two-triangle model exercises the layout;
the material's `sphere.png` suffix flags SM's environment mapping."""
import numpy as np
import pytest

from analysis.games.notitg.model_txt import is_model_reference, load_model

_MODEL = '''// MilkShape 3D ASCII
Frames: 30
Frame: 1
Meshes: 1
"tri" 0 0
3
0 -10.0 0.0 5.0 0.25 0.75 0
0 10.0 0.0 5.0 0.5 0.5 0
0 0.0 10.0 -5.0 1.0 0.0 0
2
0.0 0.0 1.0
0.0 1.0 0.0
2
0 0 1 2 0 0 1 1
0 2 1 0 1 1 0 1
Materials: 1
"None"
0.0 0.0 0.0 1.0
1.0 1.0 1.0 1.0
0.0 0.0 0.0 1.0
0.0 0.0 0.0 1.0
96.0
1.0
"textures\\\\shade sphere.png"
""
Bones: 0
'''


@pytest.fixture
def model(tmp_path):
    (tmp_path / 'textures').mkdir()
    (tmp_path / 'textures' / 'shade sphere.png').write_bytes(b'png')
    path = tmp_path / 'tri.txt'
    path.write_text(_MODEL)
    return load_model(path)


def test_triangles_flatten_with_positions_uvs_and_normals(model):
    assert model is not None and len(model) == 1
    v = model[0]['vertices']
    assert v.shape == (6, 8) and v.dtype == np.float32
    # Triangle 0 corner 0: vertex 0's xyz + uv, normal index 0.
    assert list(v[0]) == pytest.approx(
        [-10.0, 0.0, 5.0, 0.25, 0.75, 0.0, 0.0, 1.0])
    # Triangle 1 leads with vertex 2 and normal 1.
    assert list(v[3][:3]) == pytest.approx([0.0, 10.0, -5.0])
    assert list(v[3][5:]) == pytest.approx([0.0, 1.0, 0.0])


def test_material_texture_resolves_and_flags_spheremap(model):
    assert model[0]['texture'].endswith('textures/shade sphere.png')
    assert model[0]['spheremap'] is True


def test_is_model_reference_is_the_txt_suffix():
    assert is_model_reference('models/macplus.txt')
    assert not is_model_reference('images/av.png')
    assert not is_model_reference(None)
