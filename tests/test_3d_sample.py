"""End-to-end 3D on a synthetic walk-around of a block with upright markers."""

import numpy as np

from partcapture.pipeline.errors import CaptureError
from partcapture.pipeline.mesh_export import read_stl_vertices, write_stl
from partcapture.pipeline.photogrammetry import reconstruct_object
from partcapture.synthetic3d import sample_object_views


def test_too_few_photos(tmp_path):
    images, _truth = sample_object_views(n_views=3, image_size=(320, 240))
    try:
        reconstruct_object(images, tmp_path / "short")
    except CaptureError as exc:
        assert exc.code == "TOO_FEW_PHOTOS"
        return
    raise AssertionError("a 3-photo set should be rejected")


def test_sample_object_reconstructs_and_scales(tmp_path):
    images, truth = sample_object_views(n_views=12, image_size=(640, 480))
    result = reconstruct_object(images, tmp_path / "job")
    assert result["scaled_by"] == "markers"
    assert result["registered_images"] >= 8
    assert result["marker_edges"] >= 4
    assert len(result["faces"]) > 200
    size = np.array(result["extents"]["size_mm"])
    # The cloud includes the markers standing around a 100 × 70 × 50 mm block,
    # so the bbox is larger than the block and a bit soft from the voxel mesh.
    # It must be in millimetres, not an arbitrary SfM scale.
    assert 35 < size[2] < 90, size
    assert 140 < max(size[0], size[1]) < 320, size
    assert result["accuracy"]["estimated_error_mm"] > 1.0

    stl_path = tmp_path / "block.stl"
    write_stl(stl_path, result["vertices"], result["faces"])
    loaded = read_stl_vertices(stl_path)
    assert len(loaded) == len(result["faces"]) * 3
    # STL round-trip keeps the overall size.
    loaded_size = loaded.max(axis=0) - loaded.min(axis=0)
    assert np.allclose(np.sort(loaded_size), np.sort(size), atol=0.5)
