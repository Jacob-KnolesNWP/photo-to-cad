"""STL and OBJ writers. Units are whatever the vertex array is in (millimetres)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return normals / length


def write_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    normals = _face_normals(vertices, faces).astype(np.float32)
    with open(path, "wb") as handle:
        handle.write(b"TrueShape mesh, units mm".ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(faces)))
        for normal, face in zip(normals, faces):
            handle.write(struct.pack("<3f", *normal))
            for index in face:
                handle.write(struct.pack("<3f", *vertices[index]))
            handle.write(struct.pack("<H", 0))


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float64)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# TrueShape mesh, units mm\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.5f} {vertex[1]:.5f} {vertex[2]:.5f}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def read_stl_vertices(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    vertices = np.empty((count * 3, 3), dtype=np.float64)
    offset = 84
    for index in range(count):
        # normal (3) + 3 vertices
        floats = struct.unpack_from("<12f", data, offset)
        vertices[index * 3 : index * 3 + 3] = np.array(floats[3:]).reshape(3, 3)
        offset += 50
    return vertices
