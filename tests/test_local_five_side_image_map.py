from pathlib import Path
import sys
import types

# Local path-resolution tests do not need real Arena camera hardware.
arena_api = types.ModuleType("arena_api")
arena_system = types.ModuleType("arena_api.system")
arena_buffer = types.ModuleType("arena_api.buffer")
arena_system.system = types.SimpleNamespace(device_infos=[])
arena_buffer.BufferFactory = types.SimpleNamespace()
sys.modules.setdefault("arena_api", arena_api)
sys.modules.setdefault("arena_api.system", arena_system)
sys.modules.setdefault("arena_api.buffer", arena_buffer)

from src.COMMON.cycle_engine import build_local_image_map


def test_numbered_five_side_local_mapping(tmp_path: Path):
    expected = {
        "sidewall1": "1.png",
        "sidewall2": "2.jpg",
        "innerwall": "3.jpeg",
        "tread": "4.tif",
        "bead": "5.bmp",
    }
    for filename in expected.values():
        (tmp_path / filename).write_bytes(b"test")

    result = build_local_image_map(
        tmp_path,
        ["sidewall1", "sidewall2", "innerwall", "tread", "bead"],
    )

    assert {side: Path(path).name for side, path in result.items()} == expected


def test_old_single_file_env_path_uses_parent_for_five_sides(tmp_path: Path):
    for index in range(1, 6):
        (tmp_path / f"{index}.png").write_bytes(b"test")

    result = build_local_image_map(
        tmp_path / "1.png",
        ["sidewall1", "sidewall2", "innerwall", "tread", "bead"],
    )

    assert Path(result["sidewall1"]).name == "1.png"
    assert Path(result["sidewall2"]).name == "2.png"
    assert Path(result["innerwall"]).name == "3.png"
    assert Path(result["tread"]).name == "4.png"
    assert Path(result["bead"]).name == "5.png"
