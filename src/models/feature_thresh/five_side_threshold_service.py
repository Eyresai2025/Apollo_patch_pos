"""Application adapter for the AI-team five-side threshold cycle."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROLE_ORDER = ("sidewall1", "sidewall2", "tread", "innerwall", "bead")
ROLE_KIND = {"sidewall1": "sidewall", "sidewall2": "sidewall", "tread": "tread", "innerwall": "inner", "bead": "bead"}
ROLE_DISPLAY = {"sidewall1":"Sidewall 1","sidewall2":"Sidewall 2","tread":"Tread","innerwall":"Inner Side","bead":"Bead"}


def _emit(cb: Optional[Callable[[str], None]], text: str) -> None:
    if cb: cb(str(text))


def _copy_single_image(source: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.is_file() or old.is_symlink(): old.unlink()
        elif old.is_dir(): shutil.rmtree(old)
    target = folder / source.name
    try:
        os.link(str(source), str(target))
    except Exception:
        shutil.copy2(source, target)
    return target


def _find_calibration(media: Path, sku: str, role: str) -> Path:
    root = media / "offset_calibration" / sku / role
    if not root.is_dir():
        raise FileNotFoundError(f"{ROLE_DISPLAY[role]} calibration folder not found: {root}")
    candidates = sorted(
        [p for p in root.glob("*.json") if "calibration" in p.name.lower()],
        key=lambda p: (p.stat().st_mtime, p.name.lower()), reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No calibration JSON found for {ROLE_DISPLAY[role]} in {root}")
    return candidates[0].resolve()


def _validate_state(role: str, state: Dict[str, Any]) -> tuple[Path, Path, Optional[Path], Optional[Path]]:
    image = Path(str(state.get("image_path") or "")).expanduser().resolve()
    model = Path(str(state.get("model_path") or "")).expanduser().resolve()
    if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"Valid GOOD image is missing for {ROLE_DISPLAY[role]}: {image}")
    if not model.is_file():
        raise FileNotFoundError(f"PatchCore model is missing for {ROLE_DISPLAY[role]}: {model}")
    template = recipe = None
    if role.startswith("sidewall"):
        template = Path(str(state.get("template_path") or "")).expanduser().resolve()
        recipe = Path(str(state.get("recipe_path") or "")).expanduser().resolve()
        if not template.is_file(): raise FileNotFoundError(f"R template missing for {ROLE_DISPLAY[role]}: {template}")
        if not recipe.is_file(): raise FileNotFoundError(f"Fast R recipe missing for {ROLE_DISPLAY[role]}: {recipe}")
    return image, model, template, recipe


def build_five_side_config(*, media_path: Path, project_root: Path, sku_name: str,
                           states: Dict[str, Dict[str, Any]], percentile: float,
                           save_processing_images: bool = True) -> tuple[Path, Path]:
    media = Path(media_path).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    sku = str(sku_name).strip()
    pipeline = Path(__file__).resolve().parent / "five_side_threshold_pipeline"
    cycle_root = media / "feature_threshold" / sku / "five_side_cycle"
    input_root = cycle_root / "_selected_inputs"
    cycle_root.mkdir(parents=True, exist_ok=True)

    validated = {role: _validate_state(role, states[role]) for role in ROLE_ORDER}
    for role in ROLE_ORDER:
        (media / "feature_threshold" / sku / role).mkdir(parents=True, exist_ok=True)
    jobs=[]
    for role in ("sidewall1","sidewall2"):
        image, model, template, recipe = validated[role]
        selected_folder=input_root/role
        _copy_single_image(image, selected_folder)
        jobs.append({
            "name": role, "kind":"sidewall", "enabled":True,
            "script": str((pipeline/"calculate_threshold_detect_and_crop.py").resolve()),
            "good_raw_folder": str(selected_folder.resolve()),
            "r_template": str(template), "r_recipe_path": str(recipe),
            "r_detection_method":"fast", "r_fast_fallback_to_tiled":True,
            "model": str(model),
            "threshold": str((media/"feature_threshold"/sku/role/"threshold.json").resolve()),
            "processing_output_root": str((media/"feature_threshold"/sku/role/"processing").resolve()),
            "scores_csv": str((media/"feature_threshold"/sku/role/f"{role}_scores.csv").resolve()),
            "resize_width":4036, "resize_height":17920,
        })
    dims={"tread":(4032,23296),"innerwall":(2048,10000),"bead":(2048,10000)}
    for role in ("tread","innerwall","bead"):
        image, model, _, _ = validated[role]
        w,h=dims[role]
        input_key={"tread":"threshold_tread_input","innerwall":"threshold_inner_input","bead":"threshold_bead_input"}[role]
        jobs.append({
            "name":role, "kind":ROLE_KIND[role], "enabled":True,
            # Offset workers are implemented directly inside main_threshold_cycle.py.
            # The script field is retained only for AI-cycle preflight compatibility.
            "script":str((pipeline/"main_threshold_cycle.py").resolve()),
            "r_source_job":"sidewall1", input_key:str(image),
            "calibration":str(_find_calibration(media,sku,role)),
            "model":str(model),
            "threshold":str((media/"feature_threshold"/sku/role/"threshold.json").resolve()),
            "processing_output_root":str((media/"feature_threshold"/sku/role/"processing").resolve()),
            "resize_width":w,"resize_height":h,"tyre_type":f"{sku}_{role.upper()}",
        })
    cfg={
        "cycle_name":f"{sku}_five_side_threshold_cycle",
        "output_root":str(cycle_root.resolve()),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","0"),
        "cpu_threads_per_worker":1,"opencv_threads_per_worker":1,"torch_cpu_threads_per_worker":1,
        "image_batch_size":32,"memory_bank_chunk_size":20000,"percentile":float(percentile),
        "patch_width":448,"patch_height":448,"patch_stride_x":448,"patch_stride_y":448,
        "cover_complete":True,"save_raw_crop":bool(save_processing_images),
        "save_resized_crop":bool(save_processing_images),"save_generated_patches":False,
        "save_preview":bool(save_processing_images),"sidewall_parallel_workers":2,"offset_parallel_workers":3,
        "jobs":jobs,
    }
    config_path=cycle_root/"main_threshold_config.json"
    config_path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    return config_path, cycle_root


def run_five_side_threshold_cycle(*, media_path: Path, project_root: Path, sku_name: str,
                                  states: Dict[str, Dict[str, Any]], percentile: float,
                                  save_processing_images: bool = True,
                                  status_callback: Optional[Callable[[str],None]] = None) -> Dict[str, Dict[str, Any]]:
    config_path, cycle_root = build_five_side_config(
        media_path=media_path, project_root=project_root, sku_name=sku_name,
        states=states, percentile=percentile, save_processing_images=save_processing_images,
    )
    script=Path(__file__).resolve().parent/"five_side_threshold_pipeline"/"main_threshold_cycle.py"
    command=[sys.executable,"-u",str(script),"--config",str(config_path)]
    _emit(status_callback,"Starting AI-team five-side threshold cycle...")
    process=subprocess.Popen(command,cwd=str(script.parent),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,bufsize=1,encoding="utf-8",errors="replace")
    captured=[]
    assert process.stdout is not None
    for line in process.stdout:
        line=line.rstrip(); captured.append(line); _emit(status_callback,line)
    code=process.wait()
    summary_path=cycle_root/"main_threshold_cycle_summary.json"
    if not summary_path.is_file():
        tail="\n".join(captured[-30:])
        raise RuntimeError(f"Five-side threshold cycle exited with code {code} and produced no summary.\n{tail}")
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    results_by_name={str(x.get("name")):x for x in summary.get("results",[]) if isinstance(x,dict)}
    failures=[x for x in results_by_name.values() if x.get("status")!="success"]
    assets={}
    for role in ROLE_ORDER:
        item=results_by_name.get(role,{})
        threshold_path=Path(str(item.get("threshold") or (Path(media_path)/"feature_threshold"/sku_name/role/"threshold.json")))
        if item.get("status")!="success" or not threshold_path.is_file(): continue
        payload=json.loads(threshold_path.read_text(encoding="utf-8"))
        state=states[role]
        payload.update({
            "sku_name":sku_name,"role":role,"display_name":ROLE_DISPLAY[role],
            "threshold_json_path":str(threshold_path.resolve()),
            "model_path":str(Path(str(state.get("model_path") or "")).resolve()),
            "good_raw_image":str(Path(str(state.get("image_path") or "")).resolve()),
            "percentile":float(payload.get("percentile",percentile)),
            "processing_root":item.get("processing_root",str((Path(media_path)/"feature_threshold"/sku_name/role/"processing").resolve())),
            "five_side_cycle_summary":str(summary_path.resolve()),
        })
        if role.startswith("sidewall"):
            payload["R_template_path"]=str(Path(str(state.get("template_path") or "")).resolve())
            payload["R_recipe_path"]=str(Path(str(state.get("recipe_path") or "")).resolve())
        threshold_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        assets[role]=payload
    if code!=0 or failures or len(assets)!=5:
        details=[]
        for role in ROLE_ORDER:
            item=results_by_name.get(role,{})
            if item.get("status")!="success":
                details.append(f"{ROLE_DISPLAY[role]}: {item.get('status','missing')} - {item.get('error','No result')}")
        raise RuntimeError("Five-side threshold cycle did not complete all views.\n"+"\n".join(details))
    return assets
