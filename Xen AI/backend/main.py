# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, Request, Header, HTTPException, Query, Path
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path, PurePosixPath
import aiofiles
import time
import zipfile
import tempfile
import shutil
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import json
import hmac
import hashlib
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import torch

# local autolabel functions (assumed present in backend/autolabel.py)
from .autolabel import run_autolabel, build_model, make_preprocessor, compute_prototypes

# --- Directories ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LABELED_DIR = DATA_DIR / "labeled"
UNLABELED_DIR = DATA_DIR / "unlabeled"
OUTPUT_DIR = ROOT / "output"
PROJECTS_ROOT = ROOT / "projects"

LABELED_DIR.mkdir(parents=True, exist_ok=True)
UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

# --- FastAPI app ---
app = FastAPI(title="Autolabel Backend (Xen AI)")

# Mount static pages
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# OPTION A: serve project files from a different prefix so API routes under /projects/* are free.
app.mount("/project_files", StaticFiles(directory=str(PROJECTS_ROOT)), name="project_files")

# CORS (development convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# No-cache convenience middleware for static-ish endpoints
@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response: Response = await call_next(request)
    path = request.url.path
    if path.startswith("/static") or path.startswith("/output") or path.startswith("/data") or path.startswith("/project_files"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# --- Basic homepage handler (serves static index if present) ---
@app.get("/", response_class=FileResponse)
async def home():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    if index_path.exists():
        return str(index_path)
    return HTMLResponse(
        "<h2>Autolabel backend running 🚀</h2>"
        "<ul>"
        '<li><a href="/static/upload_seed.html">Upload seed images</a></li>'
        '<li><a href="/static/upload_unlabeled.html">Upload unlabeled images</a></li>'
        "</ul>"
    )

# -------------------- Upload endpoints --------------------

@app.post("/upload-seed")
async def upload_seed(label: str = Form(...), files: List[UploadFile] = File(...)):
    label_dir = LABELED_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        rel_path = PurePosixPath(f.filename)
        dest = label_dir.joinpath(*rel_path.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dest, "wb") as out_file:
            contents = await f.read()
            await out_file.write(contents)
        saved.append(str(dest.relative_to(ROOT)))
    return JSONResponse({"label": label, "count": len(saved), "saved": saved})

@app.post("/upload-unlabeled")
async def upload_unlabeled(files: List[UploadFile] = File(...)):
    session = str(int(time.time()))
    session_dir = UNLABELED_DIR / session
    session_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        rel_path = PurePosixPath(f.filename)
        dest = session_dir.joinpath(*rel_path.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dest, "wb") as out_file:
            contents = await f.read()
            await out_file.write(contents)
        saved.append(str(dest.relative_to(ROOT)))
    return JSONResponse({"session": session, "count": len(saved), "saved": saved})

@app.post("/upload-zip-seed")
async def upload_zip_seed(label: str = Form(...), zipfile_obj: UploadFile = File(...)):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        zip_path = tmpdir / zipfile_obj.filename
        async with aiofiles.open(zip_path, "wb") as f:
            await f.write(await zipfile_obj.read())
        label_dir = LABELED_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                rel = PurePosixPath(member.filename)
                dest = (label_dir / Path(*rel.parts)).resolve()
                if not str(dest).startswith(str(label_dir.resolve())):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return JSONResponse({"label": str(label), "status": "extracted"})

@app.post("/upload-zip-unlabeled")
async def upload_zip_unlabeled(zipfile_obj: UploadFile = File(...)):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        zip_path = tmpdir / zipfile_obj.filename
        async with aiofiles.open(zip_path, "wb") as f:
            await f.write(await zipfile_obj.read())
        session = str(int(time.time()))
        session_dir = UNLABELED_DIR / session
        session_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                rel = PurePosixPath(member.filename)
                dest = (session_dir / Path(*rel.parts)).resolve()
                if not str(dest).startswith(str(session_dir.resolve())):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return JSONResponse({"session": session, "status": "extracted"})

@app.get("/unlabeled/{session}")
async def list_unlabeled(session: str):
    session_dir = UNLABELED_DIR / session
    if not session_dir.exists():
        return JSONResponse({"session": session, "exists": False})
    files = [str(p.relative_to(ROOT)) for p in sorted(session_dir.rglob("*")) if p.is_file()]
    return JSONResponse({"session": session, "exists": True, "files": files})

@app.get("/latest-session")
async def latest_session():
    if not UNLABELED_DIR.exists():
        return JSONResponse({"session": None})
    sessions = [p for p in UNLABELED_DIR.iterdir() if p.is_dir()]
    if not sessions:
        return JSONResponse({"session": None})
    try:
        latest = max(sessions, key=lambda p: int(p.name))
    except Exception:
        latest = max(sessions, key=lambda p: p.stat().st_mtime)
    return JSONResponse({"session": latest.name})

# --- Run autolabel and optionally create project ---
@app.post("/autolabel/{session}")
async def autolabel_session_auto_project(session: str, create_project: bool = Query(False), project_name: Optional[str] = Query("autolabel_project")):
    labeled_dir = LABELED_DIR
    unlabeled_dir = UNLABELED_DIR / session
    out_csv = OUTPUT_DIR / f"{session}.csv"

    if not unlabeled_dir.exists():
        return JSONResponse({"error": f"Session {session} not found"}, status_code=404)

    created_project = None
    returned_api_key = None
    if create_project:
        project_id = str(uuid.uuid4())
        project_dir = PROJECTS_ROOT / project_id
        project_dir.mkdir(parents=True, exist_ok=False)

        api_key = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8")
        meta = {
            "project_id": project_id,
            "name": project_name or "autolabel_project",
            "api_key_hash": _hash_key(api_key),
            "created_at": int(time.time())
        }
        _write_meta(project_dir, meta)

        created_project = project_id
        returned_api_key = api_key

    # Run autolabel (your implementation)
    results = run_autolabel(str(labeled_dir), str(unlabeled_dir), str(out_csv))
    resp = {"session": session, "count": len(results), "output_csv": f"/output/{out_csv.name}"}

    if created_project:
        project_dir = PROJECTS_ROOT / created_project
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model()
        classes, prototypes, labeled_feats_stack, labeled_labels_all = compute_prototypes(
            str(LABELED_DIR), model, device, min_seeds=1
        )

        if prototypes.shape[0] == 0:
            try:
                shutil.rmtree(project_dir)
            except Exception:
                pass
            return JSONResponse({"error": "no prototypes computed"}, status_code=500)

        np.save(project_dir / "prototypes.npy", prototypes)
        (project_dir / "classes.json").write_text(json.dumps(classes, indent=2))
        if labeled_feats_stack.shape[0] > 0:
            np.save(project_dir / "labeled_embeddings.npy", labeled_feats_stack)
            (project_dir / "labeled_labels.json").write_text(json.dumps(labeled_labels_all, indent=2))

        try:
            meta = json.loads((project_dir / "meta.json").read_text())
        except Exception:
            meta = {}
        meta["published_at"] = int(time.time())
        _write_meta(project_dir, meta)

        resp["published"] = True
        resp["n_classes"] = len(classes)
        resp["project_id"] = created_project
        resp["api_key"] = returned_api_key

    return JSONResponse(resp)

# -------------------- Key / project helpers --------------------

_API_KEY_SALT = os.environ.get("AUTO_LABEL_KEY_SALT", "autolabel-secret-salt-v1").encode("utf-8")

def _hash_key(key: str) -> str:
    return hmac.new(_API_KEY_SALT, key.encode("utf-8"), hashlib.sha256).hexdigest()

def _write_meta(project_dir: Path, meta: dict):
    (project_dir / "meta.json").write_text(json.dumps(meta, indent=2))

@app.post("/projects/create")
async def create_project(name: str = Form("my_project")):
    project_id = str(uuid.uuid4())
    project_dir = PROJECTS_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=False)

    api_key = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8")
    meta = {
        "project_id": project_id,
        "name": name,
        "api_key_hash": _hash_key(api_key),
        "created_at": int(time.time())
    }
    _write_meta(project_dir, meta)
    return JSONResponse({"project_id": project_id, "api_key": api_key, "note": "Save the api_key now — it will not be shown again."})

def verify_project_api(project_id: str, api_key: str) -> bool:
    project_dir = PROJECTS_ROOT / project_id
    if not project_dir.exists():
        return False
    try:
        meta = json.loads((project_dir / "meta.json").read_text())
    except Exception:
        return False
    return hmac.compare_digest(meta.get("api_key_hash", ""), _hash_key(api_key))

@app.get("/projects/{project_id}/meta")
async def get_project_meta(project_id: str):
    project_dir = PROJECTS_ROOT / project_id
    if not project_dir.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = json.loads((project_dir / "meta.json").read_text())
    return JSONResponse({"project_id": meta.get("project_id"), "name": meta.get("name"), "created_at": meta.get("created_at")})

# -------------------- Export helpers --------------------

def export_pt_model(project_dir: Path, classes: list, prototypes: np.ndarray):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for .pt export: pip install torch") from e

    pt_path = project_dir / "exports" / "model.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"classes": classes, "prototypes": prototypes.astype(np.float32)}
    torch.save(payload, str(pt_path))
    return pt_path

def export_onnx_model(project_dir: Path, prototypes: np.ndarray):
    try:
        import onnx
        from onnx import helper, TensorProto
    except Exception as e:
        raise RuntimeError("ONNX packages required (pip install onnx)") from e

    num_classes, dim = prototypes.shape

    embedding = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, dim])
    class_out = helper.make_tensor_value_info("class_id", TensorProto.INT64, [])

    proto_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["prototypes_raw"],
        value=onnx.helper.make_tensor(
            name="prototypes_tensor",
            data_type=TensorProto.FLOAT,
            dims=[num_classes, dim],
            vals=prototypes.flatten().tolist()
        ),
    )

    norm_embed = helper.make_node("LpNormalization", inputs=["embedding"], outputs=["norm_embed"], p=2, axis=1)
    norm_proto = helper.make_node("LpNormalization", inputs=["prototypes_raw"], outputs=["norm_prototypes"], p=2, axis=1)
    transpose_proto = helper.make_node("Transpose", inputs=["norm_prototypes"], outputs=["norm_prototypes_T"], perm=[1, 0])
    matmul = helper.make_node("MatMul", inputs=["norm_embed", "norm_prototypes_T"], outputs=["similarity"])
    squeeze = helper.make_node("Squeeze", inputs=["similarity"], outputs=["similarity_1d"], axes=[0])
    argmax = helper.make_node("ArgMax", inputs=["similarity_1d"], outputs=["class_id"], axis=0, keepdims=0)

    graph = helper.make_graph(
        nodes=[proto_const, norm_embed, norm_proto, transpose_proto, matmul, squeeze, argmax],
        name="PrototypeClassifier",
        inputs=[embedding],
        outputs=[class_out],
    )

    model = helper.make_model(graph, producer_name="xen-ai-autolabel")
    onnx.checker.check_model(model)

    out_path = project_dir / "exports" / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))

    meta = {
        "type": "prototype-onnx",
        "embedding_input": True,
        "embedding_size": int(dim),
        "expected_image_size": [224, 224],
        "notes": "This ONNX expects embedding vectors as input. To include backbone, trace the backbone and combine."
    }
    (project_dir / "exports" / "metadata.json").write_text(json.dumps(meta, indent=2))
    return out_path

# -------------------- API endpoints: export & zip --------------------

@app.post("/projects/{project_id}/export")
async def export_project_models(project_id: str):
    project_dir = PROJECTS_ROOT / project_id
    if not project_dir.exists():
        return JSONResponse({"error": "project not found"}, status_code=404)

    try:
        prototypes = np.load(project_dir / "prototypes.npy")
        classes = json.loads((project_dir / "classes.json").read_text())
    except Exception as e:
        return JSONResponse({"error": f"missing artifacts: {str(e)}"}, status_code=400)

    exports_dir = project_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    resp = {"project_id": project_id, "downloads": {}}

    # PyTorch (.pt)
    try:
        pt_path = export_pt_model(project_dir, classes, prototypes)
        resp["downloads"]["pt"] = f"/project_files/{project_id}/exports/{pt_path.name}"
    except Exception as e:
        resp["downloads"]["pt_error"] = str(e)

    # ONNX
    try:
        onnx_path = export_onnx_model(project_dir, prototypes)
        resp["downloads"]["onnx"] = f"/project_files/{project_id}/exports/{onnx_path.name}"
    except Exception as e:
        resp["downloads"]["onnx_error"] = str(e)

    return JSONResponse(resp)

@app.get("/projects/{project_id}/download-zip")
async def download_project_exports_zip(project_id: str):
    project_dir = PROJECTS_ROOT / project_id
    exports_dir = project_dir / "exports"
    if not exports_dir.exists() or not any(exports_dir.iterdir()):
        return JSONResponse({"error": "no exports found for project"}, status_code=404)

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_zip_path = Path(tmp_zip.name)
    tmp_zip.close()

    try:
        with zipfile.ZipFile(str(tmp_zip_path), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(exports_dir.rglob("*")):
                if f.is_file():
                    arcname = f.relative_to(project_dir)
                    z.write(str(f), arcname.as_posix())
        filename = f"xen_project_{project_id}_exports.zip"
        return FileResponse(path=str(tmp_zip_path), filename=filename, media_type="application/zip")
    finally:
        # the file will be removed by OS on restart; if you want immediate cleanup,
        # consider scheduling a cleanup job. Avoid removing immediately because FileResponse
        # may still be streaming the file. For simplicity we won't delete here.
        pass

# -------------------- Publish prototypes for a project --------------------

@app.post("/projects/{project_id}/publish")
async def publish_project(project_id: str):
    project_dir = PROJECTS_ROOT / project_id
    if not project_dir.exists():
        return JSONResponse({"error": "project not found"}, status_code=404)

    if not LABELED_DIR.exists() or not any(LABELED_DIR.iterdir()):
        return JSONResponse({"error": "no labeled data"}, status_code=400)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model()

    classes, prototypes, labeled_feats_stack, labeled_labels_all = compute_prototypes(
        str(LABELED_DIR), model, device, min_seeds=1
    )

    if prototypes.shape[0] == 0:
        return JSONResponse({"error": "no prototypes computed"}, status_code=500)

    np.save(project_dir / "prototypes.npy", prototypes)
    (project_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    if labeled_feats_stack.shape[0] > 0:
        np.save(project_dir / "labeled_embeddings.npy", labeled_feats_stack)
        (project_dir / "labeled_labels.json").write_text(json.dumps(labeled_labels_all, indent=2))

    try:
        meta = json.loads((project_dir / "meta.json").read_text())
    except Exception:
        meta = {"project_id": project_id}
    meta["published_at"] = int(time.time())
    _write_meta(project_dir, meta)

    return JSONResponse({"project_id": project_id, "n_classes": len(classes), "status": "published"})

# -------------------- Prediction API (requires project headers) --------------------

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = build_model(dev)
    return _MODEL

@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    threshold: float = Form(0.5),
    x_project_id: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    if x_project_id is None or x_api_key is None:
        raise HTTPException(status_code=401, detail="X-Project-Id and X-Api-Key headers required")
    if not verify_project_api(x_project_id, x_api_key):
        raise HTTPException(status_code=403, detail="Invalid project id or api key")

    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read image file")

    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_dir = PROJECTS_ROOT / x_project_id
    prototypes_path = project_dir / "prototypes.npy"
    classes_path = project_dir / "classes.json"

    if prototypes_path.exists() and classes_path.exists():
        prototypes = np.load(prototypes_path)
        classes = json.loads(classes_path.read_text())
        labeled_emb_path = project_dir / "labeled_embeddings.npy"
        labeled_labels_path = project_dir / "labeled_labels.json"
        if labeled_emb_path.exists() and labeled_labels_path.exists():
            labeled_feats = np.load(labeled_emb_path)
            labeled_labels = json.loads(labeled_labels_path.read_text())
        else:
            labeled_feats = None
            labeled_labels = None
    else:
        if not LABELED_DIR.exists() or not any(LABELED_DIR.iterdir()):
            raise HTTPException(status_code=400, detail="No labeled classes found. Upload seed images first.")
        classes, prototypes, labeled_feats_stack, labeled_labels_all = compute_prototypes(str(LABELED_DIR), model, device, min_seeds=1)
        if prototypes.shape[0] == 0:
            raise HTTPException(status_code=500, detail="Failed to compute class prototypes")
        labeled_feats = labeled_feats_stack if labeled_feats_stack.shape[0] else None
        labeled_labels = labeled_labels_all if labeled_labels_all else None

    pre = make_preprocessor()
    x = pre(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_t = model(x)
        feat = feat_t.cpu().numpy().reshape(-1)
        feat = feat / (np.linalg.norm(feat) + 1e-10)

    sims = prototypes.dot(feat)
    top_k = max(1, int(top_k))
    k = min(len(sims), top_k)
    top_idxs = np.argsort(-sims)[:k]

    top_matches = []
    for idx in top_idxs:
        cls = classes[idx]
        sc = float(sims[idx])
        top_matches.append({"label": cls, "score": sc})

    best_score = float(sims[top_idxs[0]])
    confidence = (best_score + 1.0) / 2.0
    effective_threshold = max(0.75, float(threshold))
    predicted = "none" if confidence < effective_threshold else classes[top_idxs[0]]

    nearest_files = None
    if predicted != "none" and labeled_feats is not None:
        file_sims = labeled_feats.dot(feat)
        top_file_idxs = np.argsort(-file_sims)[:min(5, file_sims.shape[0])]
        nearest_files = []
        for fi in top_file_idxs:
            label_for_file = labeled_labels[fi] if labeled_labels and fi < len(labeled_labels) else None
            nearest_files.append({"index": int(fi), "label": label_for_file, "score": float(file_sims[fi])})

    resp = {
        "predicted_label": predicted,
        "confidence": round(float(confidence), 4),
        "top_matches": top_matches
    }
    if nearest_files:
        resp["nearest_labeled_files"] = nearest_files

    return JSONResponse(resp)

# -------------------- Small helpers to locate optional frontend pages --------------------

def _find_candidate_path(names):
    base_dir = Path(__file__).resolve().parent
    candidates = []
    for n in names:
        n = Path(n)
        candidates.extend([base_dir / n, base_dir / "static" / n, ROOT / n, ROOT / "static" / n, Path("/mnt/data") / n])
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None

@app.get("/text-to-speech", response_class=FileResponse)
async def text_to_speech_page():
    candidates = ["text_to_speech.html", "text-to-speech.html", "Text_to_Speech.html"]
    p = _find_candidate_path(candidates)
    if p:
        return str(p)
    return HTMLResponse("<h3>Text to Speech page not found — place text_to_speech.html in backend/static/ or project root.</h3>", status_code=404)

@app.get("/image-generator", response_class=FileResponse)
async def image_generator_page():
    candidates = ["Image_generator.html", "image_generator.html", "image-generator.html", "Image-generator.html"]
    p = _find_candidate_path(candidates)
    if p:
        return str(p)
    return HTMLResponse("<h3>Image Generator page not found — place Image_generator.html in backend/static/ or project root.</h3>", status_code=404)
