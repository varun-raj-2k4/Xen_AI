# backend/autolabel.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import csv
import argparse

# ---------- Utilities ----------
def get_image_paths(folder):
    p = Path(folder)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [str(x) for x in sorted(Path(folder).rglob('*')) if x.suffix.lower() in exts]

def build_model(device):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model

def make_preprocessor():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

def embed_images(model, paths, device, batch_size=16):
    pre = make_preprocessor()
    all_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception as e:
                    print(f"WARN: failed to open {p}: {e}")
                    img = Image.new('RGB', (224,224), (127,127,127))
                imgs.append(pre(img))
            x = torch.stack(imgs, dim=0).to(device)
            feats = model(x)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu().numpy())
    if len(all_feats) == 0:
        # return appropriate zero-sized array
        return np.zeros((0,2048), dtype=np.float32)
    return np.vstack(all_feats)

def compute_prototypes(labeled_folder, model, device, min_seeds=1):
    classes = []
    prototypes = []
    labeled_feats_all = []   # optional: per-image features
    labeled_labels_all = []
    for cls_dir in sorted(Path(labeled_folder).iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        img_paths = get_image_paths(cls_dir)
        if len(img_paths) < min_seeds:
            print(f"Skipping class {cls} (only {len(img_paths)} seeds)")
            continue
        feats = embed_images(model, img_paths, device)
        if feats.shape[0] == 0:
            continue
        proto = feats.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-10)
        classes.append(cls)
        prototypes.append(proto)
        # Save per-image feats for optional kNN methods
        labeled_feats_all.append(feats)
        labeled_labels_all += [cls] * feats.shape[0]
        print(f"Class {cls}: {len(img_paths)} imgs -> prototype shape {proto.shape}")
    if len(prototypes) == 0:
        return classes, np.zeros((0,2048), dtype=np.float32), np.zeros((0,2048), dtype=np.float32), []
    labeled_feats_stack = np.vstack(labeled_feats_all) if len(labeled_feats_all) else np.zeros((0,2048))
    return classes, np.vstack(prototypes), labeled_feats_stack, labeled_labels_all

def assign_labels(unlabeled_paths, prototypes, proto_classes, model, device, threshold=0.75):
    """
    Assign label only if best similarity >= threshold.
    Returns list of tuples: (path, assigned_label_or_None, score, status)
    """
    results = []
    if prototypes.shape[0] == 0:
        for p in unlabeled_paths:
            results.append((p, None, 0.0, 'no_prototypes'))
        return results

    feats = embed_images(model, unlabeled_paths, device)
    if feats.shape[0] == 0:
        return [(p, None, 0.0, 'embed_failed') for p in unlabeled_paths]

    sims = feats.dot(prototypes.T)  # cosine sims because feats and prototypes are normalized
    for i, p in enumerate(unlabeled_paths):
        best_idx = int(np.argmax(sims[i]))
        best_score = float(sims[i, best_idx])
        best_class = proto_classes[best_idx]
        if best_score >= threshold:
            assigned = best_class
            status = 'match'
        else:
            assigned = None
            status = 'no_match'
        # debug print (comment out if noisy)
        # print(f"[DEBUG] {p} -> best={best_class} score={best_score:.4f} status={status}")
        results.append((p, assigned, best_score, status))
    return results

def save_results_csv(results, outpath, include_none=True):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'assigned_class', 'score', 'status'])
        for row in results:
            path, assigned, score, status = row
            if not include_none and assigned is None:
                continue
            writer.writerow([path, assigned if assigned is not None else "None", f"{score:.6f}", status])
    print(f"Saved {len(results)} rows (include_none={include_none}) to {outpath}")

# ---------- Main callable ----------
def run_autolabel(labeled, unlabeled, out, threshold=0.75, batch=32, min_seeds=1, include_none=True):
    """
    Run autolabel pipeline.
      labeled: path to labeled folder (contains class subfolders)
      unlabeled: path to unlabeled images folder
      out: output csv path
      threshold: cosine similarity threshold for accepting a match
      include_none: if False, skip writing None matches to CSV
    Returns: results list
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    model = build_model(device)

    print('\n== Computing prototypes from labeled set ==')
    classes, prototypes, labeled_feats, labeled_labels = compute_prototypes(labeled, model, device, min_seeds=min_seeds)
    print('Prototypes count:', len(classes))

    print('\n== Embedding & assigning unlabeled images ==')
    unlabeled_paths = get_image_paths(unlabeled)
    print('Unlabeled images:', len(unlabeled_paths))

    results = assign_labels(unlabeled_paths, prototypes, classes, model, device, threshold=threshold)

    save_results_csv(results, out, include_none=include_none)
    print('\nDone.')
    return results

# CLI entrypoint for standalone use
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labeled', required=True, help='path to labeled folder (contains class subfolders)')
    p.add_argument('--unlabeled', required=True, help='path to folder with unlabeled images')
    p.add_argument('--out', default='outputs/labels.csv', help='output CSV path')
    p.add_argument('--threshold', type=float, default=0.75, help='cosine threshold for match')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--min_seeds', type=int, default=1, help='min seed images per class to use')
    p.add_argument('--include_none', action='store_true', help='include None matches in CSV')
    return p.parse_args()

def main():
    args = parse_args()
    run_autolabel(args.labeled, args.unlabeled, args.out,
                  threshold=args.threshold, batch=args.batch,
                  min_seeds=args.min_seeds, include_none=args.include_none)

if __name__ == '__main__':
    main()
