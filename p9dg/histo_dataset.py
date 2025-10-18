
# histo_dataset.py
# DataGenerator histopathologie avec préprocessing à la volée
# - Split sans fuite (train = NCT-CRC-HE-100K, val/test = CRC-VAL-HE-7K)
# - Échantillonnage équilibré sur le train (sans réutilisation en val/test)
# - Pré-sélection qualité (LaplacianVar, ShannonEntropy, WhiteRatio, SaturationRatio,
#   Tenengrad, Blockiness spatial & DCT, TissueFraction) + score combiné de blockiness (z-scores)
# - Normalisation Vahadane via torch_staintools (si dispo)
# - Pixel range [0,1] ou normalisation ImageNet
# - Visualisation rapide (grille)
# - Compatible Docker, CPU/GPU

from __future__ import annotations

import math
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, Sampler

# -------------------------------------------------------------
# Utilitaires de base
# -------------------------------------------------------------
def _read_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _rgb_to_hsv_np(rgb: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Retourne (H,S,V) en numpy (H non utilisé ici → None)."""
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.zeros_like(v, dtype=np.float32)
    nz = maxc > 0
    s[nz] = (maxc[nz] - minc[nz]) / (maxc[nz] + 1e-8)
    return None, s, v

# -------------------------------------------------------------
# Filtre qualité (métriques + décision via seuils par classe)
# -------------------------------------------------------------
class QualityFilter:
    """Calcule les métriques qualité et prend une décision via seuils.
    Métriques: lap_var, entropy, tenengrad, white_ratio, sat_ratio, block_spatial, block_dct, tissue_fract.
    Le score combiné `jpeg_blockiness` (z-moyen des 2) est injecté par le Dataset au moment du check.
    """
    def __init__(
        self,
        lap_var_min: float = 50.0,
        entropy_min: float = 3.0,
        tenengrad_min: float = 150.0,
        white_ratio_max: float = 0.85,
        sat_ratio_max: float = 0.20,
        block_spatial_max: float = 25.0,
        block_dct_min: float = 0.40,
        block_dct_max: float = 0.98,
        tissue_fract_min: float = 0.10,
    ) -> None:
        self.defaults = dict(
            lap_var_min=lap_var_min,
            entropy_min=entropy_min,
            tenengrad_min=tenengrad_min,
            white_ratio_max=white_ratio_max,
            sat_ratio_max=sat_ratio_max,
            block_spatial_max=block_spatial_max,
            block_dct_min=block_dct_min,
            block_dct_max=block_dct_max,
            tissue_fract_min=tissue_fract_min,
        )

    @staticmethod
    def _to_np(img: Image.Image) -> np.ndarray:
        return np.asarray(img)

    @staticmethod
    def _luminance(rgb: np.ndarray) -> np.ndarray:
        return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)

    @staticmethod
    def _variance_of_laplacian(gray: np.ndarray) -> float:
        try:
            from scipy.signal import convolve2d
            k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            resp = convolve2d(gray, k, mode="same", boundary="symm")
            return float(resp.var())
        except Exception:
            # fallback simple: dérivées finies
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            lap = (np.diff(gx, axis=1, prepend=gx[:, :1]) + np.diff(gy, axis=0, prepend=gy[:1, :]))
            return float(lap.var())

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        p = hist[hist > 0]
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _tenengrad(gray: np.ndarray) -> float:
        try:
            from scipy.ndimage import sobel
            gx = sobel(gray, axis=1)
            gy = sobel(gray, axis=0)
        except Exception:
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        g2 = gx * gx + gy * gy
        return float(g2.mean())

    @staticmethod
    def _blockiness_spatial(gray: np.ndarray, block: int = 8) -> float:
        h, w = gray.shape
        vb = [i for i in range(block, w, block)]
        hb = [j for j in range(block, h, block)]
        if len(vb) == 0 and len(hb) == 0:
            return 0.0
        bdiffs = []
        for c in vb:
            if c < w:
                bdiffs.append(np.abs(gray[:, c:(c + 1)] - gray[:, (c - 1):c]).mean())
        for r in hb:
            if r < h:
                bdiffs.append(np.abs(gray[r:(r + 1), :] - gray[(r - 1):r, :]).mean())
        bmean = float(np.mean(bdiffs)) if bdiffs else 0.0
        # intra-bloc
        mask = np.ones_like(gray, dtype=bool)
        mask[:, vb] = False
        mask[hb, :] = False
        intra = []
        if mask[:, 1:].any():
            intra.append(np.abs(gray[:, 1:] - gray[:, :-1])[mask[:, 1:]].mean())
        if mask[1:, :].any():
            intra.append(np.abs(gray[1:, :] - gray[:-1, :])[mask[1:, :]].mean())
        imean = float(np.mean(intra)) if len(intra) > 0 else 0.0
        return max(0.0, bmean - imean)

    @staticmethod
    def _blockiness_dct(gray: np.ndarray, block: int = 8) -> float:
        try:
            from scipy.fftpack import dct
            def dct2(x):
                return dct(dct(x.T, norm="ortho").T, norm="ortho")
        except Exception:
            def dct2(x):
                X = np.fft.fft2(x)
                return np.real(X)
        h, w = gray.shape
        H = (h // block) * block
        W = (w // block) * block
        g = gray[:H, :W]
        g = g.reshape(H // block, block, W // block, block).transpose(0, 2, 1, 3)
        lows, totals = [], []
        for bi in range(g.shape[0]):
            for bj in range(g.shape[1]):
                patch = g[bi, bj]
                C = dct2(patch)
                E = (C * C)
                low = E[:2, :2].sum()
                tot = E.sum() + 1e-8
                lows.append(low)
                totals.append(tot)
        ratio = float(np.mean(np.array(lows) / np.array(totals))) if totals else 0.0
        return ratio

    def score(self, img: Image.Image) -> Dict[str, float]:
        arr = self._to_np(img)
        y = self._luminance(arr)
        lap = self._variance_of_laplacian(y)
        ent = self._entropy(y)
        ten = self._tenengrad(y)
        # HSV
        _, S, V = _rgb_to_hsv_np(arr)
        white_ratio = float(((V > 0.95) & (S < 0.10)).mean())
        sat_ratio = float((S > 0.90).mean())
        # blockiness
        bsp = self._blockiness_spatial(y)
        bdc = self._blockiness_dct(y)
        tissue_mask = ((V < 0.98) | (S > 0.20))
        tissue_fract = float(tissue_mask.mean())
        return {
            "lap_var": lap,
            "entropy": ent,
            "tenengrad": ten,
            "white_ratio": white_ratio,
            "sat_ratio": sat_ratio,
            "block_spatial": bsp,
            "block_dct": bdc,
            "tissue_fract": tissue_fract,
        }

    def check(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
        """
        Vérifie si les métriques d'une image respectent les seuils.
        - BACK : toujours accepté (aucun filtrage)
        - ADI  : seuils ultra-permissifs
        - Autres classes : filtrage complet comme avant
        """
        # Fusion avec valeurs par défaut
        thr = {**self.defaults, **(thresholds or {})}
        cname = thr.get("class_name", None)

        # --- BACK : toujours accepté ---
        if cname == "BACK":
            return True

        # --- ADI : seuils ultra-permissifs ---
        if cname == "ADI":
            # On rejette uniquement les cas aberrants (valeurs manquantes ou infinies)
            for v in metrics.values():
                if np.isnan(v) or np.isinf(v):
                    return False
            return True

        # --- Autres classes : filtrage standard ---
        if metrics["lap_var"] < thr["lap_var_min"]:
            return False
        if metrics["entropy"] < thr["entropy_min"]:
            return False
        if metrics["tenengrad"] < thr["tenengrad_min"]:
            return False
        if metrics["white_ratio"] > thr.get("white_ratio_max", 1.0):
            return False
        if metrics["sat_ratio"] > thr.get("sat_ratio_max", 1.0):
            return False

        # Blockiness combinée si disponible
        if ("jpeg_blockiness" in metrics) and ("jpeg_blockiness_max" in thr):
            if metrics["jpeg_blockiness"] > thr["jpeg_blockiness_max"]:
                return False
        else:
            if metrics["block_spatial"] > thr.get("block_spatial_max", thr.get("jpeg_blockiness_max", 1e9)):
                return False
            if not (thr["block_dct_min"] <= metrics["block_dct"] <= thr["block_dct_max"]):
                return False

        # Seuil tissue_fract uniquement si défini
        if metrics["tissue_fract"] < thr.get("tissue_fract_min", 0.0):
            return False

        return True


# -------------------------------------------------------------
# Normalisation Vahadane (torch_staintools)
# -------------------------------------------------------------
class TorchStainNormalizer:
    """
    Normalisation Vahadane via torch_staintools (initialisation stable, tolérante et reproductible).
    - Une seule image de référence (fourni par path ou fit_reference)
    - Pas de refit dans normalize()
    - Tolérant aux échecs (retourne l'image brute si erreur)
    - Compatible CPU / GPU
    """

    def __init__(self, enable=True, target_path=None, device="cpu", seed=42):
        self.enable = enable
        self.device = device
        self.seed = seed
        self._ok = False
        self._target = None
        self._normalizer = None

        if not enable:
            return

        try:
            import torch, cv2, random, numpy as np
            from torch_staintools.normalizer import NormalizerBuilder
            from torchvision import transforms

            self.torch = torch
            self.cv2 = cv2
            self.np = np
            self.transforms = transforms
            self.builder = NormalizerBuilder
            random.seed(seed)

            # --- Création du normaliseur ---
            self._normalizer = self.builder().build("vahadane").to(device)
            self._normalizer.luminosity_threshold = 0.85  # réglage tolérant

            # --- Chargement éventuel d'une image de référence ---
            if target_path and os.path.exists(target_path):
                self.fit_reference(target_path)

        except Exception as e:
            import warnings
            warnings.warn(f"⚠️ torch_staintools indisponible ou erreur init: {e}")
            self.enable = False

    # ---------------------------
    # Conversion PIL <-> Tensor
    # ---------------------------
    @staticmethod
    def _pil_to_t(img: Image.Image) -> "torch.Tensor":
        arr = np.asarray(img, dtype=np.uint8).copy()  # writable
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t

    @staticmethod
    def _t_to_pil(t: "torch.Tensor") -> Image.Image:
        t = t.clamp(0, 1)
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    # ---------------------------
    # Fit sur une image de référence
    # ---------------------------
    def fit_reference(self, img_path: str):
        """Apprend la décomposition Vahadane à partir d'une image de référence."""
        if not self.enable or not os.path.exists(img_path):
            return

        try:
            bgr = self.cv2.imread(str(img_path), self.cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Impossible de lire {img_path}")

            rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            ref_t = self.transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)

            self._normalizer.fit(ref_t)
            self._ok = True
            self._target = img_path
            print(f"🎨 Référence Vahadane fixée : {os.path.basename(img_path)}")

        except Exception as e:
            print(f"⚠️ Erreur lors du fit_reference sur {img_path}: {e}")
            self._ok = False

    # ---------------------------
    # Application de la normalisation
    # ---------------------------
    def normalize(self, img: Image.Image) -> Image.Image:
        """
        Applique la normalisation Vahadane à une image PIL.
        Retourne l'image d'origine si la normalisation échoue.
        """
        if not self.enable or not self._ok or self._normalizer is None:
            return img

        try:
            t = self._pil_to_t(img).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                t_norm = self._normalizer.transform(t)
            t_norm = t_norm.squeeze(0).clamp(0, 1)
            return self._t_to_pil(t_norm)

        except Exception as e:
            if os.environ.get("DEBUG_VAHADANE", "0") == "1":
                print(f"[TorchStainNormalizer] ⚠️ Erreur normalisation: {e}")
            return img


# -------------------------------------------------------------
# Dataset principal
# -------------------------------------------------------------
class HistoDataset(Dataset):
    def __init__(
        self,
        root_data: str = "/data",
        split: Literal["train", "val", "test"] = "train",
        split_policy: Literal["by_dataset", "manual_csv"] = "by_dataset",
        output_size: int | Tuple[int, int] = 256,
        pixel_range: Literal["0_1", "imagenet"] = "0_1",
        balance_per_class: bool = True,
        samples_per_class_per_epoch: Optional[int] = None,
        no_repeat_eval: bool = True,
        gan_dirs: Optional[List[str]] = None,
        gan_ratio: float = 0.0,
        # Normalisation couleur
        vahadane_enable: bool = True,
        vahadane_target_path: Optional[str] = None,
        vahadane_device: str = "cpu",
        # Split manuel optionnel
        manual_csv: Optional[str] = None,
        # Seuils par classe (JSON)
        thresholds_json_path: str = "seuils_par_classe.json",
        # Calibration blockiness
        calibrate_blockiness: bool = True,
        calib_max_per_class: int = 200,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = root_data
        self.split = split
        self.split_policy = split_policy
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.pixel_range = pixel_range
        self.balance_per_class = balance_per_class   # and split == "train")
        self.samples_per_class_per_epoch = samples_per_class_per_epoch
        self.no_repeat_eval = no_repeat_eval
        self.gan_dirs = gan_dirs or []
        self.gan_ratio = gan_ratio
        self.seed = seed

        # Qualité & normalisation
        self.qf = QualityFilter()
        self.stain = TorchStainNormalizer(enable=vahadane_enable,
                                        target_path=vahadane_target_path,
                                        device=vahadane_device,
                                        seed=seed)
        # Index et mapping (à initialiser AVANT le scan) ---
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.paths_by_class: Dict[int, List[str]] = {}
        self._epoch_indices: List[Tuple[int, int]] = []
        
        # --- Scanner chemins selon politique (remplit paths_by_class etc.) ---
        rng = random.Random(seed)
        self._scan_paths(rng, manual_csv)

        # Si aucune référence fournie, on en choisit une (TUM OK)
        if vahadane_enable and (self.stain._ok is False):
            try:
                tum_paths = self.paths_by_class[self.class_to_idx.get("TUM", 0)]
                if len(tum_paths) > 0:
                    # choix reproductible
                    ref_path = tum_paths[self.seed % len(tum_paths)]
                    self.stain.fit_reference(ref_path)
                    print(f"🎨 Référence Vahadane auto: {os.path.basename(ref_path)}")
            except Exception as e:
                print(f"⚠️ Impossible de fixer la référence Vahadane: {e}")


        # Seuils par classe (JSON)
        self.class_thresholds: Dict[str, Dict[str, float]] = {}
        try:
            import json
            config_dir = os.getenv("CONFIG_DIR", ".")
            path_json = os.path.join(config_dir, os.path.basename(thresholds_json_path))

            if os.path.exists(path_json):
                with open(path_json, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.class_thresholds = data
                print(f"✅ Seuils par classe chargés depuis : {path_json}")
            else:
                print(f"⚠️ Fichier de seuils introuvable : {path_json}")
        except Exception as e:
            warnings.warn(f"Impossible de charger {path_json}: {e}")


        # Calibration blockiness (mu/sigma) par classe
        self.block_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        if calibrate_blockiness:
            self._calibrate_blockiness_stats(max_per_class=calib_max_per_class)

        # Construire indices d'epoch
        self._build_epoch_indices(rng)

    # ------------------ Scan paths ------------------
    def _scan_paths(self, rng: random.Random, manual_csv: Optional[str]):
        if self.split_policy == "by_dataset":
            if self.split == "train":
                base = os.path.join(self.root, "NCT-CRC-HE-100K")
            else:
                base = os.path.join(self.root, "CRC-VAL-HE-7K")
            if not os.path.isdir(base):
                raise FileNotFoundError(f"Dossier introuvable: {base}")
            classes = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            for c in classes:
                cdir = os.path.join(base, c)
                files = [os.path.join(cdir, f) for f in os.listdir(cdir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
                rng.shuffle(files)
                self.paths_by_class[self.class_to_idx[c]] = files
        else:
            import csv
            if not manual_csv or not os.path.exists(manual_csv):
                raise ValueError("manual_csv requis pour split_policy='manual_csv'")
            rows = []
            with open(manual_csv, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("split") == self.split:
                        rows.append((r["path"], r["label"]))
            labels = sorted(set([lab for _, lab in rows]))
            self.class_to_idx = {c: i for i, c in enumerate(labels)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            for c in labels:
                files = [p for p, l in rows if l == c]
                rng.shuffle(files)
                self.paths_by_class[self.class_to_idx[c]] = files

        empty = [self.idx_to_class[i] for i, paths in self.paths_by_class.items() if len(paths) == 0]
        if empty:
            raise RuntimeError(f"Classes vides sur split {self.split}: {empty}")

    # --------------- Build epoch indices ---------------
    def _build_epoch_indices(self, rng: random.Random):
        per_class_counts = {c: len(p) for c, p in self.paths_by_class.items()}
        self._epoch_indices = []
        # Gestion équilibrée ou non, valable aussi pour val/test
        if self.balance_per_class and self.samples_per_class_per_epoch:
#         if self.split == "train" and self.balance_per_class:
            N = self.samples_per_class_per_epoch or min(per_class_counts.values())
            for ci, paths in self.paths_by_class.items():
                for _ in range(N):
                    j = rng.randrange(len(paths))  # avec remise si besoin
                    self._epoch_indices.append((ci, j))
            rng.shuffle(self._epoch_indices)
        else:
            # val/test: on consomme chaque échantillon exactement une fois
            for ci, paths in self.paths_by_class.items():
                idxs = list(range(len(paths)))
                rng.shuffle(idxs)
                for j in idxs:
                    self._epoch_indices.append((ci, j))
            rng.shuffle(self._epoch_indices)

    # --------------- Calibration blockiness ---------------
    def _calibrate_blockiness_stats(self, max_per_class: int = 200) -> None:
        for ci, paths in self.paths_by_class.items():
            cls = self.idx_to_class[ci]
            vals_sp, vals_dc = [], []
            take = min(len(paths), max_per_class)
            for j in range(take):
                try:
                    img = _read_rgb(paths[j])
                    m = self.qf.score(img)
                    vals_sp.append(m["block_spatial"])
                    vals_dc.append(m["block_dct"])
                except Exception:
                    continue
            if len(vals_sp) == 0:
                mu_s = 0.0; sd_s = 1.0
            else:
                mu_s = float(np.mean(vals_sp)); sd_s = float(np.std(vals_sp, ddof=0) or 1e-8)
            if len(vals_dc) == 0:
                mu_d = 0.0; sd_d = 1.0
            else:
                mu_d = float(np.mean(vals_dc)); sd_d = float(np.std(vals_dc, ddof=0) or 1e-8)
            self.block_stats[cls] = {"spatial": (mu_s, sd_s), "dct": (mu_d, sd_d)}

    # --------------- API Dataset ---------------
    def set_epoch(self, epoch: int):
        """Ré-échantillonne et re-shuffle les indices à chaque epoch."""
        rng = random.Random(self.seed + epoch)
        
        # Si on veut un vrai ré-échantillonnage équilibré (train/val possible)
        if self.balance_per_class and self.samples_per_class_per_epoch:
            self._build_epoch_indices(rng)
        else:
            # Pour les splits sans sampling, on garde juste le shuffle
            rng.shuffle(self._epoch_indices)

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize(self.output_size, Image.BILINEAR)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32).copy()
        arr = np.transpose(arr, (2, 0, 1)) / 255.0
        t = torch.from_numpy(arr)
        if self.pixel_range == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = (t - mean) / std
        return t

    def _load_path(self, ci: int, j: int) -> Tuple[Image.Image, str]:
        path = self.paths_by_class[ci][j]
        img = _read_rgb(path)
        return img, path

    def __getitem__(self, idx: int):
        ci, j = self._epoch_indices[idx]
        img, path = self._load_path(ci, j)
        class_name = self.idx_to_class[ci]
        thr = self.class_thresholds.get(class_name, {})
        # 1) Scores qualité + blockiness combinée (z-scores)
        metrics = self.qf.score(img)
        if class_name in self.block_stats and all(k in self.block_stats[class_name] for k in ("spatial", "dct")):
            mu_s, sd_s = self.block_stats[class_name]["spatial"]
            mu_d, sd_d = self.block_stats[class_name]["dct"]
            sd_s = sd_s if (sd_s not in (0.0, None) and not np.isnan(sd_s)) else 1e-8
            sd_d = sd_d if (sd_d not in (0.0, None) and not np.isnan(sd_d)) else 1e-8
            z_s = (metrics["block_spatial"] - mu_s) / sd_s
            z_d = (metrics["block_dct"] - mu_d) / sd_d
            metrics["jpeg_blockiness"] = float((z_s + z_d) / 2.0)
        # 2) Entrainement: on tente des remplacements limités si rejeté
        if self.split == "train":
            if not self.qf.check(metrics, thr):
                tries = 0
                while tries < 5:
                    tries += 1
                    j = (j + 1) % len(self.paths_by_class[ci])
                    img, path = self._load_path(ci, j)
                    metrics = self.qf.score(img)
                    if class_name in self.block_stats:
                        mu_s, sd_s = self.block_stats[class_name]["spatial"]
                        mu_d, sd_d = self.block_stats[class_name]["dct"]
                        sd_s = sd_s if (sd_s not in (0.0, None) and not np.isnan(sd_s)) else 1e-8
                        sd_d = sd_d if (sd_d not in (0.0, None) and not np.isnan(sd_d)) else 1e-8
                        z_s = (metrics["block_spatial"] - mu_s) / sd_s
                        z_d = (metrics["block_dct"] - mu_d) / sd_d
                        metrics["jpeg_blockiness"] = float((z_s + z_d) / 2.0)
                    if self.qf.check(metrics, thr):
                        break
        # 3) Normalisation Vahadane (si activée)
        img = self.stain.normalize(img)
        # 4) Resize + tensor
        img = self._resize(img)
        x = self._to_tensor(img)
        y = ci
        return x, y, path

    # --------------- Utils ---------------
    def class_counts(self) -> Dict[str, int]:
        return {self.idx_to_class[i]: len(p) for i, p in self.paths_by_class.items()}

    def vis(self, n: int = 16) -> Image.Image:
        per_class = max(1, n // len(self.paths_by_class))
        picks: List[Tuple[int, int]] = []
        rng = random.Random(self.seed + 123)
        for ci, paths in self.paths_by_class.items():
            for _ in range(per_class):
                j = rng.randrange(len(paths))
                picks.append((ci, j))
        picks = picks[:n]
        tiles: List[Image.Image] = []
        for ci, j in picks:
            img, _ = self._load_path(ci, j)
            img = self.stain.normalize(img)
            img = self._resize(img)
            border = 2
            color = tuple(int(x) for x in (np.random.RandomState(ci).rand(3) * 255))
            img = ImageOps.expand(img, border=border, fill=color)
            tiles.append(img)
        cols = int(math.ceil(math.sqrt(len(tiles))))
        rows = int(math.ceil(len(tiles) / cols))
        w, h = tiles[0].size
        canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
        for k, t in enumerate(tiles):
            r = k // cols; c = k % cols
            canvas.paste(t, (c * w, r * h))
        return canvas

# -------------------------------------------------------------
# Sampler équilibré (round-robin) - optionnel
# -------------------------------------------------------------
class BalancedRoundRobinSampler(Sampler[int]):
    def __init__(self, dataset: HistoDataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        self.by_class: Dict[int, List[int]] = {}
        for i, (ci, _) in enumerate(dataset._epoch_indices):
            self.by_class.setdefault(ci, []).append(i)
        self.classes = sorted(self.by_class.keys())
        self.max_len = max(len(v) for v in self.by_class.values())
        for ci in self.classes:
            rnd = random.Random(seed + ci)
            lst = self.by_class[ci]
            while len(lst) < self.max_len:
                lst.append(rnd.choice(lst))
            rnd.shuffle(lst)
        order = []
        for i in range(self.max_len):
            for ci in self.classes:
                order.append(self.by_class[ci][i])
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self):
        for idx in self.order:
            yield idx

if __name__ == "__main__":
    ds_tr = HistoDataset(root_data="/data", split="train", output_size=256, pixel_range="0_1")
    print("Classes:", ds_tr.class_counts())
    os.makedirs("artifacts", exist_ok=True)
    grid = ds_tr.vis(16)
    grid.save("artifacts/preview_train_grid.jpg")
    print("Grid -> artifacts/preview_train_grid.jpg")
