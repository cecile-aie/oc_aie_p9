# histo_dataset.py
# DataGenerator histopathologie avec préprocessing À LA VOLÉE
# - Split sans fuite (train = NCT-CRC-HE-100K, val/test = CRC-VAL-HE-7K)
# - Échantillonnage équilibré sur le train (sans réutilisation en val/test)
# - Pré-sélection qualité complète: LaplacianVar, ShannonEntropy, WhiteRatio, SaturationRatio,
#   Tenengrad, Blockiness (spatial & DCT), TissueFraction
# - Normalisation de coloration Vahadane via **torch_staintools** (si dispo)
# - Pixel range [0,1] ou normalisation ImageNet
# - Visualisation rapide

from __future__ import annotations
import os, random, math, warnings
from typing import List, Dict, Optional, Tuple, Literal

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, Sampler

# -------------------------------------------------------------
# Utilitaires image de base
# -------------------------------------------------------------

def _read_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# HSV vectorisé (rapide, sans dépendances)
def _rgb_to_hsv_np(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    maxc = np.maximum(np.maximum(r,g), b)
    minc = np.minimum(np.minimum(r,g), b)
    v = maxc
    s = np.zeros_like(v)
    nz = maxc > 0
    s[nz] = (maxc[nz] - minc[nz]) / (maxc[nz] + 1e-8)
    # hue non utilisé ici
    return None, s, v

# -------------------------------------------------------------
# Mesures qualité (inspirées du notebook fourni)
# -------------------------------------------------------------
class QualityFilter:
    """Calcul des métriques (luminance/HSV/gradients) et **évaluation** via seuils.

    Les **seuils par classe** peuvent être fournis par le DataGenerator (JSON),
    sinon on utilise des valeurs par défaut.

    Note blockiness : on calcule **deux mesures** (spatial & DCT) et le DataGenerator
    combine ensuite en un **z-score moyen** par classe (calibré par mu/sigma de classe).
    La décision privilégie `jpeg_blockiness_max` du JSON si présent.
    """
    def __init__(
        self,
        # valeurs par défaut (fallback si JSON absent/incomplet)
        lap_var_min: float = 50.0,
        entropy_min: float = 3.0,
        tenengrad_min: float = 150.0,
        white_ratio_max: float = 0.85,
        sat_ratio_max: float = 0.20,
        block_spatial_max: float = 25.0,
        block_dct_min: float = 0.40,
        block_dct_max: float = 0.98,
        tissue_fract_min: float = 0.10
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

    # --- Helpers ---
    @staticmethod
    def _to_np(img: Image.Image) -> np.ndarray:
        return np.asarray(img)

    @staticmethod
    def _luminance(rgb: np.ndarray) -> np.ndarray:
        return (0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]).astype(np.float32)

    @staticmethod
    def _variance_of_laplacian(gray: np.ndarray) -> float:
        try:
            from scipy.signal import convolve2d
            k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
            resp = convolve2d(gray, k, mode='same', boundary='symm')
            return float(resp.var())
        except Exception:
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            lap = (np.diff(gx, axis=1, prepend=gx[:, :1]) + np.diff(gy, axis=0, prepend=gy[:1, :]))
            return float(lap.var())

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist, _ = np.histogram(gray, bins=256, range=(0,255), density=True)
        p = hist[hist>0]
        return float(-(p*np.log2(p)).sum())

    @staticmethod
    def _tenengrad(gray: np.ndarray) -> float:
        try:
            from scipy.ndimage import sobel
            gx = sobel(gray, axis=1)
            gy = sobel(gray, axis=0)
        except Exception:
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        g2 = gx*gx + gy*gy
        return float(g2.mean())

    @staticmethod
    def _blockiness_spatial(gray: np.ndarray, block: int = 8) -> float:
        h, w = gray.shape
        vb = [i for i in range(block, w, block)]
        hb = [j for j in range(block, h, block)]
        if len(vb)==0 and len(hb)==0:
            return 0.0
        bdiffs = []
        for c in vb:
            if c < w:
                bdiffs.append(np.abs(gray[:, c:(c+1)] - gray[:, (c-1):c]).mean())
        for r in hb:
            if r < h:
                bdiffs.append(np.abs(gray[r:(r+1), :] - gray[(r-1):r, :]).mean())
        bmean = float(np.mean(bdiffs)) if bdiffs else 0.0
        mask = np.ones_like(gray, dtype=bool)
        mask[:, vb] = False
        mask[hb, :] = False
        intra = []
        intra.append(np.abs(gray[:, 1:] - gray[:, :-1])[mask[:,1:]].mean() if mask[:,1:].any() else 0.0)
        intra.append(np.abs(gray[1:, :] - gray[:-1, :])[mask[1:, :]].mean() if mask[1:, :].any() else 0.0)
        imean = float(np.mean(intra)) if len(intra)>0 else 0.0
        return max(0.0, bmean - imean)

    @staticmethod
    def _blockiness_dct(gray: np.ndarray, block: int = 8) -> float:
        try:
            from scipy.fftpack import dct
            def dct2(x):
                return dct(dct(x.T, norm='ortho').T, norm='ortho')
        except Exception:
            def dct2(x):
                X = np.fft.fft2(x)
                return np.real(X)
        h, w = gray.shape
        H = (h // block) * block
        W = (w // block) * block
        g = gray[:H, :W]
        g = g.reshape(H//block, block, W//block, block).transpose(0,2,1,3)
        lows = []
        totals = []
        for bi in range(g.shape[0]):
            for bj in range(g.shape[1]):
                patch = g[bi, bj]
                C = dct2(patch)
                E = (C*C)
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
        # HSV approximé
        rgb = arr.astype(np.float32)/255.0
        r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
        v = np.maximum(np.maximum(r,g), b)
        s = np.zeros_like(v); nz = v>0; s[nz] = (v[nz]-np.minimum(np.minimum(r[nz],g[nz]), b[nz]))/(v[nz]+1e-8)
        white_ratio = float(((v>0.95) & (s<0.10)).mean())
        sat_ratio   = float((s>0.90).mean())
        bsp = self._blockiness_spatial(y)
        bdc = self._blockiness_dct(y)
        tissue_mask = ((v<0.98) | (s>0.20))
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

    def check(self, metrics: Dict[str,float], thresholds: Dict[str,float]) -> bool:
        thr = {**self.defaults, **thresholds}  # JSON override des défauts
        if metrics["lap_var"] < thr["lap_var_min"]: return False
        if metrics["entropy"] < thr["entropy_min"]: return False
        if metrics["tenengrad"] < thr["tenengrad_min"]: return False
        if metrics["white_ratio"] > thr.get("white_ratio_max", 1.0): return False
        if metrics["sat_ratio"] > thr.get("sat_ratio_max", 1.0): return False
        # --- Blockiness ---
        if "jpeg_blockiness" in metrics and "jpeg_blockiness_max" in thr:
            if metrics["jpeg_blockiness"] > thr["jpeg_blockiness_max"]:
                return False
        else:
            # fallback individuel si non calibré ou seuil combiné absent
            if metrics["block_spatial"] > thr.get("block_spatial_max", thr.get("jpeg_blockiness_max", 1e9)):
                return False
            if not (thr["block_dct_min"] <= metrics["block_dct"] <= thr["block_dct_max"]):
                return False
        if metrics["tissue_fract"] < thr.get("tissue_fract_min", 0.0): return False
        return True

# -------------------------------------------------------------
# Normalisation Vahadane via torch_staintools (si dispo)
# -------------------------------------------------------------
class TorchStainNormalizer:
    """Normalisation Vahadane sur tenseurs torch via torch_staintools (si présent).
    Si non disponible ou en cas d'erreur: no-op.
    """
    def __init__(self, enable: bool = True, target_path: Optional[str] = None, device: str = "cpu") -> None:
        self.enable = enable
        self.device = device
        self._ok = False
        self._target = None
        self._normalizer = None
        if not enable:
            return
        try:
            import torch_staintools as tst  # conforme à la demande utilisateur
            self.tst = tst
            if target_path and os.path.exists(target_path):
                self._target = self._pil_to_t(img=_read_rgb(target_path)).to(device)
            self._normalizer = None  # lazy-fit
            self._ok = True
        except Exception as e:
            warnings.warn(f"torch_staintools indisponible: {e}. Normalisation désactivée.")
            self.enable = False

    @staticmethod
    def _pil_to_t(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2,0,1).float()/255.0
        return t

    @staticmethod
    def _t_to_pil(t: torch.Tensor) -> Image.Image:
        t = t.clamp(0,1)
        arr = (t.permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def normalize(self, img: Image.Image) -> Image.Image:
        if not self.enable or not self._ok:
            return img
        try:
            t = self._pil_to_t(img).to(self.device)
            if self._normalizer is None:
                # Lazy fit sur la première image si pas de target fournie
                if self._target is None:
                    self._target = t.clone()
                # API générique: certaines versions exposent VahadaneNormalizer / fit / transform
                # On tente prudemment, sinon no-op
                try:
                    norm = self.tst.VahadaneNormalizer(device=self.device)
                except Exception:
                    norm = getattr(self.tst, 'VahadaneNormalizer', None)
                if norm is None:
                    return img
                self._normalizer = norm
                # fit
                try:
                    self._normalizer.fit(self._target)
                except Exception:
                    pass
            # transform
            try:
                t_norm = self._normalizer.transform(t)
            except Exception:
                # fallback: no-op
                return img
            return self._t_to_pil(t_norm)
        except Exception:
            return img

# -------------------------------------------------------------
# Dataset principal
# -------------------------------------------------------------
class HistoDataset(Dataset):
    # ... (init as defined above)

    def _calibrate_blockiness_stats(self, max_per_class: int = 200) -> None:
        """Calcule mu/sigma par classe pour blockiness spatial et DCT.
        Échantillonne jusqu'à `max_per_class` images par classe (shuffle stable).
        """
        stats: Dict[str, Dict[str, List[float]]] = {}
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

    def set_epoch(self, epoch: int):
        """Re-génère les indices d'epoch avec une graine dérivée pour le shuffle.
        """
        rng = random.Random(self.seed + epoch)
        self._build_epoch_indices(rng)


    def __len__(self) -> int:
        return len(self._epoch_indices)

    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize(self.output_size, Image.BILINEAR)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32)
        arr = np.transpose(arr, (2,0,1)) / 255.0
        t = torch.from_numpy(arr)
        if self.pixel_range == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
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
        # 1) Tri qualité (calcul des métriques)
        metrics = self.qf.score(img)
        # --- Combine blockiness avec z-scores par classe si dispo ---
        if class_name in self.block_stats and all(k in self.block_stats[class_name] for k in ("spatial","dct")):
            mu_s, sd_s = self.block_stats[class_name]["spatial"]
            mu_d, sd_d = self.block_stats[class_name]["dct"]
            sd_s = sd_s if (sd_s not in (0.0, None) and not np.isnan(sd_s)) else 1e-8
            sd_d = sd_d if (sd_d not in (0.0, None) and not np.isnan(sd_d)) else 1e-8
            z_s = (metrics["block_spatial"] - mu_s) / sd_s
            z_d = (metrics["block_dct"] - mu_d) / sd_d
            metrics["jpeg_blockiness"] = float((z_s + z_d) / 2.0)
        # Entraînement: possibilité de remplacement limité si rejeté
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
        # 2) Normalisation Vahadane torch_staintools (si activée)
        img = self.stain.normalize(img)
        # 3) Resize + tensor
        img = self._resize(img)
        x = self._to_tensor(img)
        y = ci
        return x, y, path

    # --- Utils ---
    def class_counts(self) -> Dict[str,int]:
        return {self.idx_to_class[i]: len(p) for i,p in self.paths_by_class.items()}

    def vis(self, n: int = 16) -> Image.Image:
        # Échantillon équilibré si possible
        per_class = max(1, n // len(self.paths_by_class))
        picks: List[Tuple[int,int]] = []
        rng = random.Random(self.seed + 123)
        for ci, paths in self.paths_by_class.items():
            for _ in range(per_class):
                j = rng.randrange(len(paths))
                picks.append((ci, j))
        picks = picks[:n]
        tiles: List[Image.Image] = []
        for ci,j in picks:
            img, _ = self._load_path(ci, j)
            img = self.stain.normalize(img)
            img = self._resize(img)
            border = 2
            color = tuple(int(x) for x in (np.random.RandomState(ci).rand(3)*255))
            img = ImageOps.expand(img, border=border, fill=color)
            tiles.append(img)
        cols = int(math.ceil(math.sqrt(len(tiles))))
        rows = int(math.ceil(len(tiles)/cols))
        w,h = tiles[0].size
        canvas = Image.new('RGB', (cols*w, rows*h), (0,0,0))
        for k,t in enumerate(tiles):
            r = k // cols; c = k % cols
            canvas.paste(t, (c*w, r*h))
        return canvas

# -------------------------------------------------------------
# Sampler équilibré round-robin (optionnel pour le train)
# -------------------------------------------------------------
class BalancedRoundRobinSampler(Sampler[int]):
    def __init__(self, dataset: HistoDataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        self.by_class: Dict[int,List[int]] = {}
        for i,(ci,_) in enumerate(dataset._epoch_indices):
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

# Usage rapide (exemple):
if __name__ == "__main__":
    ds_tr = HistoDataset(root_data="/data", split="train", output_size=256, pixel_range="0_1")
    print("Classes:", ds_tr.class_counts())
    grid = ds_tr.vis(16)
    os.makedirs("artifacts", exist_ok=True)
    grid.save("artifacts/preview_train_grid.jpg")
    print("Grid -> artifacts/preview_train_grid.jpg")
