"""No-stitch 256 px lunar patch diagnostic for DAV2 and Marigold V1/V2."""

import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import torch
import torch.nn.functional as F


BASE = Path(__file__).parent
DATA = Path("/mnt/d/nac/official_rdr/expanded")
OUT = Path("/mnt/d/nac/evaluation_20260914/lunar_native256_patch_diagnostic")
PATCHES = [
    ("TRANQPIT1", 1442, 6682, "crater_field"),
    ("RIMASHARP3", 1250, 6940, "crater_rille"),
    ("FRESH1", 1416, 8444, "fresh_crater_relief"),
    ("ORIENTALE1", 868, 6612, "crater"),
    ("MESSIER3", 1658, 7422, "high_relief"),
    ("ANTONIADI", 1613, 7700, "high_relief"),
    ("ARISTPLAT1", 1557, 7177, "low_relief_control"),
    ("RANGER9", 924, 6705, "low_relief_control"),
]


def names():
    return [f"{r}_x{x}_y{y}_{kind}" for r, x, y, kind in PATCHES]


def prepare():
    image_dir = OUT / "inputs"; gt_dir = OUT / "dtm_npy"
    image_dir.mkdir(parents=True, exist_ok=True); gt_dir.mkdir(parents=True, exist_ok=True)
    for (region, x, y, kind), name in zip(PATCHES, names()):
        folder = DATA / region
        with rasterio.open(folder / f"NAC_DTM_{region}.TIF") as dtm, \
             rasterio.open(next(folder.glob("*_2M.TIF"))) as source:
            window = Window(x, y, 256, 256)
            gt = dtm.read(1, window=window, masked=True)
            with WarpedVRT(source, crs=dtm.crs, transform=dtm.transform,
                           width=dtm.width, height=dtm.height,
                           resampling=Resampling.bilinear) as ortho:
                gray = ortho.read(1, window=window)
        if (~np.ma.getmaskarray(gt)).mean() < .99:
            raise ValueError(f"Insufficient valid DTM: {name}")
        Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2)).save(image_dir / f"{name}.png")
        np.save(gt_dir / f"{name}.npy", np.where(np.ma.getmaskarray(gt), np.nan, gt.data).astype(np.float32))


def run_baselines():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(BASE / "models/depth_anything_v2_small")
    model = AutoModelForDepthEstimation.from_pretrained(BASE / "models/depth_anything_v2_small").cuda().eval()
    folder = OUT / "dav2_npy"; folder.mkdir(parents=True, exist_ok=True)
    for name in names():
        im = Image.open(OUT / "inputs" / f"{name}.png").convert("RGB")
        values = {k:v.cuda() for k,v in processor(images=im, return_tensors="pt").items()}
        with torch.inference_mode():
            pred=model(**values).predicted_depth[:,None]
            pred=F.interpolate(pred,size=(256,256),mode="bicubic",align_corners=False)
        np.save(folder/f"{name}.npy",pred[0,0].float().cpu().numpy().astype(np.float32))
    del model; torch.cuda.empty_cache()

    from diffusers import MarigoldDepthPipeline
    pipe=MarigoldDepthPipeline.from_pretrained(BASE/"models/marigold_v1_1",variant="fp16",
                                               torch_dtype=torch.float16,local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    folder=OUT/"marigold_v1_npy";folder.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(names()):
        im=Image.open(OUT/"inputs"/f"{name}.png").convert("RGB")
        result=pipe(im,num_inference_steps=4,ensemble_size=1,processing_resolution=256,
                    match_input_resolution=True,
                    generator=torch.Generator(device="cuda").manual_seed(2025+i))
        np.save(folder/f"{name}.npy",np.asarray(result.prediction[0,:,:,0],np.float32))
    del pipe;torch.cuda.empty_cache()


def run_v2():
    destination=OUT/"marigold_v2_run"
    expected=destination/"images/predictions_npy"
    if len(list(expected.glob("*.npy")))==len(PATCHES):return
    cmd=[str(Path("/home/research/code/MMDE_studio/algorithms/.venv_marigold/bin/python")),"/home/research/code/MMDE_studio/algorithms/marigold-v2/scripts/infer.py",
         "--image_dir",str(OUT/"inputs"),"--output_dir",str(destination),
         "--width","256","--height","256"]
    with (OUT/"marigold_v2.log").open("w") as log:
        subprocess.run(cmd,cwd="/home/research/code/MMDE_studio/algorithms/marigold-v2",stdout=log,stderr=subprocess.STDOUT,check=True)


def fit(raw, gt):
    valid=np.isfinite(gt)&np.isfinite(raw)
    x=raw[valid].astype(float);y=gt[valid].astype(float)
    scale,offset=np.linalg.lstsq(np.c_[x,np.ones_like(x)],y,rcond=None)[0]
    pred=scale*raw+offset;err=pred[valid]-y
    sst=np.sum((y-y.mean())**2);r2=1-np.sum(err**2)/sst
    return pred,float(scale),float(offset),float(np.mean(np.abs(err))),float(r2)


def visualize():
    methods={"DAV2":OUT/"dav2_npy","Marigold v1.1":OUT/"marigold_v1_npy",
             "Marigold V2":OUT/"marigold_v2_run/images/predictions_npy"}
    rows=[]
    figure_dir=OUT/"comparisons_png";figure_dir.mkdir(parents=True,exist_ok=True)
    for (region,x,y,kind),name in zip(PATCHES,names()):
        rgb=np.asarray(Image.open(OUT/"inputs"/f"{name}.png"))[:,:,0]
        gt=np.load(OUT/"dtm_npy"/f"{name}.npy")
        raw={m:np.squeeze(np.load(p/f"{name}.npy")) for m,p in methods.items()}
        aligned={}; stats={}
        for method,a in raw.items():
            pred,scale,offset,mae,r2=fit(a,gt);aligned[method]=pred;stats[method]=(mae,r2)
            rows.append(dict(sample=name,region=region,x=x,y=y,type=kind,method=method,
                             oracle_scale=scale,oracle_offset=offset,oracle_mae_m=mae,oracle_r2=r2))
        zlo,zhi=np.nanpercentile(gt,(2,98))
        fig,axes=plt.subplots(1,8,figsize=(22,3.7),constrained_layout=True)
        axes[0].imshow(rgb,cmap="gray");axes[0].set_title("NAC 2 m/px")
        axes[1].imshow(gt,cmap="terrain",vmin=zlo,vmax=zhi);axes[1].set_title("Official DTM")
        for ax,(method,a) in zip(axes[2:5],raw.items()):
            lo,hi=np.percentile(a,(2,98));ax.imshow(a,cmap="Spectral",vmin=lo,vmax=hi)
            ax.set_title(f"{method}\nraw relative depth")
        for ax,(method,a) in zip(axes[5:],aligned.items()):
            mae,r2=stats[method];ax.imshow(a,cmap="terrain",vmin=zlo,vmax=zhi)
            ax.set_title(f"{method} aligned\nMAE {mae:.2f} m · R² {r2:.3f}")
        for ax in axes:ax.axis("off")
        fig.suptitle(f"{region} · {kind} · exact paired 256×256 patch · no stitching")
        fig.savefig(figure_dir/f"{name}.png",dpi=180,facecolor="white");plt.close(fig)
    with (OUT/"patch_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True);prepare();run_baselines();run_v2();visualize();print(OUT)


if __name__=="__main__":main()
