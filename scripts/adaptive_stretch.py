#!/usr/bin/env python3
"""Shared RGB histogram stretch anchored on structure, with a background cap."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import median_filter, maximum_filter, label
from scipy.optimize import brentq
import stretch


def select_samples(image, valid, detection_channels=None, structure_sigma=5., min_area=9):
    channels=image.shape[-1]
    selected=list(range(channels)) if detection_channels is None else list(detection_channels)
    if not selected or len(set(selected))!=len(selected) or any(c<0 or c>=channels for c in selected):
        raise ValueError('Detection channel indices must be unique and in range.')
    if not np.isfinite(structure_sigma) or structure_sigma<=0 or min_area<1:
        raise ValueError('Require positive structure threshold and component size.')
    complete=np.all(valid,axis=-1)
    if complete.sum()<100: raise ValueError('Fewer than 100 complete valid pixels.')
    bg=[];noise=[];dark=[];bright=[]
    footprint=np.ones((3,3),bool);footprint[1,1]=False
    for c in range(channels):
        a=image[...,c]
        _,b,s=sigma_clipped_stats(a[complete],sigma=3,maxiters=15)
        if not np.isfinite([b,s]).all() or s<=0:
            raise ValueError('Background scale is zero or nonfinite; provide manual parameters.')
        filled=np.where(complete,a,b)
        local=median_filter(filled,size=5,mode='reflect')
        neighbor=maximum_filter(filled,footprint=footprint,mode='reflect')
        dark.append(complete & (a<b-5*s) & (abs(local-b)<3*s))
        bright.append(complete & (a>b+12*s) & (neighbor<b+3*s) & (a-local>12*s))
        bg.append(float(b));noise.append(float(s))
    dark=np.stack(dark,axis=-1);bright=np.stack(bright,axis=-1)
    usable=complete & ~np.any(dark|bright,axis=-1)
    z=(image-np.array(bg))/np.array(noise)
    background=usable & np.all(abs(z[...,selected])<2,axis=-1)
    candidate=usable & (np.max(z[...,selected],axis=-1)>structure_sigma)
    components,_=label(candidate,structure=np.ones((3,3)))
    counts=np.bincount(components.ravel());counts[0]=0
    structure=candidate & (counts[components]>=min_area)
    if background.sum()<100 or structure.sum()<30:
        raise ValueError('Insufficient background or structure samples; inspect masks or use manual parameters.')
    kept=np.broadcast_to(complete[...,None],image.shape)&~(dark|bright)
    white=float(image[kept].max())
    retained_low=float(image[kept].min())
    intensity=np.mean(image,axis=-1)
    _,joint_bg,joint_sigma=sigma_clipped_stats(intensity[complete],sigma=3,maxiters=15)
    # Do not claim the image has empty sky merely because this estimator converges.
    start=max(retained_low,float(joint_bg-3*joint_sigma))
    info={'background_RGB':bg,'sigma_RGB':noise,'detection_channels':selected,
          'structure_sigma':float(structure_sigma),'minimum_component_pixels':int(min_area),
          'dark_candidates_RGB':dark.sum(axis=(0,1)).tolist(),
          'bright_candidates_RGB':bright.sum(axis=(0,1)).tolist(),
          'background_pixels':int(background.sum()),'structure_pixels':int(structure.sum()),
          'joint_background':float(joint_bg),'joint_sigma':float(joint_sigma),
          'retained_low':retained_low,'shared_initial_black':start,'shared_white':white}
    return background,structure,dark,bright,info


def solve_black(image, background, structure, initial, white, cap=.1):
    if not np.isfinite(cap) or not 0<cap<.5:
        raise ValueError('Background cap must be inside (0,0.5).')
    bg=image[background];signal=image[structure]
    def evaluate(black):
        if black>=white: raise ValueError('Black must be less than white.')
        signal_u=np.clip((signal-black)/(white-black),0,1)
        m=float(np.median(np.mean(signal_u,axis=-1)))
        if not 0<m<1: raise ValueError('Structure anchor degenerated under clipping.')
        values=stretch.mtf(np.clip((bg-black)/(white-black),0,1),m)
        return float(np.median(values.mean(axis=-1))),m,float(np.median((np.rint(values*255)/255).mean(axis=-1)))
    target=cap-min(.001,cap*.1)
    upper=min(float(np.max(bg)),np.nextafter(white,-np.inf))
    for attempt in range(4):
        first=evaluate(initial)
        if first[0]<=target:
            black=initial
        else:
            bracket=None;left=initial
            for right in np.linspace(initial,upper,129)[1:]:
                try: value=evaluate(float(right))[0]
                except ValueError: break
                if value<=target:
                    bracket=(left,float(right));break
                left=float(right)
            if bracket is None:
                raise ValueError('Cannot meet cap while retaining a nondegenerate structure anchor.')
            black=float(brentq(lambda x:evaluate(x)[0]-target,*bracket,xtol=1e-13))
        measured,m,png=evaluate(black)
        if measured<cap and png<cap:
            return black,m,{'background_cap':cap,'numerical_target':target,
                            'background_median_float':measured,'background_median_png':png,
                            'solver_attempts':attempt+1}
        target=max(0.,target-1/255)
    raise ValueError('PNG quantization failed background cap verification.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('--outdir',required=True)
    p.add_argument('--hdu');p.add_argument('--mask-hdu');p.add_argument('--channel-axis',type=int)
    p.add_argument('--detection-channels',type=int,nargs='+')
    p.add_argument('--structure-sigma',type=float,default=5)
    p.add_argument('--min-area',type=int,default=9)
    p.add_argument('--background-cap',type=float,default=.1)
    p.add_argument('--midtones-factor',type=float,default=2.,help='Multiply the solved structure anchor after fixing black/white points (default: 2).')
    p.add_argument('--origin',choices=['lower','upper'])
    args=p.parse_args(argv)
    path=Path(args.input).expanduser().resolve();out=Path(args.outdir).expanduser().resolve()
    if out.exists(): raise ValueError('Output directory exists; select a new directory.')
    digest=stretch.sha256(path)
    image,valid,header,_,meta=stretch.read_image(path,args.hdu,args.channel_axis,args.mask_hdu)
    bg,signal,dark,bright,info=select_samples(image,valid,args.detection_channels,args.structure_sigma,args.min_area)
    black,m,check=solve_black(image,bg,signal,info['shared_initial_black'],info['shared_white'],args.background_cap)
    base_m=m
    m=base_m*args.midtones_factor
    if not np.isfinite(args.midtones_factor) or args.midtones_factor<=0 or not 0<m<1:
        raise ValueError('Require a positive finite midtones factor and 0 < base_m * factor < 1; choose an explicit smaller factor.')
    values=stretch.mtf(np.clip((image[bg]-black)/(info['shared_white']-black),0,1),m)
    if not np.median(values.mean(axis=-1))<args.background_cap or not np.median((np.rint(values*255)/255).mean(axis=-1))<args.background_cap:
        raise ValueError('Requested midtones factor violates the background cap; choose a larger factor.')
    rawlo=float(image[valid].min());rawhi=float(image[valid].max())
    white=info['shared_white']
    if not rawlo<rawhi: raise ValueError('Constant image is unsupported.')
    s=(black-rawlo)/(rawhi-rawlo);h=(white-rawlo)/(rawhi-rawlo)
    command=[sys.executable,str(Path(__file__).with_name('stretch.py')),'apply',str(path),
             '--input-range',str(rawlo),str(rawhi),'--shadows',str(s),'--highlights',str(h),
             '--midtones',str(m),'--outdir',str(out)]
    for key in ['hdu','mask_hdu','channel_axis','origin']:
        value=getattr(args,key)
        if value is not None: command += ['--'+key.replace('_','-'),str(value)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    from astropy.io import fits
    from PIL import Image
    if meta['format']=='fits':
        stored=fits.getdata(out/'stretched.fits')
        result=stored[...,None] if stored.ndim==2 else np.moveaxis(stored,meta['channel_axis'],-1)
    else:
        import tifffile
        stored=tifffile.imread(out/'stretched.tiff')
        result=stored[...,None] if stored.ndim==2 else stored
    measured=float(np.median(result[bg].mean(axis=-1)))
    preview=np.asarray(Image.open(out/'preview.png').convert('RGB'),dtype=float)/255
    origin=args.origin or ('lower' if meta['format']=='fits' else 'upper')
    png_bg=np.flipud(bg) if origin=='lower' else bg
    png_measured=float(np.median(preview[png_bg].mean(axis=-1)))
    if not measured<args.background_cap or not png_measured<args.background_cap:
        raise RuntimeError('Saved outputs fail background cap; inspect output before delivery.')
    if stretch.sha256(path)!=digest: raise RuntimeError('Input changed during processing.')
    np.savez_compressed(out/'selection_masks.npz',background=bg,structure=signal,dark=dark,bright=bright)
    record={**info,**check,'source_sha256':digest,'source_unchanged':True,
            'base_midtones':base_m,'midtones_factor':args.midtones_factor,
            'shared_black':black,'shared_midtones':m,'HT_shadows':s,'HT_highlights':h,
            'background_median_saved_float':measured,'background_median_saved_png':png_measured,
            'structure_output_median':float(np.median(result[signal].mean(axis=-1))),
            'background_zero_fraction_RGB':np.mean(result[bg]==0,axis=0).tolist(),
            'mask_axes':'background/structure: YX; dark/bright: YXC; source array spatial orientation',
            'interpretation':'base_midtones is the median of mean-RGB structure intensity after black/white rescaling; final m = base_midtones * midtones_factor (default 2), with fixed black/white points. Solver background metrics describe the base curve; saved metrics describe the final curve.',
            'command':command}
    (out/'adaptive_parameters.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as error:
        raise SystemExit(f'Error: {error}')
