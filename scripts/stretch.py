#!/usr/bin/env python3
"""Document-based HistogramTransformation for FITS/TIFF display products."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


def mtf(x, m):
    """Equivalent to equation [1], for finite 0 < m < 1."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if not np.all(np.isfinite(m) & (m > 0) & (m < 1)):
        raise ValueError("Midtones must satisfy 0 < m < 1.")
    num = (1 - m) * x
    out = num / (num + m * (1 - x))
    out = np.where(x == 0, 0.0, np.where(x == 1, 1.0, out))
    return out


def validate_parameters(s, h, m, low, high):
    s, h, m, low, high = np.broadcast_arrays(s, h, m, low, high)
    params = np.asarray([s, h, m, low, high], dtype=float)
    if not np.all(np.isfinite(params)):
        raise ValueError("All transformation parameters must be finite.")
    if np.any((s < 0) | (h > 1) | (s >= h)):
        raise ValueError("Require 0 <= shadows < highlights <= 1.")
    if np.any((m <= 0) | (m >= 1)):
        raise ValueError("Require 0 < midtones < 1.")
    if np.any((low > 0) | (high < 1)) or not np.all(np.isfinite(high - low)):
        raise ValueError("Require lower <= 0, upper >= 1 and finite range width.")


def transform(x, s=0.0, h=1.0, m=0.5, low=0.0, high=1.0):
    """Clip, apply MTF, then rescale the expanded dynamic range."""
    s, h, m, low, high = (np.asarray(v, dtype=float) for v in (s, h, m, low, high))
    validate_parameters(s, h, m, low, high)
    u = np.clip((np.asarray(x, dtype=np.float64) - s) / (h - s), 0, 1)
    return (mtf(u, m) - low) / (high - low)


def solve_midtones(q, target):
    q, target = np.asarray(q, dtype=float), np.asarray(target, dtype=float)
    if not np.all(np.isfinite(q) & np.isfinite(target) &
                  (q > 0) & (q < 1) & (target > 0) & (target < 1)):
        raise ValueError("Median anchor and pre-expansion target must be inside (0,1).")
    return q * (1 - target) / (q + target - 2 * q * target)


def stats(values):
    a = np.asarray(values)
    if a.dtype.kind not in "uif":
        return {"dtype": str(a.dtype), "shape": list(a.shape), "unsupported": True}
    good = a[np.isfinite(a)]
    result = {"shape": list(a.shape), "dtype": str(a.dtype),
              "finite_samples": int(good.size), "invalid_samples": int(a.size - good.size)}
    if good.size:
        result.update(min=float(good.min()), median=float(np.median(good)), max=float(good.max()))
    return result


def file_kind(path):
    name = path.name.lower()
    if name.endswith((".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fits.fz", ".fz")):
        return "fits"
    if name.endswith((".tif", ".tiff")):
        return "tiff"
    raise ValueError("Supported inputs: FITS and TIFF.")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hdu_key(value):
    return int(value) if value.lstrip("-").isdigit() else value


def select_hdu(hdus, value):
    key = hdu_key(value)
    if isinstance(key, str):
        matches = [i for i, hdu in enumerate(hdus) if hdu.name.upper() == key.upper()]
        if len(matches) != 1:
            raise ValueError(f"HDU name {key!r} must be unique; select an HDU index instead.")
        return matches[0]
    return hdus.index_of(key)


def inspect_file(path):
    kind = file_kind(path)
    report = {"path": str(path), "format": kind}
    if kind == "fits":
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdus:
            report["hdus"] = [
                {"index": i, "name": hdu.name, "type": type(hdu).__name__,
                 "unit": hdu.header.get("BUNIT"), "display_product": hdu.header.get("HTDISP", False),
                 "statistics": stats(hdu.data) if hdu.data is not None else None}
                for i, hdu in enumerate(hdus)
            ]
    else:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            report["pages"] = len(tf.pages)
            report["series"] = [{"index": i, "axes": series.axes, "shape": list(series.shape),
                                 "statistics": stats(series.asarray())}
                                for i, series in enumerate(tf.series)]
    return report


def read_image(path, hdu=None, channel_axis=None, mask_hdu=None):
    kind, header, axis_hint, mask = file_kind(path), None, None, None
    meta = {"format": kind}
    if kind == "fits":
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdus:
            choices = [i for i, item in enumerate(hdus)
                       if isinstance(item, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                       and item.data is not None and item.data.ndim in (2, 3)]
            if hdu is None:
                if len(choices) != 1:
                    raise ValueError(f"Select science image explicitly with --hdu; image HDUs: {choices}")
                selected = choices[0]
            else:
                selected = select_hdu(hdus, hdu)
            obj = hdus[selected]
            if selected not in choices:
                raise ValueError("Selected HDU is not a 2D/3D image.")
            data, header = np.array(obj.data), obj.header.copy()
            if any("-TAB" in str(header.get(key, "")) for key in header if key.startswith("CTYPE")):
                raise ValueError("Table-dependent WCS is unsupported; external WCS tables would be lost.")
            if any(key.startswith(("CPDIS", "CQDIS", "D2IMDIS", "DET2IM")) for key in header):
                raise ValueError("Lookup-table WCS distortion is unsupported; retain its external tables separately.")
            if mask_hdu is not None:
                mask = np.array(hdus[select_hdu(hdus, mask_hdu)].data)
            meta.update(hdu_index=selected, hdu_name=obj.name, original_unit=header.get("BUNIT"),
                        previously_stretched=bool(header.get("HTDISP", False)), mask_hdu=mask_hdu)
    else:
        if hdu is not None or mask_hdu is not None:
            raise ValueError("--hdu and --mask-hdu apply only to FITS.")
        import tifffile
        with tifffile.TiffFile(path) as tf:
            if len(tf.series) != 1 or len(tf.pages) != 1:
                raise ValueError("Only single-page, single-series TIFF is supported.")
            page = tf.pages[0]
            if int(page.photometric) not in (1, 2) or page.extrasamples:
                raise ValueError("TIFF must be MINISBLACK grayscale or RGB without alpha/extra samples.")
            if page.tags.get("Orientation") and page.tags["Orientation"].value != 1:
                raise ValueError("Non-default TIFF orientation must be resolved before processing.")
            series = tf.series[0]
            data = series.asarray()
            axes = series.axes
            if data.ndim == 3 and "S" in axes and int(page.photometric) == 2:
                axis_hint = axes.index("S")
            meta["tiff_axes"] = axes
    if data.dtype.kind not in "uif":
        raise ValueError("Image must contain real unsigned integer, signed integer or float samples.")
    original_shape, dtype = data.shape, data.dtype
    if data.ndim == 2:
        if channel_axis is not None:
            raise ValueError("--channel-axis is only for confirmed RGB images.")
        image = data[..., None]
        axis = None
    elif data.ndim == 3:
        axis = channel_axis if channel_axis is not None else axis_hint
        if axis is None:
            raise ValueError("3D data requires --channel-axis after confirming that it is RGB.")
        if axis not in (-3, -2, -1, 0, 1, 2):
            raise ValueError("Invalid channel axis.")
        axis %= 3
        if data.shape[axis] != 3:
            raise ValueError("Only exactly three RGB channels are supported.")
        if axis_hint is not None and axis != axis_hint:
            raise ValueError("--channel-axis conflicts with TIFF RGB sample metadata.")
        image = np.moveaxis(data, axis, -1)
    else:
        raise ValueError("Only 2D grayscale or 3D RGB images are supported.")
    image = image.astype(np.float64)
    valid = np.isfinite(image)
    if mask is not None:
        if mask.shape == original_shape:
            mask = mask[..., None] if axis is None else np.moveaxis(mask, axis, -1)
        elif mask.shape == image.shape[:2]:
            mask = mask[..., None]
        else:
            raise ValueError("Mask shape must match science data or its two spatial axes.")
        valid &= np.isfinite(mask) & (mask > 0)
    if any(not np.any(valid[..., c]) for c in range(image.shape[-1])):
        raise ValueError("Each channel must contain valid samples.")
    image[~valid] = np.nan
    meta.update(original_shape=list(original_shape), original_dtype=str(dtype), channel_axis=axis)
    return image, valid, header, dtype, meta


def normalize(image, valid, kind, dtype, mode="auto", bounds=None):
    values = image[valid]
    if bounds is not None:
        a, b = bounds
        method = "explicit"
    elif mode == "minmax":
        a, b = float(values.min()), float(values.max())
        method = "shared_valid_minmax"
    elif kind == "tiff" and dtype.kind == "u":
        a, b = 0.0, float(np.iinfo(dtype).max)
        method = "unsigned_tiff_dtype_range"
    elif values.min() >= 0 and values.max() <= 1:
        a, b = 0.0, 1.0
        method = "already_normalized"
    else:
        raise ValueError("Input is outside [0,1]; choose --input-range LOW HIGH or --normalize minmax.")
    if not np.isfinite([a, b, b - a]).all() or b <= a:
        raise ValueError("Input normalization needs finite LOW < HIGH and finite range width.")
    with np.errstate(over="ignore"):
        raw = (image - a) / (b - a)
    counts = []
    for c in range(image.shape[-1]):
        samples = image[..., c][valid[..., c]]
        counts.append({"valid": int(samples.size), "below_range": int(np.count_nonzero(samples < a)),
                       "above_range": int(np.count_nonzero(samples > b))})
    return np.clip(raw, 0, 1), {"method": method, "low": a, "high": b, "clipping": counts}


def vector(values, channels, label):
    result = np.asarray(values, dtype=float)
    if len(result) == 1:
        return np.repeat(result, channels)
    if len(result) != channels:
        raise ValueError(f"{label}: supply one value, or {channels} channel values.")
    return result


def channel_report(before, after, valid, s, h):
    report = []
    for c in range(before.shape[-1]):
        x, y = before[..., c][valid[..., c]], after[..., c][valid[..., c]]
        low, high = int(np.count_nonzero(x < s[c])), int(np.count_nonzero(x > h[c]))
        report.append({
            "channel": "K" if before.shape[-1] == 1 else "RGB"[c], "valid_samples": int(x.size),
            "shadows_outside": low, "highlights_outside": high,
            "shadows_outside_percent": 100 * low / x.size,
            "highlights_outside_percent": 100 * high / x.size,
            "output_zero_samples": int(np.count_nonzero(y == 0)),
            "output_one_samples": int(np.count_nonzero(y == 1)),
            "before": stats(x), "after": stats(y)})
    return report


def display_pixels(data, origin):
    data = data if data.shape[-1] == 3 else np.repeat(data, 3, axis=-1)
    valid = np.all(np.isfinite(data), axis=-1)
    rgb = np.rint(np.clip(np.nan_to_num(data, nan=0.0), 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([rgb, valid.astype(np.uint8) * 255])
    return np.flipud(rgba) if origin == "lower" else rgba


def write_products(outdir, normalized, result, valid, header, meta, params, origin):
    from PIL import Image
    axis = meta["channel_axis"]
    restored = result[..., 0] if axis is None else np.moveaxis(result, -1, axis)
    restored_valid = valid[..., 0] if axis is None else np.moveaxis(valid, -1, axis)
    if meta["format"] == "fits":
        from astropy.io import fits
        output = outdir / "stretched.fits"
        output_header = header.copy()
        for key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM", "DATAMIN", "DATAMAX",
                    "EXTNAME", "EXTVER", "EXTLEVEL", "BUNIT"):
            output_header.remove(key, ignore_missing=True, remove_all=True)
        output_header["BUNIT"] = ("1", "Dimensionless display values")
        output_header["HTDISP"] = (True, "Nonlinear display product; not calibrated flux")
        output_header.add_history("astro-histogram-stretch: clip -> MTF -> range expansion.")
        output_header.add_history("Original physical unit: " + str(meta.get("original_unit")))
        output_header.add_history("Parameters and provenance: parameters.json")
        hdus = fits.HDUList([fits.PrimaryHDU(restored.astype(np.float64), output_header),
                             fits.ImageHDU(restored_valid.astype(np.uint8), name="VALID")])
        hdus.writeto(output, checksum=True)
        with fits.open(output, checksum=True, memmap=False) as written:
            actual = written[0].data
            verified = bool(np.array_equal(actual, restored, equal_nan=True))
            checksums = all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in written)
    else:
        import tifffile
        output = outdir / "stretched.tiff"
        stored = result[..., 0] if axis is None else result
        tifffile.imwrite(output, stored.astype(np.float32),
                         photometric="minisblack" if axis is None else "rgb",
                         metadata={"axes": "YX" if axis is None else "YXS"},
                         description="Nonlinear dimensionless display product; see parameters.json.")
        actual = tifffile.imread(output)
        verified = bool(np.array_equal(actual, stored.astype(np.float32), equal_nan=True))
        checksums = None
    if not verified or checksums is False:
        raise RuntimeError("Output read-back validation failed.")
    Image.fromarray(display_pixels(result, origin)).save(outdir / "preview.png")

    # Plot in a disposable cache directory, without changing user configuration.
    with tempfile.TemporaryDirectory(prefix="ht-mpl-") as cache:
        old = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
            axes[0, 0].imshow(display_pixels(normalized, origin))
            axes[0, 0].set_title("Before: explicit normalized display")
            axes[0, 1].imshow(display_pixels(result, origin))
            axes[0, 1].set_title("After: HistogramTransformation")
            for ax in axes[0]:
                ax.set_axis_off()
            for c in range(result.shape[-1]):
                color = "black" if result.shape[-1] == 1 else ("red", "green", "blue")[c]
                for arr, style, label in ((normalized, "--", "Input"), (result, "-", "Output")):
                    counts, edges = np.histogram(arr[..., c][valid[..., c]], bins=1024, range=(0, 1))
                    axes[1, 0].stairs(counts, edges, color=color, linestyle=style,
                                      label=label if c == 0 else None, linewidth=0.8)
                grid = np.linspace(0, 1, 4097)
                curve = transform(grid, *(params[key][c] for key in
                                          ("shadows", "highlights", "midtones", "lower", "upper")))
                axes[1, 1].plot(grid, curve, color=color, linewidth=1.2)
            axes[1, 0].set(yscale="symlog", xlabel="Normalized sample", ylabel="Count",
                            title="Histograms: 1024 display bins")
            axes[1, 0].legend()
            axes[1, 1].plot([0, 1], [0, 1], ":", color="gray")
            axes[1, 1].set(xlabel="Input", ylabel="Output", title="Transfer curve",
                            xlim=(0, 1), ylim=(0, 1))
            fig.savefig(outdir / "diagnostics.png", dpi=140)
            plt.close(fig)
        finally:
            if old is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old
    return {"path": str(output), "sha256": sha256(output), "readback_equal": verified,
            "fits_checksums_valid": checksums,
            "dtype": "float64" if meta["format"] == "fits" else "float32"}


def apply(args):
    path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if outdir.exists():
        raise ValueError("Output directory already exists; choose a new directory.")
    source_hash = sha256(path)
    image, valid, header, dtype, meta = read_image(path, args.hdu, args.channel_axis, args.mask_hdu)
    if args.input_range is not None and args.normalize != "auto":
        raise ValueError("Use either --input-range or --normalize minmax, not both.")
    norm, normalization = normalize(image, valid, meta["format"], dtype, args.normalize, args.input_range)
    channels = image.shape[-1]
    params = {key: vector(getattr(args, key), channels, key)
              for key in ("shadows", "highlights", "lower", "upper")}
    s, h, low, high = (params[k] for k in ("shadows", "highlights", "lower", "upper"))
    mids = vector(args.midtones if args.midtones is not None else [0.5], channels, "midtones")
    validate_parameters(s, h, mids, low, high)
    policy = {"method": "explicit_midtones"}
    if args.target_median is not None:
        u = np.clip((norm - s) / (h - s), 0, 1)
        shared = all(np.all(v == v[0]) for v in params.values())
        q = np.repeat(np.median(u[valid]), channels) if shared else np.array(
            [np.median(u[..., c][valid[..., c]]) for c in range(channels)])
        mids = solve_midtones(q, low + args.target_median * (high - low))
        policy = {"method": "target_median_anchor", "target": args.target_median,
                  "shared": bool(shared), "pre_mtf_anchor": q.tolist()}
    params["midtones"] = mids
    result = transform(norm, s, h, mids, low, high)
    if not np.isfinite(result[valid]).all():
        raise ValueError("Transformation produced nonfinite valid samples.")
    origin = args.origin or ("lower" if meta["format"] == "fits" else "upper")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Nonlinear dimensionless display product; not calibrated flux",
              "algorithm": "PixInsight HistogramTransformation document, equation [1] and range expansion",
              "input": {"path": str(path), "sha256": source_hash, **meta, "statistics": stats(image)},
              "normalization": normalization,
              "parameters": {key: val.tolist() for key, val in params.items()},
              "parameter_policy": policy, "display_origin": origin,
              "channel_statistics": channel_report(norm, result, valid, s, h),
              "versions": {"python": sys.version.split()[0],
                           **{name: version(name) for name in ("numpy", "astropy", "tifffile", "Pillow", "matplotlib")}},
              "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]}
    # All data/parameter checks precede directory creation.
    outdir.mkdir(parents=True, exist_ok=False)
    report["output"] = write_products(outdir, norm, result, valid, header, meta, params, origin)
    if sha256(path) != source_hash:
        raise RuntimeError("Input file changed during processing.")
    report["input_unchanged"] = True
    (outdir / "parameters.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return {"outdir": str(outdir), "output": report["output"],
            "parameters": report["parameters"], "channel_statistics": report["channel_statistics"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="Read image metadata and valid-sample statistics.")
    inspect.add_argument("input")
    run = sub.add_parser("apply", help="Create a separate stretched display product and diagnostics.")
    run.add_argument("input")
    run.add_argument("--outdir", required=True, help="New directory; existing directories are refused.")
    run.add_argument("--hdu", help="FITS science HDU index or unique extension name.")
    run.add_argument("--mask-hdu", help="FITS validity mask: finite positive = valid. Not a DQ bit mask.")
    run.add_argument("--channel-axis", type=int, help="Required for confirmed 3D RGB FITS; e.g. 0 for (3,H,W).")
    run.add_argument("--normalize", choices=["auto", "minmax"], default="auto")
    run.add_argument("--input-range", type=float, nargs=2, metavar=("LOW", "HIGH"))
    for name, default in (("shadows", 0), ("highlights", 1), ("lower", 0), ("upper", 1)):
        run.add_argument("--" + name, type=float, nargs="+", default=[default],
                         help="One shared value or three R G B values.")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--midtones", type=float, nargs="+", help="One shared value or three R G B values; default 0.5.")
    group.add_argument("--target-median", type=float, help="Solve MTF parameter for a final median anchor; not AutoSTF.")
    run.add_argument("--origin", choices=["lower", "upper"], help="Preview orientation only.")
    args = parser.parse_args(argv)
    try:
        result = inspect_file(Path(args.input).expanduser().resolve()) if args.action == "inspect" else apply(args)
    except (ValueError, OSError, KeyError, IndexError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
