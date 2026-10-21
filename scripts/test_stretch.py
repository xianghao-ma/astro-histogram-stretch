"""Numerical invariants and real FITS/TIFF I/O checks; no PixInsight dependency."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
import tifffile

import stretch


class FormulaTests(unittest.TestCase):
    def test_identity(self):
        x = np.linspace(0, 1, 10001)
        assert_allclose(stretch.transform(x), x, rtol=0, atol=2e-16)

    def test_doc_equation_fixed_points_and_monotonicity(self):
        x = np.linspace(0, 1, 10001)
        for m in (0.001, 0.1, 0.25, 0.5, 0.75, 0.999):
            expected = (m - 1) * x / ((2 * m - 1) * x - m)
            actual = stretch.mtf(x, m)
            assert_allclose(actual, expected, rtol=3e-13, atol=3e-15)
            assert_allclose(stretch.mtf([0, m, 1], m), [0, 0.5, 1], rtol=0, atol=0)
            self.assertTrue(np.all(np.diff(actual) > 0))
            self.assertTrue(np.all(actual >= x) if m < 0.5 else True)
            self.assertTrue(np.all(actual <= x) if m > 0.5 else True)

    def test_inverse(self):
        x = np.linspace(0, 1, 10001)
        assert_allclose(stretch.mtf(stretch.mtf(x, 0.15), 0.85), x, atol=2e-15)

    def test_broadcast_rgb_parameters(self):
        x = np.array([[0.2, 0.2, 0.2]])
        assert_allclose(stretch.transform(x, m=[0.1, 0.5, 0.9]),
                        [[9 / 13, 0.2, 1 / 37]])

    def test_clipping_and_expansion(self):
        x = np.array([0, 0.1, 0.4, 0.7, 1])
        assert_allclose(stretch.transform(x, s=0.1, h=0.7), [0, 0, 0.5, 1, 1])
        assert_allclose(stretch.transform(x, low=-1, high=2), (x + 1) / 3)
        # The interior pixel 0.3 first maps to 0.25, then MTF(0.25;0.25)=0.5.
        assert_allclose(stretch.transform([0.3], s=0.1, h=0.9, m=0.25, low=-1, high=2), [0.5])

    def test_target_solver(self):
        for q in (0.001, 0.05, 0.3, 0.9):
            for t in (0.1, 0.25, 0.75):
                m = stretch.solve_midtones(q, t)
                assert_allclose(stretch.mtf(q, m), t, atol=1e-14)
        with self.assertRaises(ValueError):
            stretch.solve_midtones(0, 0.2)

    def test_degenerate_or_nonfinite_parameters_rejected(self):
        for values in ({"s": 0.5, "h": 0.5}, {"m": 0}, {"m": 1}, {"m": np.nan},
                       {"low": 0.1}, {"high": 0.5}, {"s": -0.1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stretch.transform([0.2], **values)

    def test_negative_flux_normalization_is_explicit_and_shared(self):
        image = np.array([[[-5, 0, 5], [10, np.nan, 15]]], dtype=float)
        valid = np.isfinite(image)
        with self.assertRaises(ValueError):
            stretch.normalize(image, valid, "fits", image.dtype)
        normalized, info = stretch.normalize(image, valid, "fits", image.dtype, "minmax")
        assert_allclose(normalized, (image + 5) / 20, equal_nan=True)
        self.assertEqual((info["low"], info["high"]), (-5, 15))
        self.assertEqual(sum(r["below_range"] + r["above_range"] for r in info["clipping"]), 0)

    def test_separate_clipping_counts(self):
        raw = np.array([[[-2], [0], [0.25], [0.75], [1], [2]]])
        norm, info = stretch.normalize(raw, np.isfinite(raw), "fits", raw.dtype, bounds=[0, 1])
        self.assertEqual(info["clipping"], [{"valid": 6, "below_range": 1, "above_range": 1}])
        out = stretch.transform(norm, s=0.25, h=0.75)
        counts = stretch.channel_report(norm, out, np.isfinite(norm), [0.25], [0.75])[0]
        self.assertEqual((counts["shadows_outside"], counts["highlights_outside"]), (2, 2))
        self.assertEqual((counts["output_zero_samples"], counts["output_one_samples"]), (3, 3))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        with redirect_stdout(io.StringIO()) as captured:
            stretch.main(list(map(str, args)))
        return json.loads(captured.getvalue())

    def test_fits_science_mask_wcs_units_and_no_overwrite(self):
        path = self.root / "science.fits"
        data = np.linspace(-2, 20, 120).reshape(10, 12)
        data[0, 0], data[0, 1] = np.nan, np.inf
        valid = np.ones_like(data, dtype=np.uint8)
        valid[1, 1] = 0
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [6, 5]
        wcs.wcs.crval = [120.5, -35]
        wcs.wcs.cdelt = [-1e-5, 1e-5]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header = wcs.to_header()
        header["BUNIT"] = "MJy/sr"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header, name="SCI"),
                      fits.ImageHDU(valid, name="VALID")]).writeto(path)
        original_hash = stretch.sha256(path)
        with self.assertRaises(ValueError):
            stretch.read_image(path)
        out = self.root / "output"
        self.run_cli("apply", path, "--hdu", "SCI", "--mask-hdu", "VALID",
                     "--normalize", "minmax", "--target-median", "0.25", "--outdir", out)
        report = json.loads((out / "parameters.json").read_text())
        self.assertTrue(report["input_unchanged"])
        self.assertEqual(stretch.sha256(path), original_hash)
        self.assertEqual(report["input"]["statistics"]["invalid_samples"], 3)
        self.assertTrue(report["output"]["fits_checksums_valid"])
        with fits.open(out / "stretched.fits") as hdus:
            self.assertEqual(hdus[0].header["BUNIT"], "1")
            self.assertTrue(hdus[0].header["HTDISP"])
            self.assertEqual(hdus[0].data.shape, data.shape)
            self.assertEqual(np.isnan(hdus[0].data).sum(), 3)
            self.assertEqual(hdus["VALID"].data.sum(), 117)
            expected = wcs.all_pix2world([[0, 0], [5, 5], [11, 9]], 0)
            assert_allclose(WCS(hdus[0].header).all_pix2world([[0, 0], [5, 5], [11, 9]], 0),
                            expected, rtol=0, atol=1e-12)
        with Image.open(out / "preview.png") as preview:
            self.assertEqual(preview.size, (12, 10))
            self.assertEqual((np.array(preview)[..., 3] == 0).sum(), 3)
        with self.assertRaises(SystemExit):
            self.run_cli("apply", path, "--outdir", out)

    def test_fits_rgb_axis_preserved_and_independent_curves(self):
        path = self.root / "rgb.fits"
        data = np.stack([np.linspace(0, 1, 40).reshape(5, 8)] * 3)
        fits.writeto(path, data)
        with self.assertRaises(ValueError):
            stretch.read_image(path)
        out = self.root / "rgb-output"
        self.run_cli("apply", path, "--channel-axis", "0", "--midtones", "0.1", "0.5", "0.9", "--outdir", out)
        actual = fits.getdata(out / "stretched.fits")
        self.assertEqual(actual.shape, data.shape)
        assert_allclose(actual[1], data[1])
        self.assertTrue(np.all(actual[0] >= actual[1]))
        self.assertTrue(np.all(actual[2] <= actual[1]))

    def test_uint16_tiff_normalization_and_float_output(self):
        path = self.root / "rgb.tif"
        data = np.arange(8 * 9 * 3, dtype=np.uint16).reshape(8, 9, 3) * 200
        tifffile.imwrite(path, data, photometric="rgb")
        out = self.root / "tiff-output"
        self.run_cli("apply", path, "--midtones", "0.25", "--outdir", out)
        actual = tifffile.imread(out / "stretched.tiff")
        assert_array_equal(actual, stretch.mtf(data.astype(float) / 65535, 0.25).astype(np.float32))
        self.assertEqual(actual.dtype, np.dtype("float32"))

    def test_float_tiff_invalid_samples_and_target_with_expansion(self):
        path = self.root / "float.tiff"
        data = np.linspace(0, 1, 101).reshape(1, 101).astype(np.float32)
        data[0, 1] = np.nan
        tifffile.imwrite(path, data, photometric="minisblack")
        out = self.root / "float-output"
        self.run_cli("apply", path, "--lower", "-1", "--upper", "2",
                     "--target-median", "0.6", "--outdir", out)
        record = json.loads((out / "parameters.json").read_text())
        q = record["parameter_policy"]["pre_mtf_anchor"][0]
        m = record["parameters"]["midtones"][0]
        assert_allclose(stretch.transform(q, m=m, low=-1, high=2), 0.6)
        self.assertTrue(np.isnan(tifffile.imread(out / "stretched.tiff")[0, 1]))

    def test_ambiguous_or_unsupported_data_rejected(self):
        for shape in ((4, 7, 8), (2, 3, 7, 8)):
            path = self.root / f"cube-{len(shape)}.fits"
            fits.writeto(path, np.zeros(shape))
            with self.assertRaises(ValueError):
                stretch.read_image(path, hdu="0", channel_axis=0)
        path = self.root / "alpha.tif"
        tifffile.imwrite(path, np.zeros((8, 9, 4), dtype=np.uint8), photometric="rgb")
        with self.assertRaises(ValueError):
            stretch.read_image(path)
        path = self.root / "duplicate-science.fits"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((8, 9)), name="SCI"),
                      fits.ImageHDU(np.zeros((8, 9)), name="SCI")]).writeto(path)
        with self.assertRaises(ValueError):
            stretch.read_image(path, hdu="SCI")
        self.assertEqual(stretch.read_image(path, hdu="1")[0].shape, (8, 9, 1))
        path = self.root / "constant.fits"
        fits.writeto(path, np.ones((8, 9)) * 20)
        image, valid, _, dtype, meta = stretch.read_image(path)
        with self.assertRaises(ValueError):
            stretch.normalize(image, valid, meta["format"], dtype, "minmax")
        path = self.root / "table-wcs.fits"
        header = fits.Header({"CTYPE1": "RA---TAB", "CTYPE2": "DEC--TAB"})
        fits.writeto(path, np.ones((8, 9)), header)
        with self.assertRaises(ValueError):
            stretch.read_image(path)

    def test_rgb_shared_target_parameters(self):
        path = self.root / "rgb.fits"
        data = np.linspace(0.01, 0.2, 3 * 8 * 9).reshape(3, 8, 9)
        fits.writeto(path, data)
        out = self.root / "shared-output"
        info = self.run_cli("apply", path, "--channel-axis", "0", "--target-median", "0.25", "--outdir", out)
        mids = info["parameters"]["midtones"]
        self.assertEqual(mids[0], mids[1])
        self.assertEqual(mids[1], mids[2])


if __name__ == "__main__":
    unittest.main()
