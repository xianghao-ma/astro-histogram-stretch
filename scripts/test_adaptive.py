import unittest
import numpy as np
from adaptive_stretch import select_samples,solve_black

class AdaptiveTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42)
        image=rng.normal(0,.02,(128,128,3))
        image[40:80,40:80,:]+=.3
        image[10,10,:]=-3
        image[115,115,:]=10
        return image,np.ones(image.shape,bool)

    def test_masks_exclude_outliers_and_find_structure(self):
        image,valid=self.fixture()
        bg,signal,dark,bright,info=select_samples(image,valid)
        self.assertTrue(dark[10,10].all())
        self.assertTrue(bright[115,115].all())
        self.assertFalse(signal[10,10] or signal[115,115])
        self.assertTrue(signal[60,60])
        self.assertLess(info['shared_white'],1)
        self.assertFalse(np.any(bg & signal))

    def test_shared_cap_and_structure_anchor(self):
        image,valid=self.fixture()
        bg,signal,_,_,info=select_samples(image,valid)
        black,m,check=solve_black(image,bg,signal,info['shared_initial_black'],info['shared_white'])
        self.assertLess(check['background_median_float'],.1)
        self.assertLess(check['background_median_png'],.1)
        u=np.clip((image[signal]-black)/(info['shared_white']-black),0,1)
        self.assertAlmostEqual(m,np.median(u.mean(axis=-1)),places=14)

    def test_grayscale(self):
        image,valid=self.fixture();image=image[...,:1];valid=valid[...,:1]
        bg,signal,_,_,info=select_samples(image,valid)
        _,_,check=solve_black(image,bg,signal,info['shared_initial_black'],info['shared_white'])
        self.assertLess(check['background_median_float'],.1)

    def test_degenerate_background_and_bad_channels(self):
        image,valid=self.fixture()
        with self.assertRaises(ValueError):select_samples(image,valid,[3])
        with self.assertRaises(ValueError):select_samples(np.ones_like(image),valid)
        with self.assertRaises(ValueError):select_samples(image,np.zeros_like(valid))

    def test_invalid_cap(self):
        image,valid=self.fixture();bg,signal,_,_,info=select_samples(image,valid)
        for cap in [0,.5,np.nan]:
            with self.assertRaises(ValueError):
                solve_black(image,bg,signal,info['shared_initial_black'],info['shared_white'],cap)
