################################################################################
# UNIT TESTS
################################################################################

import unittest

import numpy as np
import numpy.ma as ma

from psfmodel import PSF


class Test_PSF(unittest.TestCase):

    def runTest(self):

        #
        # Background gradient fit
        #
        img = np.zeros((5,5))
        img[:] = 3*(np.arange(5.)-2)**2+2*(np.arange(5.)-2)
        bkgnd_params, img_mask = PSF.background_gradient_fit(img)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 0)
        img2 = PSF.background_gradient((5,5), bkgnd_params)
        self.assertTrue(np.sum((img-img2)**2) < 1e-10)

        bkgnd_params, img_mask = PSF.background_gradient_fit(img, order=3)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0,0,0,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 0)
        img2 = PSF.background_gradient((5,5), bkgnd_params)
        self.assertTrue(np.sum((img-img2)**2) < 1e-10)

        # All masked
        img = np.zeros((5,5)).view(ma.MaskedArray)
        img[:,:] = ma.masked
        bkgnd_params, img_mask = PSF.background_gradient_fit(img)
        self.assertIsNone(bkgnd_params)
        self.assertIsNone(img_mask)

        # Ignore center
        img = np.zeros((5,5)).view(ma.MaskedArray)
        img[:] = 3*(np.arange(5.)-2)**2+2*(np.arange(5.)-2)
        img[2,2] = 1000
        bkgnd_params, img_mask = PSF.background_gradient_fit(img)
        self.assertFalse(np.sum((np.array(bkgnd_params)-
                                 np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 0)
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=1)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 9)
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=2)
        self.assertIsNone(bkgnd_params)
        self.assertIsNone(img_mask)
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, ignore_center=3)
        self.assertIsNone(bkgnd_params)
        self.assertIsNone(img_mask)

        # Removal of bad pixels
        img = np.zeros((5,5)).view(ma.MaskedArray)
        img[:] = 3*(np.arange(5.)-2)**2+2*(np.arange(5.)-2)
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=5)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 0)
        img[2,2] = 10000
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 2)
        img[0,0] = 100000
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 2)
        img[0,4] = 10000
        bkgnd_params, img_mask = PSF.background_gradient_fit(img, num_sigma=3)
        self.assertTrue(np.sum((np.array(bkgnd_params)-
                                np.array([0,2,0,3,0,0]))**2) < 1e-10)
        self.assertEqual(np.sum(img_mask), 4)
