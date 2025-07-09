from pathlib import Path
import sys

import numpy as np
from tkinter import *


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psfmodel.gaussian import GaussianPSF
# from psfmodel.hst import HSTPSF


display_size = 64

PSF_TYPE = 'gaussian'
# PSF_TYPE = 'acshrc'
# PSF_TYPE = 'wfc3uvis'
# PSF_TYPE = 'wfpc2pc1'

psfobj = None

def command_refresh_psf(val):
    global psfobj

    if (psfobj is None or
        ((PSF_TYPE in ('acshrc', 'wfpc2pc1', 'wfc3uvis') and
         psfobj.subsample != var_subsample.get()))):
        if PSF_TYPE == 'gaussian':
            print((var_motiony.get(), var_motionx.get()))
            psfobj = GaussianPSF()
        elif PSF_TYPE == 'acshrc':
            psfobj = HSTPSF('ACS', 'HRC', 'F660N', 512, 512,
                            subsample=var_subsample.get()*2+1,
                            movement=(var_motiony.get(), var_motionx.get()))
        elif PSF_TYPE == 'wfc3uvis':
            psfobj = HSTPSF('WFC3', 'UVIS', 'F606W', 128, 128,
                            subsample=var_subsample.get()*2+1,
                            movement=(var_motiony.get(), var_motionx.get()),
                            aperture='UVIS2-C512C-SUB')
        elif PSF_TYPE == 'wfpc2pc1':
            psfobj = HSTPSF('WFPC2', 'PC1', 'F606W', 128, 128,
                            subsample=var_subsample.get()*2+1,
                            movement=(var_motiony.get(), var_motionx.get()),
                            aperture='UVIS2-C512C-SUB')

    psf = psfobj.eval_rect(((var_psf_ysize.get()//2)*2+1,
                            (var_psf_xsize.get()//2)*2+1),
                           (var_y.get(), var_x.get()),
                           movement=(var_motiony.get(), var_motionx.get()),
                           sigma=(var_sigmay.get(), var_sigmax.get()),
                           angle=np.radians(var_angle.get()))
    print('PSF SUM', np.sum(psf))
    psf = psf**.5 #np.log10(psf+1e-10)

#    bkgnd = astrofit.background_gradient((var_psf_xsize.get(), var_psf_ysize.get()),
#                                         var_bkgnd_bias.get(), var_bkgnd_scale.get(), var_bkgnd_angle.get())
#    psf += bkgnd

    pix_scale = canvas_size // display_size
    ctr_x = canvas_size // 2
    ctr_y = canvas_size // 2
    min_val = 0 # np.min(psf)
    max_val = np.max(psf)
    canvas.delete('rect')
    for y in range(psf.shape[0]):
        for x in range(psf.shape[1]):
            val = int(max((psf[y, x]-min_val) / (max_val-min_val) * 255, 0))
            color = '#%02x%02x%02x' % (val, val, val)
            canvas.create_rectangle(((x-psf.shape[1]//2)*pix_scale+ctr_x,
                                     (y-psf.shape[0]//2)*pix_scale+ctr_y,
                                     (x-psf.shape[1]//2+1)*pix_scale+ctr_x,
                                     (y-psf.shape[0]//2+1)*pix_scale+ctr_y),
                                    outline=color, fill=color, tags='rect')

if __name__ == "__main__":
    toplevel_frame = Frame()
    canvas_size = 512
    canvas = Canvas(toplevel_frame, width=canvas_size, height=canvas_size, bg='black',
                    cursor='crosshair')
    canvas.grid(row=0, column=0, sticky=NW)

    # Control sliders
    control_frame = Frame(toplevel_frame)

    var_x = DoubleVar()
    var_x.set(0.)
    var_y = DoubleVar()
    var_y.set(0.)
    var_sigmax = DoubleVar()
    var_sigmax.set(2.)
    var_sigmay = DoubleVar()
    var_sigmay.set(2.)
    var_angle = DoubleVar()
    var_angle.set(0.)
    var_psf_xsize = IntVar()
    var_psf_xsize.set(21)
    var_psf_ysize = IntVar()
    var_psf_ysize.set(21)
    var_subsample = IntVar()
    var_subsample.set(0)
    var_motionx = DoubleVar()
    var_motionx.set(0.)
    var_motiony = DoubleVar()
    var_motiony.set(0.)
    var_bkgnd_bias = DoubleVar()
    var_bkgnd_bias.set(0.)
    var_bkgnd_scale = DoubleVar()
    var_bkgnd_scale.set(0.)
    var_bkgnd_angle = DoubleVar()
    var_bkgnd_angle.set(0.)

    gridrow = 0

    label = Label(control_frame, text='X')
    label.grid(row=gridrow, column=0, sticky=W)
    scale_x = Scale(control_frame, orient=HORIZONTAL, from_=-5., to=5., resolution=0.01,
                    variable=var_x, command=command_refresh_psf)
    scale_x.grid(row=gridrow, column=1)
    gridrow += 1

    label = Label(control_frame, text='Y')
    label.grid(row=gridrow, column=0, sticky=W)
    scale_y = Scale(control_frame, orient=HORIZONTAL, from_=-5., to=5., resolution=0.01,
                    variable=var_y, command=command_refresh_psf)
    scale_y.grid(row=gridrow, column=1)
    gridrow += 1

    if PSF_TYPE == 'gaussian':
        label = Label(control_frame, text='SIGMA X')
        label.grid(row=gridrow, column=0, sticky=W)
        scale_sigmax = Scale(control_frame, orient=HORIZONTAL, from_=0.001, to=5., resolution=0.001,
                            variable=var_sigmax, command=command_refresh_psf)
        scale_sigmax.grid(row=gridrow, column=1)
        gridrow += 1

        label = Label(control_frame, text='SIGMA Y')
        label.grid(row=gridrow, column=0, sticky=W)
        scale_sigmay = Scale(control_frame, orient=HORIZONTAL, from_=0.001, to=5., resolution=0.001,
                            variable=var_sigmay, command=command_refresh_psf)
        scale_sigmay.grid(row=gridrow, column=1)
        gridrow += 1

        label = Label(control_frame, text='ANGLE')
        label.grid(row=gridrow, column=0, sticky=W)
        scale_sigmay = Scale(control_frame, orient=HORIZONTAL, from_=0., to=180, resolution=1,
                            variable=var_angle, command=command_refresh_psf)
        scale_sigmay.grid(row=gridrow, column=1)
        gridrow += 1

    # label = Label(control_frame, text='BKGND BIAS')
    # label.grid(row=gridrow, column=0, sticky=W)
    # scale_bkgnd_bias = Scale(control_frame, orient=HORIZONTAL, from_=0., to=1, resolution=0.001,
    #                      variable=var_bkgnd_bias, command=command_refresh_psf)
    # scale_bkgnd_bias.grid(row=gridrow, column=1)
    # gridrow += 1
    #
    # label = Label(control_frame, text='BKGND SCALE')
    # label.grid(row=gridrow, column=0, sticky=W)
    # scale_bkgnd_scale = Scale(control_frame, orient=HORIZONTAL, from_=0., to=.05, resolution=0.001,
    #                           variable=var_bkgnd_scale, command=command_refresh_psf)
    # scale_bkgnd_scale.grid(row=gridrow, column=1)
    # gridrow += 1
    #
    # label = Label(control_frame, text='BKGND ANGLE')
    # label.grid(row=gridrow, column=0, sticky=W)
    # scale_bkgnd_angle = Scale(control_frame, orient=HORIZONTAL, from_=0., to=360., resolution=1.,
    #                           variable=var_bkgnd_angle, command=command_refresh_psf)
    # scale_bkgnd_angle.grid(row=gridrow, column=1)
    # gridrow += 1

    gridrow = 0

    label = Label(control_frame, text='PSF X SIZE')
    label.grid(row=gridrow, column=2, sticky=W)
    scale_psf_xsize = Scale(control_frame, orient=HORIZONTAL, from_=1, to=101., resolution=1,
                            variable=var_psf_xsize, command=command_refresh_psf)
    scale_psf_xsize.grid(row=gridrow, column=3)
    gridrow += 1

    label = Label(control_frame, text='PSF Y SIZE')
    label.grid(row=gridrow, column=2, sticky=W)
    scale_psf_ysize = Scale(control_frame, orient=HORIZONTAL, from_=1, to=101., resolution=1,
                            variable=var_psf_ysize, command=command_refresh_psf)
    scale_psf_ysize.grid(row=gridrow, column=3)
    gridrow += 1

    label = Label(control_frame, text='SUBSAMPLE (*2+1)')
    label.grid(row=gridrow, column=2, sticky=W)
    scale_subsample = Scale(control_frame, orient=HORIZONTAL, from_=0, to=4., resolution=1,
                            variable=var_subsample, command=command_refresh_psf)
    scale_subsample.grid(row=gridrow, column=3)
    gridrow += 1

    label = Label(control_frame, text='MOTION X')
    label.grid(row=gridrow, column=2, sticky=W)
    scale_motionx = Scale(control_frame, orient=HORIZONTAL, from_=-10., to=10., resolution=.1,
                        variable=var_motionx, command=command_refresh_psf)
    scale_motionx.grid(row=gridrow, column=3)
    gridrow += 1

    label = Label(control_frame, text='MOTION Y')
    label.grid(row=gridrow, column=2, sticky=W)
    scale_motiony = Scale(control_frame, orient=HORIZONTAL, from_=-10., to=10., resolution=.1,
                        variable=var_motiony, command=command_refresh_psf)
    scale_motiony.grid(row=gridrow, column=3)
    gridrow += 1

    control_frame.grid(row=1, column=0, sticky=NW)
    toplevel_frame.pack()

    command_refresh_psf(0)

    mainloop()
