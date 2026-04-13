################################################################################
# psf_gui/main.py — interactive PSF visualization (Tkinter).
################################################################################

import logging
import tkinter as tk
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from psfmodel.gaussian import GaussianPSF

logger = logging.getLogger(__name__)

DISPLAY_SIZE = 64
DEFAULT_CANVAS_SIZE = 512

PsfTypeLiteral = Literal['gaussian', 'acshrc', 'wfc3uvis', 'wfpc2pc1']


def _hst_psf(*args: Any, **kwargs: Any) -> Any:
    """Return an ``HSTPSF`` instance.

    Mypy treats ``HSTPSF`` as abstract because ``_eval_rect`` is not annotated on
    the subclass; runtime construction is valid.
    """
    from psfmodel.hst import HSTPSF

    return HSTPSF(*args, **kwargs)  # type: ignore[abstract, no-untyped-call]


class PsfGuiApp:
    """Build sliders and a canvas to render a PSF patch from ``psfmodel``."""

    def __init__(self, *, psf_type: PsfTypeLiteral = 'gaussian') -> None:
        self._psf_type: PsfTypeLiteral = psf_type
        self._psfobj: Any = None
        self.canvas_size = DEFAULT_CANVAS_SIZE
        self._root = tk.Tk()
        self._root.title('PSF GUI')

        self._toplevel = tk.Frame(self._root)
        self.canvas = tk.Canvas(
            self._toplevel,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='crosshair',
        )
        self.canvas.grid(row=0, column=0, sticky=tk.NW)

        self.var_x = tk.DoubleVar(value=0.0)
        self.var_y = tk.DoubleVar(value=0.0)
        self.var_sigmax = tk.DoubleVar(value=2.0)
        self.var_sigmay = tk.DoubleVar(value=2.0)
        self.var_angle = tk.DoubleVar(value=0.0)
        self.var_psf_xsize = tk.IntVar(value=21)
        self.var_psf_ysize = tk.IntVar(value=21)
        self.var_subsample = tk.IntVar(value=0)
        self.var_motionx = tk.DoubleVar(value=0.0)
        self.var_motiony = tk.DoubleVar(value=0.0)

        self._build_controls()
        self._toplevel.pack()

    def _build_controls(self) -> None:
        control = tk.Frame(self._toplevel)
        gridrow = 0

        def add_scale_row_left(
            label_text: str,
            variable: tk.DoubleVar | tk.IntVar,
            *,
            from_: float,
            to: float,
            resolution: float,
        ) -> None:
            nonlocal gridrow
            lbl = tk.Label(control, text=label_text)
            lbl.grid(row=gridrow, column=0, sticky=tk.W)
            scale = tk.Scale(
                control,
                orient=tk.HORIZONTAL,
                from_=from_,
                to=to,
                resolution=resolution,
                variable=variable,
                command=lambda _s: self.refresh_psf(),
            )
            scale.grid(row=gridrow, column=1)
            gridrow += 1

        add_scale_row_left('X', self.var_x, from_=-5.0, to=5.0, resolution=0.01)
        add_scale_row_left('Y', self.var_y, from_=-5.0, to=5.0, resolution=0.01)

        if self._psf_type == 'gaussian':
            add_scale_row_left('SIGMA X', self.var_sigmax, from_=0.001, to=5.0, resolution=0.001)
            add_scale_row_left('SIGMA Y', self.var_sigmay, from_=0.001, to=5.0, resolution=0.001)
            add_scale_row_left('ANGLE', self.var_angle, from_=0.0, to=180.0, resolution=1.0)

        gridrow_right = 0
        col = 2

        def add_scale_row_right(
            label_text: str,
            variable: tk.DoubleVar | tk.IntVar,
            *,
            from_: float,
            to: float,
            resolution: float,
        ) -> None:
            nonlocal gridrow_right
            lbl = tk.Label(control, text=label_text)
            lbl.grid(row=gridrow_right, column=col, sticky=tk.W)
            scale = tk.Scale(
                control,
                orient=tk.HORIZONTAL,
                from_=from_,
                to=to,
                resolution=resolution,
                variable=variable,
                command=lambda _s: self.refresh_psf(),
            )
            scale.grid(row=gridrow_right, column=col + 1)
            gridrow_right += 1

        add_scale_row_right('PSF X SIZE', self.var_psf_xsize, from_=1.0, to=101.0, resolution=1.0)
        add_scale_row_right('PSF Y SIZE', self.var_psf_ysize, from_=1.0, to=101.0, resolution=1.0)
        add_scale_row_right(
            'SUBSAMPLE (*2+1)', self.var_subsample, from_=0.0, to=4.0, resolution=1.0
        )
        add_scale_row_right('MOTION X', self.var_motionx, from_=-10.0, to=10.0, resolution=0.1)
        add_scale_row_right('MOTION Y', self.var_motiony, from_=-10.0, to=10.0, resolution=0.1)

        control.grid(row=1, column=0, sticky=tk.NW)

    def _ensure_psf_object(self) -> None:
        subsample_val = int(self.var_subsample.get())
        motion_y = float(self.var_motiony.get())
        motion_x = float(self.var_motionx.get())
        expected_sub = subsample_val * 2 + 1

        need_rebuild = self._psfobj is None
        if not need_rebuild and self._psf_type in ('acshrc', 'wfpc2pc1', 'wfc3uvis'):
            need_rebuild = getattr(self._psfobj, 'subsample', None) != expected_sub

        if not need_rebuild:
            return

        if self._psf_type == 'gaussian':
            logger.debug('motion=(%s, %s)', motion_y, motion_x)
            self._psfobj = GaussianPSF()
            return

        if self._psf_type == 'acshrc':
            self._psfobj = _hst_psf(
                'ACS',
                'HRC',
                'F660N',
                512,
                512,
                subsample=expected_sub,
                movement=(motion_y, motion_x),
            )
        elif self._psf_type == 'wfc3uvis':
            self._psfobj = _hst_psf(
                'WFC3',
                'UVIS',
                'F606W',
                128,
                128,
                subsample=expected_sub,
                movement=(motion_y, motion_x),
                aperture='UVIS2-C512C-SUB',
            )
        elif self._psf_type == 'wfpc2pc1':
            self._psfobj = _hst_psf(
                'WFPC2',
                'PC1',
                'F606W',
                128,
                128,
                subsample=expected_sub,
                movement=(motion_y, motion_x),
            )

    def refresh_psf(self) -> None:
        """Redraw the PSF patch on the canvas from current slider values."""

        self._ensure_psf_object()
        if self._psfobj is None:
            return

        ysize = int(self.var_psf_ysize.get())
        xsize = int(self.var_psf_xsize.get())
        rect_h = (ysize // 2) * 2 + 1
        rect_w = (xsize // 2) * 2 + 1

        offset_y = float(self.var_y.get())
        offset_x = float(self.var_x.get())
        motion_y = float(self.var_motiony.get())
        motion_x = float(self.var_motionx.get())

        kwargs: dict[str, Any] = {'movement': (motion_y, motion_x)}
        if self._psf_type == 'gaussian':
            kwargs['sigma'] = (float(self.var_sigmay.get()), float(self.var_sigmax.get()))
            kwargs['angle'] = np.radians(float(self.var_angle.get()))

        raw = self._psfobj.eval_rect(
            (rect_h, rect_w),
            (offset_y, offset_x),
            **kwargs,
        )
        psf = cast(npt.NDArray[np.floating], np.asarray(raw, dtype=np.float64))
        logger.info('PSF sum=%s', float(np.sum(psf)))
        psf = np.sqrt(psf)

        pix_scale = self.canvas_size // DISPLAY_SIZE
        ctr_x = self.canvas_size // 2
        ctr_y = self.canvas_size // 2
        min_val = 0.0
        max_val = float(np.max(psf))
        denom = max_val - min_val
        if denom <= 0.0:
            denom = 1.0

        self.canvas.delete('rect')
        for y in range(psf.shape[0]):
            for x in range(psf.shape[1]):
                raw_px = (float(psf[y, x]) - min_val) / denom * 255.0
                val = int(max(raw_px, 0.0))
                color = f'#{val:02x}{val:02x}{val:02x}'
                self.canvas.create_rectangle(
                    (x - psf.shape[1] // 2) * pix_scale + ctr_x,
                    (y - psf.shape[0] // 2) * pix_scale + ctr_y,
                    (x - psf.shape[1] // 2 + 1) * pix_scale + ctr_x,
                    (y - psf.shape[0] // 2 + 1) * pix_scale + ctr_y,
                    outline=color,
                    fill=color,
                    tags='rect',
                )

    def run(self) -> None:
        """Start the Tk main loop."""

        self.refresh_psf()
        self._root.mainloop()


def main() -> None:
    """Entry point for ``psf-gui`` and ``python -m psf_gui``."""

    logging.basicConfig(level=logging.INFO)
    PsfGuiApp(psf_type='gaussian').run()
