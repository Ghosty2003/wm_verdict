"""Minimal plotting layer for the paper figures, with two interchangeable backends.

  matplotlib (preferred; vector PDF)  -- used when `import matplotlib` works
  Pillow     (fallback; raster PNG and a PDF that embeds the same raster) -- used otherwise, or with PLOT_BACKEND=pil

Both backends draw the same primitives from the same calls, so the figure scripts contain no backend-specific code.
Coordinates: Figure.axes(rect=(left, bottom, width, height)) in figure fractions; Axes primitives in data units.
Sizes are in points (1/72 inch).
"""
import os

BACKEND = os.environ.get("PLOT_BACKEND", "")
if BACKEND != "pil":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        BACKEND = "matplotlib"
    except ImportError:
        BACKEND = "pil"

FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
FONT_BOLD = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]


def backend_note():
    return BACKEND


# ----------------------------------------------------------------------------------------------------------------------
class _MplAxes:
    def __init__(self, fig, rect, xlim, ylim, xlabel, ylabel, title, xticks, yticks, frame, grid, fontsize):
        ax = fig.add_axes(rect)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if frame:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("#52514e")
            ax.tick_params(labelsize=fontsize - 1, colors="#52514e")
            if xticks is not None:
                ax.set_xticks(xticks)
            if yticks is not None:
                ax.set_yticks(yticks)
            if grid:
                ax.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
                ax.set_axisbelow(True)
        else:
            ax.axis("off")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fontsize, color="#0b0b0b")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize, color="#0b0b0b")
        if title:
            ax.set_title(title, fontsize=fontsize + 1, color="#0b0b0b", loc="left")
        self.ax = ax

    def scatter(self, x, y, color, size=5, marker="o", filled=True, alpha=1.0, edge=None, edgewidth=0.6, z=3):
        kw = dict(s=size ** 2, marker=marker, alpha=alpha, zorder=z, linewidths=edgewidth)
        if filled:
            kw.update(c=color, edgecolors=edge or "white")
        else:
            kw.update(facecolors="none", edgecolors=color)
        self.ax.scatter(x, y, **kw)

    def line(self, x, y, color, width=1.0, dash=False, z=2):
        self.ax.plot(x, y, color=color, linewidth=width, linestyle=(0, (4, 3)) if dash else "-", zorder=z)

    def rect(self, x0, y0, x1, y1, face, edge=None, width=0.8, hatch=False, dash=False, z=2):
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face, edgecolor=edge or face, linewidth=width,
                                    hatch="////" if hatch else None, linestyle=(0, (3, 2)) if dash else "-", zorder=z))

    def text(self, x, y, s, size=8, color="#0b0b0b", ha="left", va="center", bold=False, z=5):
        self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight="bold" if bold else "normal", zorder=z)


class _MplFigure:
    def __init__(self, w_in, h_in, dpi):
        plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "hatch.color": "#8a8984",
                             "hatch.linewidth": 0.6, "svg.hashsalt": "verdictbench"})
        self.fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
        self.dpi = dpi

    def axes(self, rect, xlim, ylim, xlabel="", ylabel="", title="", xticks=None, yticks=None, frame=True, grid=True,
             fontsize=8):
        return _MplAxes(self.fig, rect, xlim, ylim, xlabel, ylabel, title, xticks, yticks, frame, grid, fontsize)

    def text(self, fx, fy, s, size=8, color="#0b0b0b", ha="left", va="center", bold=False):
        self.fig.text(fx, fy, s, fontsize=size, color=color, ha=ha, va=va, fontweight="bold" if bold else "normal")

    def save(self, pdf, png):
        self.fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
        self.fig.savefig(png, dpi=self.dpi)
        plt.close(self.fig)


# ----------------------------------------------------------------------------------------------------------------------
def _hex(c, alpha=1.0):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), int(round(255 * alpha)))


class _PilAxes:
    def __init__(self, fig, rect, xlim, ylim, xlabel, ylabel, title, xticks, yticks, frame, grid, fontsize):
        self.f = fig
        W, H = fig.W, fig.H
        l, b, w, h = rect
        self.px0, self.px1 = l * W, (l + w) * W
        self.py0, self.py1 = (1 - b) * H, (1 - b - h) * H          # bottom, top in pixels
        self.xlim, self.ylim = xlim, ylim
        if frame:
            if grid:
                for t in (xticks or []):
                    self.line([t, t], list(ylim), "#e4e3df", 0.6, z=0)
                for t in (yticks or []):
                    self.line(list(xlim), [t, t], "#e4e3df", 0.6, z=0)
            ax_c = "#52514e"
            fig.draw_line([(self.px0, self.py0), (self.px1, self.py0)], ax_c, 0.8, z=1)
            fig.draw_line([(self.px0, self.py0), (self.px0, self.py1)], ax_c, 0.8, z=1)
            tick = fig.pt(3)
            for t in (xticks or []):
                X, _ = self.tr(t, ylim[0])
                fig.draw_line([(X, self.py0), (X, self.py0 + tick)], ax_c, 0.8, z=1)
                fig.draw_text(X, self.py0 + tick + fig.pt(1.5), _fmt(t), fontsize - 1, ax_c, "center", "top")
            for t in (yticks or []):
                _, Y = self.tr(xlim[0], t)
                fig.draw_line([(self.px0 - tick, Y), (self.px0, Y)], ax_c, 0.8, z=1)
                fig.draw_text(self.px0 - tick - fig.pt(2), Y, _fmt(t), fontsize - 1, ax_c, "right", "center")
        if xlabel:
            fig.draw_text((self.px0 + self.px1) / 2, self.py0 + fig.pt(fontsize + 9), xlabel, fontsize, "#0b0b0b", "center", "top")
        if ylabel:
            fig.draw_text(self.px0 - fig.pt(fontsize + 16), (self.py0 + self.py1) / 2, ylabel, fontsize, "#0b0b0b",
                          "center", "center", rotate=90)
        if title:
            fig.draw_text(self.px0, self.py1 - fig.pt(4), title, fontsize + 1, "#0b0b0b", "left", "bottom")

    def tr(self, x, y):
        X = self.px0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.px1 - self.px0)
        Y = self.py0 + (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.py1 - self.py0)
        return X, Y

    def scatter(self, x, y, color, size=5, marker="o", filled=True, alpha=1.0, edge=None, edgewidth=0.6, z=3):
        r = self.f.pt(size) / 2
        for a, b in zip(x, y):
            X, Y = self.tr(a, b)
            self.f.draw_marker(X, Y, r, marker, color if filled else None, (edge or "#ffffff") if filled else color,
                               edgewidth, alpha, z)

    def line(self, x, y, color, width=1.0, dash=False, z=2):
        self.f.draw_line([self.tr(a, b) for a, b in zip(x, y)], color, width, dash=dash, z=z)

    def rect(self, x0, y0, x1, y1, face, edge=None, width=0.8, hatch=False, dash=False, z=2):
        X0, Y0 = self.tr(x0, y0)
        X1, Y1 = self.tr(x1, y1)
        self.f.draw_rect(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1), face, edge or face, width, hatch, dash, z)

    def text(self, x, y, s, size=8, color="#0b0b0b", ha="left", va="center", bold=False, z=5):
        X, Y = self.tr(x, y)
        self.f.draw_text(X, Y, s, size, color, ha, va, bold=bold, z=z)


def _paste(layer, im, x, y):
    """alpha_composite that tolerates images partly outside the canvas."""
    if x < 0 or y < 0:
        im = im.crop((max(0, -x), max(0, -y), im.size[0], im.size[1]))
        x, y = max(0, x), max(0, y)
    if im.size[0] > 0 and im.size[1] > 0 and x < layer.size[0] and y < layer.size[1]:
        layer.alpha_composite(im, (x, y))


def _fmt(t):
    return f"{t:g}"


class _PilFigure:
    def __init__(self, w_in, h_in, dpi):
        from PIL import Image, ImageDraw, ImageFont
        self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
        self.dpi = dpi
        self.W, self.H = int(round(w_in * dpi)), int(round(h_in * dpi))
        self.ops = []                       # (z, seq, fn) drawn in z order at save time
        self._fonts = {}

    def pt(self, v):
        return v * self.dpi / 72.0

    def _font(self, size, bold):
        key = (round(self.pt(size)), bold)
        if key not in self._fonts:
            f = None
            for p in (FONT_BOLD if bold else FONT_CANDIDATES):
                if os.path.exists(p):
                    f = self.ImageFont.truetype(p, key[0])
                    break
            self._fonts[key] = f or self.ImageFont.load_default(size=key[0])
        return self._fonts[key]

    def _add(self, z, fn):
        self.ops.append((z, len(self.ops), fn))

    def axes(self, rect, xlim, ylim, xlabel="", ylabel="", title="", xticks=None, yticks=None, frame=True, grid=True,
             fontsize=8):
        return _PilAxes(self, rect, xlim, ylim, xlabel, ylabel, title, xticks, yticks, frame, grid, fontsize)

    def text(self, fx, fy, s, size=8, color="#0b0b0b", ha="left", va="center", bold=False):
        self.draw_text(fx * self.W, (1 - fy) * self.H, s, size, color, ha, va, bold=bold)

    # primitives in pixel space -----------------------------------------------------------------------------------------
    def _dashed(self, d, pts, color, w):
        on, off = self.pt(4), self.pt(3)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            s = 0.0
            while s < L:
                e = min(L, s + on)
                d.line([(x0 + (x1 - x0) * s / L, y0 + (y1 - y0) * s / L), (x0 + (x1 - x0) * e / L, y0 + (y1 - y0) * e / L)],
                       fill=_hex(color), width=w)
                s = e + off

    def draw_line(self, pts, color, width, dash=False, z=2):
        w = max(1, int(round(self.pt(width))))

        def fn(layer):
            d = self.ImageDraw.Draw(layer)
            if dash:
                self._dashed(d, pts, color, w)
            else:
                d.line(pts, fill=_hex(color), width=w)
        self._add(z, fn)

    def draw_marker(self, X, Y, r, marker, face, edge, edgewidth, alpha, z):
        w = max(1, int(round(self.pt(edgewidth))))

        def fn(layer):
            R = int(r * 1.2) + w + 2
            ov = self.Image.new("RGBA", (2 * R + 1, 2 * R + 1), (0, 0, 0, 0))
            d = self.ImageDraw.Draw(ov)
            f = _hex(face, alpha) if face else None
            e = _hex(edge, alpha) if edge else None
            cx, cy = R + (X - int(X)), R + (Y - int(Y))                 # keep sub-pixel position
            if marker == "o":
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=f, outline=e, width=w)
            elif marker == "s":
                d.rectangle([cx - r * 0.9, cy - r * 0.9, cx + r * 0.9, cy + r * 0.9], fill=f, outline=e, width=w)
            else:   # "^"
                d.polygon([(cx, cy - r * 1.1), (cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8)], fill=f, outline=e, width=w)
            _paste(layer, ov, int(X) - R, int(Y) - R)
        self._add(z, fn)

    def draw_rect(self, x0, y0, x1, y1, face, edge, width, hatch, dash, z):
        w = max(1, int(round(self.pt(width))))

        def fn(layer):
            d = self.ImageDraw.Draw(layer)
            d.rectangle([x0, y0, x1, y1], fill=_hex(face))
            if hatch:                                   # 45-degree lines clipped to the rectangle
                step = self.pt(3.5)
                k = -(y1 - y0)
                while k < (x1 - x0):
                    t0, t1 = max(0.0, -k), min(y1 - y0, x1 - x0 - k)
                    if t0 < t1:
                        d.line([(x0 + k + t0, y1 - t0), (x0 + k + t1, y1 - t1)], fill=_hex("#8a8984"), width=1)
                    k += step
            if dash:
                self._dashed(d, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], edge, w)
            else:
                d.rectangle([x0, y0, x1, y1], outline=_hex(edge), width=w)
        self._add(z, fn)

    def draw_text(self, X, Y, s, size, color, ha="left", va="center", rotate=0, bold=False, z=5):
        def fn(layer):
            font = self._font(size, bold)
            l, t, r, b = font.getbbox(s)
            tw, th = r - l, b - t
            im = self.Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
            self.ImageDraw.Draw(im).text((2 - l, 2 - t), s, font=font, fill=_hex(color))
            if rotate:
                im = im.rotate(rotate, expand=True)
            w, h = im.size
            x = X - (w / 2 if ha == "center" else w if ha == "right" else 0)
            y = Y - (h / 2 if va == "center" else h if va == "bottom" else 0)
            _paste(layer, im, int(round(x)), int(round(y)))
        self._add(z, fn)

    def save(self, pdf, png):
        layer = self.Image.new("RGBA", (self.W, self.H), (255, 255, 255, 255))
        for _, _, fn in sorted(self.ops, key=lambda o: (o[0], o[1])):
            fn(layer)
        rgb = layer.convert("RGB")
        rgb.save(png, dpi=(self.dpi, self.dpi))
        rgb.save(pdf, "PDF", resolution=self.dpi)


def figure(w_in, h_in, dpi=300):
    return _MplFigure(w_in, h_in, dpi) if BACKEND == "matplotlib" else _PilFigure(w_in, h_in, dpi)
