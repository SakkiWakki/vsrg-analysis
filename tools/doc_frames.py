"""Render what the DRAWABLE DOC draws, at given chart times, as PNGs.

`tools/render_frames.py` renders the app stack, and the drawable pipeline
declines to run there ("painter is not on a GL engine") - so the one path
being migrated to was the one path that could not be looked at without
launching the GUI. This drives the doc through the RASTER executor instead,
which needs no GL context, and writes one PNG per requested time.

    python tools/doc_frames.py <chart.sm> t1 [t2 ...] [options]

    --out DIR        output directory (default ./doc_frames)
    --elements 0|1   emit storyboard elements from the doc (default 1)
    --notes MODE     'captured' (default) draws each field copy as a
                     labelled placeholder, so where the copies land is
                     visible; 'inline' takes the note-feed path, which
                     draws nothing here because no feed is attached
    --dump           also print the op stream for each time
    --backend MODE   'raster' (default) or 'gl': compose through the real
                     GLExecutor on an offscreen context

What this is and is not: the raster executor is the REFERENCE backend, not
the shipping one. It draws unshaded and unfaded (it says so, once, per run)
and skips Mesh sources. Geometry - which is what a placement bug is about -
is the same records the GL executor consumes.

`--backend gl` runs the SHIPPING backend instead, so the per-item shader tier
is finally visible offline - a rig whose whole visual is its `Frag=` (gat 2's
lumikey tunnel, monitor, horizon) draws as a plain copy on raster and can
only be judged here. Needs a GL 3.2 core context; falls back to raster with a
note when none can be made.

The field placeholder is deliberately not a real notefield: a labelled grid
makes a flipped, cropped or mis-scaled copy obvious, where real notes would
just look like notes.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen  # noqa: E402

from analysis.player.render.storyboard import record as rec  # noqa: E402
from analysis.player.render.storyboard.executor import (  # noqa: E402
    CLEAR_TRANSPARENT, SCREEN_ID, RasterExecutor)
from analysis.player.render.storyboard.pipeline import (  # noqa: E402
    _drawable_sizes_of, _lazy_images)

_SRC_NAMES = ('image', 'drawable', 'mesh', 'fill', 'lines')
_OP_NAMES = ('BEGIN', 'BLIT', 'COPY', 'END')


def field_placeholder(w: int, h: int) -> QImage:
    """A labelled grid standing in for a notefield capture.

    Asymmetric on both axes on purpose: the corner label and the single wide
    band at the top make a vertical flip, a crop or a mirrored copy obvious
    at a glance, which a symmetric grid or a real notefield would not."""
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.fillRect(QRectF(0, 0, w, h * 0.08), QColor(255, 90, 90, 200))
    painter.setPen(QPen(QColor(90, 200, 255, 160), 1.0))
    for i in range(1, 8):
        painter.drawLine(int(w * i / 8), 0, int(w * i / 8), h)
        painter.drawLine(0, int(h * i / 8), w, int(h * i / 8))
    painter.setPen(QPen(QColor(255, 255, 255, 230)))
    painter.setFont(QFont('DejaVu Sans', max(8, h // 24)))
    painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, 'FIELD')
    painter.drawText(QRectF(4, h * 0.10, w, h), Qt.AlignmentFlag.AlignLeft,
                     'top-left')
    painter.end()
    return image


def describe(u, f, index: int) -> str:
    """One record as a readable line."""
    kind = int(u[index, rec.U_KIND])
    name = _OP_NAMES[kind] if kind < len(_OP_NAMES) else f'op{kind}'
    if kind != rec.OP_BLIT:
        return f'{name} a={u[index, rec.U_A]} b={u[index, rec.U_B]}'
    src = int(u[index, rec.U_A])
    frec = f[index]
    return (f'BLIT {_SRC_NAMES[src]:>8} id={u[index, rec.U_B]:<3} '
            f'tag={u[index, rec.U_TAG]:<8} '
            f'op={frec[rec.F_OPACITY]:.2f} '
            f'size=({frec[rec.F_SIZE]:.0f},{frec[rec.F_SIZE + 1]:.0f}) '
            f'org=({frec[rec.F_ORIGIN]:.2f},{frec[rec.F_ORIGIN + 1]:.2f}) '
            f'tr=({frec[2]:.0f},{frec[5]:.0f})')


def warm_frames(t: float, count: int, step: float, since: float | None):
    """The times to run before `t` so RETAINED drawables hold real content.

    An AFT slot keeps its last capture across frames - that retention IS the
    freeze a still-frames rig relies on. Composing a single frame cold leaves
    every slot the chart captured EARLIER empty, so a sampler blitting one
    draws nothing and the frame reads as a blackout that the running app
    never shows.

    A short warm-up is NOT always enough, and a freeze is exactly the case:
    the rig gates its capture off for the whole frozen section, so the slot's
    content dates from before the section began. `since` walks from an
    explicit earlier time in `count` steps instead, which is how you reach a
    capture that a two-second warm-up runs straight past."""
    if since is not None and since < t:
        step = (t - since) / max(1, count)
        return [since + step * i for i in range(count)]
    return [t - step * i for i in range(count, 0, -1)]


def gl_executor(images, sizes, id_maps):
    """A GLExecutor on a current offscreen GL 3.2 core context, wired the way
    the pipeline wires it, or None when no context can be made.

    The GL tier is where the shader lanes actually run: the raster backend
    says so itself ("shader lane not implemented"), so a rig whose whole
    visual IS its per-item .frag - gat 2's lumikey tunnel, monitor, horizon -
    cannot be looked at on the raster path at all. Keeps the CONTEXT alive on
    the returned executor: dropping it frees every FBO mid-run."""
    from PySide6.QtGui import (QOffscreenSurface, QOpenGLContext,
                               QSurfaceFormat)
    from analysis.player.render.storyboard.gl_executor import GLExecutor

    fmt = QSurfaceFormat()
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    context = QOpenGLContext()
    context.setFormat(fmt)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not (context.create() and surface.isValid()
            and context.makeCurrent(surface)):
        return None
    executor = GLExecutor(
        images, sizes,
        image_grids=id_maps.get('image_grids'),
        image_specs=id_maps.get('image_specs'))
    # Positional by the shader id the BLIT lanes carry. Without it every
    # shader lane resolves to None and a shaded sampler blits through the
    # plain textured program - the exact failure this backend exists to see.
    executor.set_shaders(id_maps.get('shaders') or [])
    executor._offscreen = (context, surface)
    return executor


def seed_field(executor, drawable_id: int, w: int, h: int):
    """Put the labelled placeholder in a field drawable, whichever backend is
    running, and return whatever must stay alive for it to keep working.

    The two executors take a field's stand-in content differently: raster
    takes the QImage, GL takes an EXTERNAL texture id it does not own - so on
    GL the caller has to hold the QOpenGLTexture, or it is collected and the
    drawable samples a deleted name."""
    image = field_placeholder(w, h)
    if not hasattr(executor, 'set_drawable_texture'):
        executor.set_drawable_image(drawable_id, image)
        return None
    from PySide6.QtGui import QTransform
    from PySide6.QtOpenGL import QOpenGLTexture
    # GL textures are bottom-up and the executor samples an external one like
    # the drawable's own FBO texture, so the top-down QImage flips first.
    texture = QOpenGLTexture(image.transformed(QTransform().scale(1.0, -1.0)))
    executor.set_drawable_texture(drawable_id, texture.textureId(), w, h)
    return texture


def render(compiled, times, out_dir, dump, warm, warm_from,
           backend='raster') -> None:
    from analysis.games.notitg import drawable_doc as dd

    evaluator, id_maps, report = dd.build_static_doc(compiled)
    print(f'doc: {report}')
    sizes = _drawable_sizes_of(id_maps, evaluator)
    images = _lazy_images(id_maps)
    executor = None
    if backend == 'gl':
        images.warm()
        executor = gl_executor(images, sizes, id_maps)
        if executor is None:
            print('no OpenGL context available; falling back to raster')
    if executor is None:
        executor = RasterExecutor(images, sizes)
    # Transparent, matching the pipeline: an opaque clear would make every
    # frame's alpha 255 and hide what the doc did NOT draw.
    executor.set_clear(SCREEN_ID, CLEAR_TRANSPARENT)
    seeded = []
    for scope, drawable_id in (id_maps.get('fields') or {}).items():
        w, h = sizes[drawable_id]
        seeded.append(seed_field(executor, drawable_id, int(w), int(h)))
        print(f'seeded {scope!r} (drawable {drawable_id}) at {int(w)}x{int(h)}')

    os.makedirs(out_dir, exist_ok=True)

    def compose(t):
        u_raw, f_raw, uf_raw, n = evaluator.frame(float(t))
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
        return executor.execute(u, f, np.frombuffer(uf_raw, dtype=np.float32))

    for t in times:
        for warm_t in warm_frames(float(t), warm, 1.0 / 60.0, warm_from):
            compose(warm_t)
        u_raw, f_raw, uf_raw, n = evaluator.frame(float(t))
        u = np.frombuffer(u_raw, dtype=np.uint32).reshape(n, evaluator.u_stride)
        f = np.frombuffer(f_raw, dtype=np.float32).reshape(n, evaluator.f_stride)
        uf = np.frombuffer(uf_raw, dtype=np.float32)
        if dump:
            print(f'\n=== t={t} ({n} ops) ===')
            for i in range(n):
                print(f'{i:>4} {describe(u, f, i)}')
        screen = executor.execute(u, f, uf)
        path = os.path.join(out_dir, f'doc_{t:g}.png')
        screen.save(path)
        print(f'wrote {path}  ({n} ops)')

    # While the context is still current: a GL texture collected after it goes
    # away warns that it "has not been destroyed" and leaks the name.
    for texture in seeded:
        if texture is not None:
            texture.destroy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('chart')
    parser.add_argument('times', nargs='+', type=float)
    parser.add_argument('--out', default='doc_frames')
    parser.add_argument('--elements', default='1', choices=('0', '1'))
    parser.add_argument('--notes', default='captured',
                        choices=('captured', 'inline'))
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--backend', default='raster', choices=('raster', 'gl'),
                        help="'gl' composes through the real GLExecutor on an "
                             'offscreen context, which is the only way to see '
                             'the per-item shader tier (lumikey, monitor)')
    parser.add_argument('--warm', type=int, default=120,
                        help='frames to compose before each time so retained '
                             'AFT slots hold real content (0 = cold)')
    parser.add_argument('--warm-from', type=float, default=None,
                        help='spread the warm-up from this chart time instead '
                             'of the 60Hz window, to reach a capture taken '
                             'before a frozen section began')
    args = parser.parse_args()

    os.environ['VSRG_DRAWABLE_ELEMENTS'] = args.elements
    os.environ['VSRG_DRAWABLE_NOTES'] = '0' if args.notes == 'captured' else '1'

    # QFontDatabase aborts without one, and the placeholder draws text.
    from PySide6.QtWidgets import QApplication
    QApplication(sys.argv[:1])

    from analysis.games.notitg.sim.producers import (
        compile_via_sim, wait_for_upgrade)

    compiled = compile_via_sim(args.chart)
    if compiled is None:
        print(f'could not compile {args.chart}')
        return 1
    wait_for_upgrade(compiled)
    render(compiled, args.times, args.out, args.dump, args.warm,
           args.warm_from, args.backend)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
