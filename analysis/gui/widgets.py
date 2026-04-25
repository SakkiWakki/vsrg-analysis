"""Small reusable Qt widgets: JumpSlider, MplTab, HtmlTab, _viz_toolbar."""
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QObject
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QSizePolicy, QSlider, QStyle,
                               QStyleOptionSlider, QScrollArea)


class WheelToScroll(QObject):
    """Forward wheel events from a child (e.g. a matplotlib canvas that
    would otherwise eat them) to a QScrollArea's vertical scrollbar."""
    def __init__(self, scroll_area):
        super().__init__(scroll_area)
        self.scroll_area = scroll_area

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel:
            bar = self.scroll_area.verticalScrollBar()
            px = ev.pixelDelta().y()
            dy = px if px else int(ev.angleDelta().y() / 2)
            bar.setValue(bar.value() - dy)
            return True
        return False

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar


class JumpSlider(QSlider):
    """QSlider where a left-click on the groove jumps to that position (default
    Qt behavior is a page step ; we want absolute seek). Uses Qt's native
    SH_Slider_AbsoluteSetButtons style hint so sliderPressed/sliderReleased
    fire at the right times for both clicks and drags."""

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            # Tell Qt to treat left-click as "jump to absolute position" for
            # this event, matching the behavior of media playbars.
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            p = ev.position().toPoint()
            handle = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            if not handle.contains(p):
                groove = self.style().subControlRect(
                    QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
                if self.orientation() == Qt.Horizontal:
                    pos = p.x() - groove.x() - handle.width() / 2
                    span = groove.width() - handle.width()
                else:
                    pos = p.y() - groove.y() - handle.height() / 2
                    span = groove.height() - handle.height()
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(),
                    int(pos), int(max(1, span)),
                    opt.upsideDown)
                self.setValue(val)
                # Fall through to the base class so it starts a drag from the
                # new handle position (which is now under the cursor). This
                # also emits sliderPressed, and sliderReleased fires correctly
                # on mouseReleaseEvent.
        super().mousePressEvent(ev)


def _viz_toolbar(on_play=None):
    """Bottom bar with optional Play button on the left. Returns a QHBoxLayout."""
    bar = QHBoxLayout()
    bar.setContentsMargins(4, 2, 4, 2)
    if on_play is not None:
        btn = QPushButton('▶ Play replay')
        btn.clicked.connect(lambda _checked=False: on_play())
        bar.addWidget(btn)
    bar.addStretch(1)
    return bar


class MplTab(QWidget):
    """Tab wrapping a matplotlib Figure with navigation toolbar + optional play.
    Canvas sits inside a vertical-only QScrollArea so tall figures don't clip."""
    def __init__(self, fig, on_play=None):
        super().__init__()
        _WheelToScroll = WheelToScroll
        self.fig = fig
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvas(fig)
        h_in = fig.get_figheight()
        dpi = fig.get_dpi()
        self.canvas.setMinimumHeight(int(h_in * dpi * 0.6))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(NavToolbar(self.canvas, self))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        host = QWidget()
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.addWidget(self.canvas, 1)
        scroll.setWidget(host)
        self._wheel_filter = _WheelToScroll(scroll)
        self.canvas.installEventFilter(self._wheel_filter)
        layout.addWidget(scroll, 1)

        if on_play is not None:
            layout.addLayout(_viz_toolbar(on_play))


class HtmlTab(QWidget):
    """Embedded HTML report viewer (uses QTextBrowser; full QWebEngine is heavier)."""
    def __init__(self, html_path):
        super().__init__()
        from PySide6.QtWidgets import QTextBrowser
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel(f'file: {html_path}'))
        open_btn = QPushButton('Open in browser')
        open_btn.clicked.connect(
            lambda: __import__('webbrowser').open(f'file://{Path(html_path).resolve()}'))
        top.addWidget(open_btn)
        top.addStretch(1)
        layout.addLayout(top)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        with open(html_path, encoding='utf-8') as f:
            self.browser.setHtml(f.read())
        layout.addWidget(self.browser, 1)
