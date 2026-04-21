"""Dark palette + QSS for the GUI. Import `apply_dark_palette(app)`."""
from PySide6.QtGui import QPalette, QColor


DARK_QSS = """
QMainWindow, QWidget { background:#1e1e1e; color:#e0e0e0; }
QTabWidget::pane { border:1px solid #333; background:#1e1e1e; }
QTabBar::tab { background:#2a2a2a; color:#e0e0e0; padding:6px 12px; border:1px solid #333; }
QTabBar::tab:selected { background:#3d3d3d; color:#ffab91; }
QTabBar::close-button { subcontrol-position: right; }
QPushButton { background:#3d3d3d; color:#e0e0e0; border:1px solid #555; padding:4px 10px; }
QPushButton:hover { background:#555; }
QPushButton:pressed { background:#666; }
QLineEdit, QComboBox, QPlainTextEdit, QTreeWidget {
    background:#2a2a2a; color:#e0e0e0; border:1px solid #3d3d3d;
    selection-background-color:#ff8a65; selection-color:#1e1e1e;
}
QPlainTextEdit { font-family: monospace; }
QHeaderView::section { background:#333; color:#ffab91; padding:4px; border:1px solid #1e1e1e; }
QTreeWidget::item:selected { background:#ff8a65; color:#1e1e1e; }
QComboBox QAbstractItemView {
    background:#2a2a2a; color:#e0e0e0;
    selection-background-color:#ff8a65; selection-color:#1e1e1e;
}
QCheckBox { color:#e0e0e0; }
QLabel { color:#e0e0e0; }
QScrollBar:vertical, QScrollBar:horizontal { background:#1e1e1e; border:none; }
QScrollBar::handle { background:#3d3d3d; }
QScrollBar::handle:hover { background:#555; }
"""


def apply_dark_palette(app):
    app.setStyle('Fusion')
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor('#1e1e1e'))
    pal.setColor(QPalette.WindowText, QColor('#e0e0e0'))
    pal.setColor(QPalette.Base, QColor('#2a2a2a'))
    pal.setColor(QPalette.AlternateBase, QColor('#252525'))
    pal.setColor(QPalette.Text, QColor('#e0e0e0'))
    pal.setColor(QPalette.Button, QColor('#3d3d3d'))
    pal.setColor(QPalette.ButtonText, QColor('#e0e0e0'))
    pal.setColor(QPalette.Highlight, QColor('#ff8a65'))
    pal.setColor(QPalette.HighlightedText, QColor('#1e1e1e'))
    app.setPalette(pal)
    app.setStyleSheet(DARK_QSS)
