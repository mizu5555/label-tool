import os
import sys
import json
import time
import math
import csv
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QIcon, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem, QLabel,
    QHBoxLayout, QVBoxLayout, QSlider, QPushButton, QMessageBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog,
    QScrollArea, QShortcut
)

# ======================
# Constants / Settings
# ======================
KEY_LastFrame = 'z'
KEY_NextFrame = 'x'
KEY_Last50Frame = 'd'
KEY_Next50Frame = 'f'
KEY_ClearFrame = 'c'
KEY_AimBot = 'v'
KEY_FlowTrack = 'a'
KEY_ToggleAuto = 'b'
KEY_Save = 's'
KEY_ExportClip = 'e'
KEY_Quit = 'q'
KEY_Esc = 27
KEY_Up = 'i'
KEY_Left = 'j'
KEY_Down = 'k'
KEY_Right = 'l'

AIMBOT_RANGE = 60
FLOW_FRAMES = 15
GAP_TOLERANCE = 30
CLIP_TAIL_FRAMES = 60
LABEL_COLORS = {
    0: (0, 0, 255),      # Normal
    1: (0, 255, 255),    # Hit
    2: (0, 255, 0),      # Land
    3: (255, 0, 0),      # Serve
    4: (255, 180, 0),    # Net
    5: (180, 0, 255),    # Bounce
}
EVENT_NAMES = {
    0: 'Normal',
    1: 'Hit',
    2: 'Land',
    3: 'Serve',
    4: 'Net',
    5: 'Bounce',
}
LABEL_SIZE = 3
TEXT_SIZE = 0.8
TEXT_COLOR = (0, 255, 255)  # BGR

HISTORY = 4  # history frames overlay
FRAME_CACHE_LIMIT = 120
MARKER_RADIUS_MIN = 2
MARKER_RADIUS_MAX = 12
COLORS = [
    QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
    QColor(255, 165, 0), QColor(128, 0, 128), QColor(0, 255, 255),
    QColor(255, 192, 203), QColor(128, 128, 0)
]

COLUMNS = ['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp']
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.mpg', '.mpeg'
}

# ======================
# Helpers
# ======================

def empty_row(frame_idx: int, fps: float) -> Dict:
    return {
        'Frame': frame_idx,
        'Visibility': 0,
        'X': 0.0,
        'Y': 0.0,
        'Z': 0.0,
        'Event': 0,
        'Timestamp': frame_idx / fps if fps else 0.0,
    }

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def app_base_dir() -> str:
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def has_console() -> bool:
    stdin_isatty = getattr(sys.stdin, 'isatty', lambda: False)
    stdout_isatty = getattr(sys.stdout, 'isatty', lambda: False)
    return bool(stdin_isatty() and stdout_isatty())

def list_videos(video_dir: str) -> List[str]:
    if not os.path.isdir(video_dir):
        return []
    return [
        f for f in os.listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]

def prompt_video_from_console(video_dir: str) -> Optional[str]:
    vids = list_videos(video_dir)
    if not vids:
        return None
    for i, f in enumerate(vids):
        video_path = os.path.join(video_dir, f)
        flag = os.path.isfile(csv_path_for_video(video_path, create_dir=False))
        print(f"{i+1:3d}: {f}{' (labeled)' if flag else ''}")
    idx = int(input('Enter the video number: '))
    return os.path.join(video_dir, vids[idx-1])

def csv_path_for_video(video_path: str, create_dir: bool = True) -> str:
    video_path = os.path.abspath(video_path)
    video_dir = os.path.dirname(video_path)
    if os.path.basename(video_dir).lower() == 'video':
        output_root = os.path.dirname(video_dir)
    else:
        output_root = video_dir
    csv_dir = os.path.join(output_root, 'csv')
    if create_dir:
        ensure_dir(csv_dir)
    csv_file_name = os.path.splitext(os.path.basename(video_path))[0] + '_ball.csv'
    return os.path.join(csv_dir, csv_file_name)

def is_dark_frame(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < 3.0 and int(gray.max()) < 18

def marker_radius_for_frame(frame_h: int, frame_w: int) -> int:
    radius = int(min(frame_h, frame_w) * 0.002)
    return max(MARKER_RADIUS_MIN, min(MARKER_RADIUS_MAX, radius))

def event_color(event_id: int) -> Tuple[int, int, int]:
    return LABEL_COLORS.get(int(event_id), LABEL_COLORS[0])

def event_name(event_id: int) -> str:
    return EVENT_NAMES.get(int(event_id), EVENT_NAMES[0])

def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

def hough_aimbot(crop_gray: np.ndarray) -> Tuple[int, int]:
    if crop_gray is None or crop_gray.size == 0:
        return -1, -1
    blurred = cv2.GaussianBlur(crop_gray, (5, 5), 1)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=15,
        minRadius=2,
        maxRadius=25,
    )
    if circles is None:
        return -1, -1
    c = circles[0][0]
    return int(c[0]), int(c[1])

def draw_pickleball_marker(image: np.ndarray, row: Dict):
    if not row or not row.get('Visibility'):
        return
    x = int(row['X'])
    y = int(row['Y'])
    radius = max(2, int(row.get('Z', 0))) if row.get('Z', 0) else 6
    color = event_color(int(row.get('Event', 0)))
    cv2.line(image, (x - 1, y), (x + 1, y), color, 1)
    cv2.line(image, (x, y - 1), (x, y + 1), color, 1)
    cv2.putText(
        image,
        event_name(int(row.get('Event', 0))),
        (x + radius + 4, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
    )

def choose_video_with_dialog(initial_dir: str) -> Optional[str]:
    video_filter = (
        "Video files (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.mpg *.mpeg);;"
        "All files (*)"
    )
    video_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select video",
        initial_dir,
        video_filter,
    )
    return video_path or None

def resolve_video_path() -> Optional[str]:
    if len(sys.argv) >= 2 and not os.path.isdir(sys.argv[1]):
        return sys.argv[1]

    work_dir = app_base_dir() if len(sys.argv) < 2 else os.path.abspath(sys.argv[1])
    os.chdir(work_dir)
    return None

# ======================
# Graphics View
# ======================
class VideoView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #111;")
        self.source_pixmap: Optional[QPixmap] = None
        self.displayed_pixmap: Optional[QPixmap] = None
        self.frame_w = 1
        self.frame_h = 1
        self.zoom_factor = 1.0
        self.pan_x = 0.5
        self.pan_y = 0.5
        self.pan_dragging = False
        self.pan_last_pos = None
        self.scroll_area: Optional[QScrollArea] = None

    def set_frame_pixmap(self, pixmap: QPixmap, width: int, height: int):
        self.source_pixmap = pixmap
        self.frame_w = max(1, int(width))
        self.frame_h = max(1, int(height))
        self.clamp_pan()
        self.update_display()

    def clear_frame(self):
        self.source_pixmap = None
        self.displayed_pixmap = None
        self.pan_dragging = False
        self.pan_last_pos = None
        self.clear()

    def update_display(self):
        if self.source_pixmap is None or self.source_pixmap.isNull():
            self.clear()
            return
        viewport_size = self.size()
        if self.scroll_area is not None:
            viewport_size = self.scroll_area.viewport().size()
        base_size = self.source_pixmap.size()
        base_size.scale(viewport_size, Qt.KeepAspectRatio)
        target_w = max(1, int(base_size.width() * self.zoom_factor))
        target_h = max(1, int(base_size.height() * self.zoom_factor))
        self.displayed_pixmap = self.source_pixmap.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(self.displayed_pixmap)
        self.resize(self.displayed_pixmap.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def scale(self, factor_x, factor_y):
        self.zoom_around(float(factor_x), None)

    def zoom_around(self, factor: float, pos):
        old_w = self.displayed_pixmap.width() if self.displayed_pixmap and not self.displayed_pixmap.isNull() else 0
        old_h = self.displayed_pixmap.height() if self.displayed_pixmap and not self.displayed_pixmap.isNull() else 0
        rel_x = rel_y = viewport_x = viewport_y = None
        if pos is not None and old_w > 0 and old_h > 0:
            rel_x = pos.x() / old_w
            rel_y = pos.y() / old_h
            if self.scroll_area is not None:
                viewport_x = pos.x() - self.scroll_area.horizontalScrollBar().value()
                viewport_y = pos.y() - self.scroll_area.verticalScrollBar().value()
        self.zoom_factor = max(1.0, min(12.0, self.zoom_factor * factor))
        self.update_display()
        if (
            self.scroll_area is not None and rel_x is not None and rel_y is not None
            and viewport_x is not None and viewport_y is not None
            and self.displayed_pixmap is not None and not self.displayed_pixmap.isNull()
        ):
            hbar = self.scroll_area.horizontalScrollBar()
            vbar = self.scroll_area.verticalScrollBar()
            hbar.setValue(int(rel_x * self.displayed_pixmap.width() - viewport_x))
            vbar.setValue(int(rel_y * self.displayed_pixmap.height() - viewport_y))

    def resetTransform(self):
        self.zoom_factor = 1.0
        self.pan_x = 0.5
        self.pan_y = 0.5
        self.update_display()
        if self.scroll_area is not None:
            self.scroll_area.horizontalScrollBar().setValue(0)
            self.scroll_area.verticalScrollBar().setValue(0)

    def fitInView(self, *args, **kwargs):
        self.zoom_factor = 1.0
        self.update_display()

    def sceneRect(self):
        return None

    def clamp_pan(self):
        self.pan_x = max(0.0, min(1.0, float(self.pan_x)))
        self.pan_y = max(0.0, min(1.0, float(self.pan_y)))

    def displayed_offsets(self) -> Tuple[float, float]:
        pix = self.displayed_pixmap
        if pix is None or pix.isNull():
            return 0.0, 0.0
        return (self.width() - pix.width()) / 2.0, (self.height() - pix.height()) / 2.0

    def current_crop(self) -> Tuple[int, int, int, int]:
        src_w = max(1, self.frame_w)
        src_h = max(1, self.frame_h)
        visible_w = max(1, min(src_w, int(round(src_w / self.zoom_factor))))
        visible_h = max(1, min(src_h, int(round(src_h / self.zoom_factor))))
        left = int(round(self.pan_x * src_w - visible_w / 2))
        top = int(round(self.pan_y * src_h - visible_h / 2))
        left = max(0, min(src_w - visible_w, left))
        top = max(0, min(src_h - visible_h, top))
        return left, top, visible_w, visible_h

    def view_pos_to_frame_fraction(self, pos):
        if pos is None or self.displayed_pixmap is None or self.displayed_pixmap.isNull():
            return None
        px = pos.x()
        py = pos.y()
        if px < 0 or py < 0 or px >= self.displayed_pixmap.width() or py >= self.displayed_pixmap.height():
            return None
        rel_x = px / max(1, self.displayed_pixmap.width())
        rel_y = py / max(1, self.displayed_pixmap.height())
        frame_x = rel_x * self.frame_w
        frame_y = rel_y * self.frame_h
        return frame_x / max(1, self.frame_w), frame_y / max(1, self.frame_h), rel_x, rel_y

    def map_to_frame_xy(self, event):
        pix = self.displayed_pixmap
        if pix is None or pix.isNull():
            return -1, -1
        px = event.pos().x()
        py = event.pos().y()
        if px < 0 or py < 0 or px >= pix.width() or py >= pix.height():
            return -1, -1
        x = int(px * self.frame_w / pix.width())
        y = int(py * self.frame_h / pix.height())
        return x, y

    def mousePressEvent(self, event):
        if self.window() is None:
            return super().mousePressEvent(event)
        mw = self.window()
        if not getattr(mw, 'video_path', None):
            return super().mousePressEvent(event)
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier
        ):
            self.pan_dragging = True
            self.pan_last_pos = event.pos()
            return
        x, y = self.map_to_frame_xy(event)
        if 0 <= x < mw.frame_w and 0 <= y < mw.frame_h:
            if event.button() == Qt.LeftButton:
                mw.begin_bbox(x, y)
            elif event.button() == Qt.RightButton:
                mw.aimbot_click(x, y)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        mw = self.window()
        if mw is None or not getattr(mw, 'video_path', None):
            return super().mouseMoveEvent(event)
        if self.pan_dragging and self.pan_last_pos is not None:
            dx = event.pos().x() - self.pan_last_pos.x()
            dy = event.pos().y() - self.pan_last_pos.y()
            if self.scroll_area is not None:
                hbar = self.scroll_area.horizontalScrollBar()
                vbar = self.scroll_area.verticalScrollBar()
                hbar.setValue(hbar.value() - dx)
                vbar.setValue(vbar.value() - dy)
            self.pan_last_pos = event.pos()
            return
        x, y = self.map_to_frame_xy(event)
        if 0 <= x < mw.frame_w and 0 <= y < mw.frame_h:
            mw.update_bbox(x, y)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        mw = self.window()
        if mw is None or not getattr(mw, 'video_path', None):
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self.pan_dragging
        ):
            self.pan_dragging = False
            self.pan_last_pos = None
            return
        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)
        x, y = self.map_to_frame_xy(event)
        if 0 <= x < mw.frame_w and 0 <= y < mw.frame_h:
            mw.finish_bbox(x, y)
        else:
            mw.cancel_bbox()
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        mw = self.window()
        if mw is None:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_around(1.1 if delta > 0 else 1 / 1.1, event.pos())
        slider_value = int(round(math.log(self.zoom_factor, 1.1))) if self.zoom_factor > 0 else 0
        mw.zoom_slider.blockSignals(True)
        mw.zoom_slider.setValue(max(mw.zoom_slider.minimum(), min(mw.zoom_slider.maximum(), slider_value)))
        mw.zoom_slider.blockSignals(False)
        mw.scale = self.zoom_factor
        mw.zoom_label.setText(f"Zoom: {self.zoom_factor:.1f}x")


# ======================
# Main Window
# ======================
class MainWindow(QMainWindow):
    def __init__(self, video_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Label Tool (PyQt5)")

        # Video / data state
        self.video_path: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.df: Dict[int, Dict] = {}
        self.fps = 30.0
        self.total_frames = 0
        self.frame_h = 1
        self.frame_w = 1
        self.scale = 1.0
        self.playback_speed = 1.0
        self.current_frame = 0
        self.brightness = 1.0
        self.brightness_step = 0.1
        self.aimbot_pending = False
        self.auto_predict_enabled = True
        self.bbox_drawing = False
        self.bbox_start: Optional[Tuple[int, int]] = None
        self.bbox_current: Optional[Tuple[int, int]] = None

        # Widgets
        self.video_view = VideoView(self)
        self.video_scroll = QScrollArea()
        self.video_scroll.setWidget(self.video_view)
        self.video_scroll.setWidgetResizable(False)
        self.video_scroll.setAlignment(Qt.AlignCenter)
        self.video_scroll.setFocusPolicy(Qt.NoFocus)
        self.video_view.scroll_area = self.video_scroll
        self.video_view.setFocusPolicy(Qt.NoFocus)
        font = QFont()

        # Zoom controls
        font.setPointSize(14)
        self.zoom_slider = QSlider(Qt.Vertical)
        self.zoom_slider.setRange(0, 18)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.zoom_label = QLabel("Zoom: 1.0x")
        self.zoom_label.setFont(font)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_layout = QVBoxLayout()
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_widget = QWidget()
        zoom_widget.setLayout(zoom_layout)

        # Show history
        self.show_history = False
        self.history_checkbox = QCheckBox("Show last 4 frames")
        self.history_checkbox.setFont(font)
        self.history_checkbox.setChecked(False)
        self.history_checkbox.stateChanged.connect(self.on_history_toggled)

        # Frame/Status labels
        font.setPointSize(20)
        self.frame_label = QLabel("Frame: -/-")
        self.frame_label.setFont(font)
        self.event_label = QLabel("Event: 0")
        self.event_label.setFont(font)
        font.setPointSize(12)
        self.decode_label = QLabel("Video: none")
        self.decode_label.setFont(font)
        self.decode_label.setWordWrap(True)

        font.setPointSize(14)
        self.open_video_btn = QPushButton("Open Video")
        self.open_video_btn.setFont(font)
        self.open_video_btn.clicked.connect(self.open_video_dialog)

        # Navigation & controls
        font.setPointSize(16)
        self.play_btn = QPushButton("Play")
        self.play_btn.setFont(font)
        self.play_btn.clicked.connect(self.play_pause)

        self.prev50_btn = QPushButton("<< 50")
        self.prev50_btn.clicked.connect(lambda: self.seek_frame(self.current_frame - 50))
        self.prev1_btn = QPushButton("< 1")
        self.prev1_btn.clicked.connect(lambda: self.seek_frame(self.current_frame - 1))
        self.next1_btn = QPushButton("1 >")
        self.next1_btn.clicked.connect(lambda: self.seek_frame(self.current_frame + 1))
        self.next50_btn = QPushButton("50 >>")
        self.next50_btn.clicked.connect(lambda: self.seek_frame(self.current_frame + 50))

        # Speed
        self.speed_label = QLabel("Speed: 1.0x")
        self.speed_label.setFont(font)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.speed_slider.setMaximumWidth(150)

        # Frame slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.seek_frame)

        # Brightness
        font.setPointSize(14)
        self.brighten_btn = QPushButton("Brighten")
        self.brighten_btn.setFont(font)
        self.darken_btn = QPushButton("Darken")
        self.darken_btn.setFont(font)
        self.brighten_btn.clicked.connect(self.on_brighten)
        self.darken_btn.clicked.connect(self.on_darken)

        # Layouts
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 4, 8, 4)
        self.frame_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.event_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        left_layout.addSpacing(16)
        left_layout.addWidget(self.frame_label)
        left_layout.addWidget(self.open_video_btn)
        left_layout.addSpacing(16)
        left_layout.addWidget(self.event_label)
        left_layout.addWidget(self.decode_label)
        left_layout.addStretch(1)

        ctrl_panel = QWidget()
        cp_layout = QHBoxLayout(ctrl_panel)
        cp_layout.addWidget(self.prev50_btn)
        cp_layout.addWidget(self.prev1_btn)
        cp_layout.addWidget(self.slider)
        cp_layout.addWidget(self.next1_btn)
        cp_layout.addWidget(self.next50_btn)
        cp_layout.addWidget(self.play_btn)
        cp_layout.addWidget(self.speed_label)
        cp_layout.addWidget(self.speed_slider)
        cp_layout.addWidget(self.brighten_btn)
        cp_layout.addWidget(self.darken_btn)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        h_layout = QHBoxLayout()
        h_layout.addWidget(left_panel, 2)
        h_layout.addWidget(self.video_scroll, 12)
        h_layout.addWidget(zoom_widget, 1)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(ctrl_panel)
        self.setCentralWidget(central)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.setup_shortcuts()

        self.set_video_controls_enabled(False)
        self.show_empty_view()

        if video_path:
            self.load_video(video_path)

    # ---------- Data IO ----------
    def setup_shortcuts(self):
        shortcuts = {
            KEY_LastFrame: lambda: self.seek_frame(self.current_frame - 1),
            KEY_NextFrame: lambda: self.seek_frame(self.current_frame + 1),
            KEY_Last50Frame: lambda: self.seek_frame(self.current_frame - 50),
            KEY_Next50Frame: lambda: self.seek_frame(self.current_frame + 50),
            KEY_ClearFrame: self.clear_current_frame,
            KEY_AimBot: self.next_frame_aimbot,
            KEY_FlowTrack: lambda: self.optical_flow_track(FLOW_FRAMES),
            KEY_ToggleAuto: self.toggle_auto_predict,
            KEY_Save: self.save_with_message,
            KEY_ExportClip: self.export_clips,
            KEY_Quit: self.save_and_close,
            KEY_Up: lambda: self.nudge_current_point(0, -1),
            KEY_Left: lambda: self.nudge_current_point(-1, 0),
            KEY_Down: lambda: self.nudge_current_point(0, 1),
            KEY_Right: lambda: self.nudge_current_point(1, 0),
            '0': lambda: self.set_current_event(0),
            '1': lambda: self.set_current_event(1),
            '2': lambda: self.set_current_event(2),
            '3': lambda: self.set_current_event(3),
            '4': lambda: self.set_current_event(4),
            '5': lambda: self.set_current_event(5),
            'Space': self.play_pause,
            'Esc': self.confirm_close,
        }
        self.shortcuts = []
        for key, callback in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def has_video(self) -> bool:
        return bool(self.video_path and self.cap is not None)

    def set_video_controls_enabled(self, enabled: bool):
        for widget in (
            self.zoom_slider, self.history_checkbox, self.play_btn,
            self.prev50_btn, self.prev1_btn, self.next1_btn, self.next50_btn,
            self.speed_slider, self.slider, self.brighten_btn, self.darken_btn,
        ):
            widget.setEnabled(enabled)

    def clear_current_frame(self):
        if not self.has_video():
            return
        self.df[self.current_frame] = empty_row(self.current_frame, self.fps)
        self.update_frame()

    def save_with_message(self):
        if not self.has_video():
            return
        self.save_csv()
        QMessageBox.information(self, "Saved", f"Saved CSV to:\n{self.csv_path}")

    def save_and_close(self):
        if not self.has_video():
            self.close()
            return
        self.save_csv()
        self.close()

    def set_current_event(self, event_id: int):
        if not self.has_video():
            return
        self.ensure_row(self.current_frame)
        self.df[self.current_frame]['Event'] = int(event_id)
        self.update_frame()

    def nudge_current_point(self, dx: int, dy: int):
        if not self.has_video():
            return
        row = self.df.get(self.current_frame)
        if not row or not row['Visibility']:
            return
        row['X'] = float(max(0, min(self.frame_w - 1, int(row['X']) + dx)))
        row['Y'] = float(max(0, min(self.frame_h - 1, int(row['Y']) + dy)))
        self.update_frame()

    def toggle_auto_predict(self):
        if not self.has_video():
            return
        self.auto_predict_enabled = not self.auto_predict_enabled
        self.update_frame()

    def show_empty_view(self):
        self.timer.stop()
        self.play_btn.setText("Play")
        self.video_view.clear_frame()
        self.frame_label.setText("Frame: -/-")
        self.event_label.setText("Event: 0")
        self.decode_label.setText("Video: none")
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.setWindowTitle("Label Tool (PyQt5)")

    def maybe_save_current_video(self) -> bool:
        if not self.has_video():
            return True
        ans = QMessageBox.question(
            self, "Open another video", "Save current labels before opening another video?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if ans == QMessageBox.Yes:
            try:
                self.save_csv()
            except Exception as e:
                QMessageBox.warning(self, "Save failed", str(e))
                return False
            return True
        if ans == QMessageBox.No:
            return True
        return False

    def open_video_dialog(self):
        if not self.maybe_save_current_video():
            return
        if self.video_path:
            initial_dir = os.path.dirname(self.video_path)
        else:
            video_dir = os.path.join(app_base_dir(), 'video')
            initial_dir = video_dir if os.path.isdir(video_dir) else app_base_dir()
        video_path = choose_video_with_dialog(initial_dir)
        if not video_path:
            return
        try:
            self.load_video(video_path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to open video", str(e))

    def load_video(self, video_path: str):
        video_path = os.path.abspath(video_path)
        new_cap = cv2.VideoCapture(video_path)
        if not new_cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = new_cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(new_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ret, first_frame = new_cap.read()
        if not ret:
            new_cap.release()
            raise ValueError("Cannot load video or video has no frames.")
        if total_frames <= 0:
            total_frames = 1

        initial_frame_idx = 0
        if is_dark_frame(first_frame):
            search_limit = min(total_frames, 600)
            for idx in range(1, search_limit):
                new_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, candidate = new_cap.read()
                if ret and not is_dark_frame(candidate):
                    initial_frame_idx = idx
                    first_frame = candidate
                    break

        self.timer.stop()
        self.play_btn.setText("Play")
        if self.cap is not None:
            self.cap.release()

        self.video_path = video_path
        self.cap = new_cap
        self.fps = fps
        self.total_frames = total_frames
        self.frame_h, self.frame_w = first_frame.shape[:2]
        self.frame_cache = OrderedDict()
        self.frame_cache[initial_frame_idx] = first_frame
        self.current_frame = initial_frame_idx
        self.brightness = 1.0
        self.auto_predict_enabled = True
        self.bbox_drawing = False
        self.bbox_start = None
        self.bbox_current = None
        self.csv_path = csv_path_for_video(video_path, create_dir=True)
        self.df = {}
        self.load_csv_if_any()

        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(self.current_frame)
        self.zoom_slider.setValue(0)
        self.speed_slider.setValue(10)
        self.set_video_controls_enabled(True)
        self.setWindowTitle(f"Label Tool - {os.path.basename(video_path)}")
        self.update_frame()
        self.video_view.resetTransform()
        self.video_view.fitInView(self.video_view.sceneRect(), Qt.KeepAspectRatio)

    def get_frame(self, idx: int) -> np.ndarray:
        if self.cap is None:
            raise ValueError("No video is loaded.")
        if idx in self.frame_cache:
            self.frame_cache.move_to_end(idx)
            return self.frame_cache[idx]

        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {idx}.")

        self.frame_cache[idx] = frame
        while len(self.frame_cache) > FRAME_CACHE_LIMIT:
            self.frame_cache.popitem(last=False)
        return frame

    def load_csv_if_any(self):
        if not self.csv_path:
            self.df = {}
            return
        if os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            required = {'Frame', 'Visibility', 'X', 'Y'}
            if not required.issubset(df.columns):
                raise ValueError(f"{self.csv_path} is missing required columns from {sorted(required)}!")
            # Build dict by index
            self.df = {int(row['Frame']): {
                'Frame': int(row['Frame']),
                'Visibility': int(row['Visibility']),
                'X': float(row['X']),
                'Y': float(row['Y']),
                'Z': float(row.get('Z', 0.0)),
                'Event': int(row.get('Event', 0)),
                'Timestamp': float(row.get('Timestamp', int(row['Frame']) / self.fps if self.fps else 0.0)),
            } for _, row in df.iterrows()}
        else:
            self.df = {}
        # Ensure at least row 0 exists
        if 0 not in self.df:
            self.df[0] = empty_row(0, self.fps)

    def save_csv(self):
        if not self.csv_path:
            raise ValueError("No video is loaded.")
        # Ensure contiguous rows up to last frame labeled or total_frames-1
        last_idx = max(self.df.keys()) if self.df else 0
        rows = []
        for i in range(0, max(last_idx, self.total_frames-1) + 1):
            row = self.df.get(i)
            if row is None:
                row = empty_row(i, self.fps)
            rows.append(row)
        pdf = pd.DataFrame(rows, columns=COLUMNS)
        pdf.to_csv(self.csv_path, index=False, encoding='utf-8')

    def export_clips(self):
        if not self.has_video() or not self.video_path:
            return
        self.save_csv()
        labeled = sorted(idx for idx, row in self.df.items() if row.get('Visibility'))
        if not labeled:
            QMessageBox.information(self, "Export Clips", "No labeled frames to export.")
            return

        clips = []
        start = end = labeled[0]
        for frame_idx in labeled[1:]:
            if frame_idx - end <= GAP_TOLERANCE:
                end = frame_idx
            else:
                clips.append((start, end))
                start = end = frame_idx
        clips.append((start, end))

        output_root = os.path.dirname(os.path.dirname(self.csv_path)) if self.csv_path else os.path.dirname(self.video_path)
        clip_video_dir = os.path.join(output_root, 'clips', 'video')
        clip_csv_dir = os.path.join(output_root, 'clips', 'csv')
        ensure_dir(clip_video_dir)
        ensure_dir(clip_csv_dir)

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        exported = 0
        for clip_idx, (start, end) in enumerate(clips):
            clip_frames = list(range(start, min(end + CLIP_TAIL_FRAMES, self.total_frames - 1) + 1))
            if not clip_frames:
                continue
            video_path = os.path.join(clip_video_dir, f"{base}_clip{clip_idx:03d}.mp4")
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.frame_w, self.frame_h), isColor=True)
            for frame_idx in clip_frames:
                frame = self.get_frame(frame_idx)
                if frame is not None:
                    writer.write(frame)
            writer.release()

            csv_path = os.path.join(clip_csv_dir, f"{base}_clip{clip_idx:03d}_ball.csv")
            rows = []
            for new_idx, original_idx in enumerate(clip_frames):
                row = self.df.get(original_idx, empty_row(original_idx, self.fps)).copy()
                row['Frame'] = new_idx
                row['Timestamp'] = new_idx / self.fps if self.fps else 0.0
                rows.append(row)
            pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False, encoding='utf-8')
            exported += 1

        QMessageBox.information(
            self,
            "Export Clips",
            f"Exported {exported} clip(s).\nVideo: {clip_video_dir}\nCSV: {clip_csv_dir}",
        )

    # ---------- UI plumbing ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_view.update_display()

    def on_zoom_changed(self, value):
        target_scale = 1.1 ** value
        factor = target_scale / self.scale
        self.video_view.scale(factor, factor)
        self.scale = target_scale
        self.zoom_label.setText(f"Zoom: {target_scale:.1f}x")

    def on_speed_changed(self, val):
        self.playback_speed = val / 10.0
        self.speed_label.setText(f"Speed: {self.playback_speed:.1f}x")
        if self.timer.isActive():
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))

    def on_history_toggled(self, state):
        self.show_history = (state == Qt.Checked)
        if self.has_video():
            self.update_frame()

    def on_brighten(self):
        if not self.has_video():
            return
        self.brightness = min(self.brightness + self.brightness_step, 3.0)
        self.update_frame()

    def on_darken(self):
        if not self.has_video():
            return
        self.brightness = max(self.brightness - self.brightness_step, 0.1)
        self.update_frame()

    def play_pause(self):
        if not self.has_video():
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            if self.current_frame >= self.total_frames - 1:
                self.current_frame = 0
                self.update_frame()
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
            self.play_btn.setText("Pause")

    def next_frame(self):
        if not self.has_video():
            return
        if self.current_frame + 1 < self.total_frames:
            self.current_frame += 1
            prev_labeled = next(
                (self.df[i] for i in range(self.current_frame - 1, max(self.current_frame - 51, -1), -1)
                 if self.df.get(i, {}).get('Visibility', 0)),
                None,
            )
            target_has_ball = bool(self.df.get(self.current_frame, {}).get('Visibility', 0))
            after_land = prev_labeled is not None and int(prev_labeled.get('Event', 0)) == 2
            if self.auto_predict_enabled and prev_labeled is not None and not target_has_ball and not after_land:
                self.auto_predict_next(self.current_frame)
            self.update_frame()
        else:
            self.timer.stop()
            self.play_btn.setText("Play")

    def seek_frame(self, idx: int):
        if not self.has_video():
            return
        idx = max(0, min(self.total_frames - 1, int(idx)))
        self.current_frame = idx
        self.update_frame()

    # ---------- Label ops ----------
    def ensure_row(self, idx: int):
        if idx not in self.df:
            self.df[idx] = empty_row(idx, self.fps)

    def set_point_at(self, x: int, y: int):
        if not self.has_video():
            return
        self.ensure_row(self.current_frame)
        row = self.df[self.current_frame]
        row['Visibility'] = 1
        row['X'] = float(x)
        row['Y'] = float(y)
        row['Z'] = 0.0
        self.update_frame()

    def set_bbox_at(self, x0: int, y0: int, x1: int, y1: int):
        if abs(x1 - x0) < 5 and abs(y1 - y0) < 5:
            cx, cy, radius = x1, y1, 0
        else:
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            radius = max(abs(x1 - x0), abs(y1 - y0)) // 2
        cx = max(0, min(self.frame_w - 1, int(cx)))
        cy = max(0, min(self.frame_h - 1, int(cy)))
        self.ensure_row(self.current_frame)
        row = self.df[self.current_frame]
        row['Visibility'] = 1
        row['X'] = float(cx)
        row['Y'] = float(cy)
        row['Z'] = float(radius)
        self.update_frame()

    def begin_bbox(self, x: int, y: int):
        self.bbox_drawing = True
        self.bbox_start = (x, y)
        self.bbox_current = (x, y)
        self.update_frame()

    def update_bbox(self, x: int, y: int):
        if not self.bbox_drawing:
            return
        self.bbox_current = (x, y)
        self.update_frame()

    def finish_bbox(self, x: int, y: int):
        if not self.bbox_drawing or self.bbox_start is None:
            return
        x0, y0 = self.bbox_start
        self.bbox_drawing = False
        self.bbox_start = None
        self.bbox_current = None
        self.set_bbox_at(x0, y0, x, y)

    def cancel_bbox(self):
        self.bbox_drawing = False
        self.bbox_start = None
        self.bbox_current = None
        if self.has_video():
            self.update_frame()

    def aimbot_click(self, x: int, y: int):
        if not self.has_video():
            return
        frame = self.get_frame(self.current_frame)
        x1 = max(0, x - AIMBOT_RANGE)
        y1 = max(0, y - AIMBOT_RANGE)
        x2 = min(self.frame_w, x + AIMBOT_RANGE)
        y2 = min(self.frame_h, y + AIMBOT_RANGE)
        kx, ky = hough_aimbot(to_gray(frame[y1:y2, x1:x2]))
        self.ensure_row(self.current_frame)
        row = self.df[self.current_frame]
        row['Visibility'] = 1
        row['Z'] = 0.0
        if kx != -1:
            row['X'] = float(x1 + kx)
            row['Y'] = float(y1 + ky)
        else:
            row['X'] = float(x)
            row['Y'] = float(y)
        self.update_frame()

    def next_frame_aimbot(self):
        if not self.has_video():
            return
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
        self.ensure_row(self.current_frame)
        last_vis = next(
            (self.df[i] for i in range(self.current_frame - 1, -1, -1)
             if self.df.get(i, {}).get('Visibility', 0)),
            None,
        )
        if last_vis:
            self.aimbot_click(int(last_vis['X']), int(last_vis['Y']))
        else:
            self.update_frame()

    def optical_flow_track(self, n_frames: int = FLOW_FRAMES):
        if not self.has_video():
            return
        start_idx = self.current_frame
        if self.df.get(start_idx, {}).get('Visibility', 0) == 0:
            QMessageBox.information(self, "Flow Track", "Current frame has no labeled ball.")
            return
        prev_gray = to_gray(self.get_frame(start_idx))
        pt = np.array([[self.df[start_idx]['X'], self.df[start_idx]['Y']]], dtype=np.float32).reshape(-1, 1, 2)
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        end_idx = min(start_idx + n_frames, self.total_frames - 1)
        tracked_to = start_idx
        for idx in range(start_idx + 1, end_idx + 1):
            cur_gray = to_gray(self.get_frame(idx))
            new_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pt, None, **lk_params)
            if status[0][0] != 1:
                break
            nx, ny = new_pt[0][0]
            self.ensure_row(idx)
            self.df[idx]['Visibility'] = 1
            self.df[idx]['X'] = float(nx)
            self.df[idx]['Y'] = float(ny)
            self.df[idx]['Z'] = self.df[start_idx].get('Z', 0.0)
            prev_gray = cur_gray
            pt = new_pt
            tracked_to = idx
        self.current_frame = tracked_to
        self.update_frame()

    def auto_predict_next(self, target_idx: int):
        prev_idx = None
        for idx in range(target_idx - 1, max(target_idx - 10, -1), -1):
            if self.df.get(idx, {}).get('Visibility', 0):
                prev_idx = idx
                break
        if prev_idx is None:
            return
        prev_gray = to_gray(self.get_frame(prev_idx))
        cur_gray = to_gray(self.get_frame(target_idx))
        px, py = float(self.df[prev_idx]['X']), float(self.df[prev_idx]['Y'])
        pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        new_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pt, None, **lk_params)
        if status[0][0] == 1:
            nx, ny = new_pt[0][0]
            if abs(nx - px) < 80 and abs(ny - py) < 80:
                self.ensure_row(target_idx)
                self.df[target_idx].update(Visibility=1, X=float(nx), Y=float(ny), Z=self.df[prev_idx].get('Z', 0.0))
                return
        x = int(px)
        y = int(py)
        x1 = max(0, x - AIMBOT_RANGE)
        y1 = max(0, y - AIMBOT_RANGE)
        x2 = min(self.frame_w, x + AIMBOT_RANGE)
        y2 = min(self.frame_h, y + AIMBOT_RANGE)
        kx, ky = hough_aimbot(cur_gray[y1:y2, x1:x2])
        if kx != -1:
            self.ensure_row(target_idx)
            self.df[target_idx].update(Visibility=1, X=float(x1 + kx), Y=float(y1 + ky), Z=self.df[prev_idx].get('Z', 0.0))

    def predict_next_from_history(self) -> Optional[Tuple[float, float]]:
        """If we have >=3 visible history points, polyfit (quadratic) to predict next.
        Returns (x_pred, y_pred) or None.
        """
        # collect last visible indices
        vis = []
        for k in range(self.current_frame-1, -1, -1):
            row = self.df.get(k)
            if row and row['Visibility']:
                vis.append((k, row['X'], row['Y']))
                if len(vis) >= 4:
                    break
        if len(vis) < 2:
            return None
        vis = list(reversed(vis))  # oldest -> newest
        t = np.arange(1, len(vis)+1, dtype=float)
        xs = np.array([p[1] for p in vis], dtype=float)
        ys = np.array([p[2] for p in vis], dtype=float)
        deg = 2 if len(vis) >= 3 else 1
        try:
            coef_x = np.polyfit(t, xs, deg)
            coef_y = np.polyfit(t, ys, deg)
            t_next = len(vis) + 1
            x_pred = np.polyval(coef_x, t_next)
            y_pred = np.polyval(coef_y, t_next)
            return float(x_pred), float(y_pred)
        except Exception:
            return None

    # ---------- Paint ----------
    def update_frame(self):
        if not self.has_video():
            self.show_empty_view()
            return
        src = self.get_frame(self.current_frame).copy()
        display = cv2.convertScaleAbs(src, alpha=self.brightness, beta=0)
        frame_mean = float(src.mean())
        frame_max = int(src.max())

        # current
        rowc = self.df.get(self.current_frame)
        draw_pickleball_marker(display, rowc)

        if self.bbox_drawing and self.bbox_start and self.bbox_current:
            x0, y0 = self.bbox_start
            x1, y1 = self.bbox_current
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 200, 255), 1)
            cv2.drawMarker(display, ((x0 + x1) // 2, (y0 + y1) // 2), (0, 200, 255), cv2.MARKER_CROSS, 10, 1)

        # Push to Qt through encoded image bytes instead of raw ndarray memory.
        # This avoids format/stride issues that can show up as black or rainbow frames.
        ok, encoded = cv2.imencode('.png', display)
        pix = QPixmap()
        if ok:
            pix.loadFromData(encoded.tobytes(), 'PNG')
        if not ok or pix.isNull():
            debug_path = None
            if self.csv_path:
                debug_path = os.path.join(os.path.dirname(self.csv_path), 'debug_last_frame.png')
                cv2.imwrite(debug_path, display)
            detail = f"Saved debug frame to:\n{debug_path}" if debug_path else "No debug path is available."
            QMessageBox.warning(self, "Display failed", f"Qt could not create a pixmap from this frame.\n{detail}")
            return
        self.video_view.set_frame_pixmap(pix, self.frame_w, self.frame_h)

        # labels
        ev = rowc['Event'] if rowc else 0
        self.frame_label.setText(f"Frame: {self.current_frame}/{self.total_frames-1}")
        auto_txt = "ON" if self.auto_predict_enabled else "OFF"
        self.event_label.setText(f"Event: {ev} {event_name(ev)}\nAuto: {auto_txt}")
        self.decode_label.setText(f"Decoded: mean {frame_mean:.1f}, max {frame_max}")
        self.slider.setValue(self.current_frame)

    # ---------- Key handling ----------
    def keyPressEvent(self, event):
        k = event.key()
        text = event.text()
        handled = True

        if not self.has_video():
            if k == Qt.Key_Escape:
                self.close()
            else:
                handled = False
            if not handled:
                super().keyPressEvent(event)
            return

        if text == KEY_LastFrame:
            self.seek_frame(self.current_frame - 1)
        elif text == KEY_NextFrame:
            self.seek_frame(self.current_frame + 1)
        elif text == KEY_Last50Frame:
            self.seek_frame(self.current_frame - 50)
        elif text == KEY_Next50Frame:
            self.seek_frame(self.current_frame + 50)
        elif text == KEY_ClearFrame:
            self.df[self.current_frame] = empty_row(self.current_frame, self.fps)
            self.update_frame()
        elif text == KEY_AimBot:
            self.next_frame_aimbot()
        elif text == KEY_FlowTrack:
            self.optical_flow_track(FLOW_FRAMES)
        elif text == KEY_ToggleAuto:
            self.auto_predict_enabled = not self.auto_predict_enabled
            self.update_frame()
        elif text == KEY_Save:
            self.save_csv()
            QMessageBox.information(self, "Saved", f"Saved CSV to:\n{self.csv_path}")
        elif text == KEY_ExportClip:
            self.export_clips()
        elif text == KEY_Quit:
            self.save_csv()
            self.close()
        elif text in ('0', '1', '2', '3', '4', '5'):
            self.ensure_row(self.current_frame)
            self.df[self.current_frame]['Event'] = int(text)
            self.update_frame()
        elif text == KEY_Up:
            r = self.df.get(self.current_frame)
            if r and r['Visibility']:
                r['Y'] = float(max(0, int(r['Y']) - 1))
                self.update_frame()
        elif text == KEY_Down:
            r = self.df.get(self.current_frame)
            if r and r['Visibility']:
                r['Y'] = float(min(self.frame_h-1, int(r['Y']) + 1))
                self.update_frame()
        elif text == KEY_Left:
            r = self.df.get(self.current_frame)
            if r and r['Visibility']:
                r['X'] = float(max(0, int(r['X']) - 1))
                self.update_frame()
        elif text == KEY_Right:
            r = self.df.get(self.current_frame)
            if r and r['Visibility']:
                r['X'] = float(min(self.frame_w-1, int(r['X']) + 1))
                self.update_frame()
        elif k == Qt.Key_Space:
            self.play_pause()
        elif k == Qt.Key_Escape:
            self.confirm_close()
        else:
            handled = False

        if not handled:
            super().keyPressEvent(event)

    # ---------- Close ----------
    def confirm_close(self):
        if not self.has_video():
            self.close()
            return
        ans = QMessageBox.question(
            self, "Confirm Exit", "Save and quit?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if ans == QMessageBox.Yes:
            try:
                self.save_csv()
            except Exception as e:
                QMessageBox.warning(self, "Save failed", str(e))
            self.close()
        elif ans == QMessageBox.No:
            self.close()
        else:
            pass

    def closeEvent(self, event):
        if not self.has_video():
            event.accept()
            return
        # Ensure user gets a chance to save
        ans = QMessageBox.question(
            self, "Confirm Exit", "Save and quit?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if ans == QMessageBox.Yes:
            try:
                self.save_csv()
            except Exception as e:
                QMessageBox.warning(self, "Save failed", str(e))
            if self.cap is not None:
                self.cap.release()
            event.accept()
        elif ans == QMessageBox.No:
            if self.cap is not None:
                self.cap.release()
            event.accept()
        else:
            event.ignore()


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_name = resolve_video_path()
    try:
        w = MainWindow(video_name)
    except Exception as e:
        QMessageBox.critical(None, "Failed to open video", str(e))
        w = MainWindow()

    w.showMaximized()
    sys.exit(app.exec_())
