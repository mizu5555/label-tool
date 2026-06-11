import os
import sys
import json
import time
import math
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem, QLabel,
    QHBoxLayout, QVBoxLayout, QSlider, QPushButton, QMessageBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
)

# ======================
# Constants / Settings
# ======================
KEY_LastFrame = 'z'
KEY_NextFrame = 'x'
KEY_Last50Frame = 'd'
KEY_Next50Frame = 'f'
KEY_ClearFrame = 'c'
KEY_Save = 's'
KEY_Quit = 'q'
KEY_Esc = 27
KEY_Up = 'i'
KEY_Left = 'j'
KEY_Down = 'k'
KEY_Right = 'l'

AIMBOT_RANGE = 50
LABEL_COLOR = (0, 0, 255)  # BGR
LABEL_SIZE = 3
TEXT_SIZE = 0.8
TEXT_COLOR = (0, 255, 255)  # BGR

HISTORY = 4  # history frames overlay
COLORS = [
    QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
    QColor(255, 165, 0), QColor(128, 0, 128), QColor(0, 255, 255),
    QColor(255, 192, 203), QColor(128, 128, 0)
]

COLUMNS = ['Frame', 'Visibility', 'X', 'Y']
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
        csv_dir = os.path.join(os.path.dirname(video_dir), 'csv')
        csv_name = os.path.join(csv_dir, os.path.splitext(f)[0] + '_ball.csv')
        flag = os.path.isfile(csv_name)
        print(f"{i+1:3d}: {f}{' (labeled)' if flag else ''}")
    idx = int(input('Enter the video number: '))
    return os.path.join(video_dir, vids[idx-1])

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
    video_dir = os.path.join(work_dir, 'video')

    if has_console():
        video_path = prompt_video_from_console(video_dir)
        if video_path:
            return video_path

    return choose_video_with_dialog(video_dir if os.path.isdir(video_dir) else work_dir)

# ======================
# Graphics View
# ======================
class VideoView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        if parent is not None:
            self.setSceneRect(0, 0, parent.frame_w, parent.frame_h)

    def map_to_frame_xy(self, event):
        pt = self.mapToScene(event.pos())
        return int(pt.x()), int(pt.y())

    def mousePressEvent(self, event):
        if self.window() is None:
            return super().mousePressEvent(event)
        mw = self.window()
        x, y = self.map_to_frame_xy(event)
        if 0 <= x < mw.frame_w and 0 <= y < mw.frame_h:
            if event.button() == Qt.LeftButton:
                mw.set_point_at(x, y)
            elif event.button() == Qt.RightButton:
                mw.aimbot_click(x, y)
        else:
            super().mousePressEvent(event)

    def wheelEvent(self, event):
        mw = self.window()
        if mw is None:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        new_val = mw.zoom_slider.value() + (1 if delta > 0 else -1)
        mw.zoom_slider.setValue(new_val)


# ======================
# Main Window
# ======================
class MainWindow(QMainWindow):
    def __init__(self, video_path: str):
        super().__init__()
        self.setWindowTitle("Label Tool (PyQt5)")

        # Video
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.frames: List[np.ndarray] = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        self.cap.release()
        if not self.frames:
            raise ValueError("Cannot load video or video has no frames.")

        self.total_frames = len(self.frames)
        self.frame_h, self.frame_w = self.frames[0].shape[:2]

        # CSV paths
        parent_dir = os.path.dirname(os.path.dirname(video_path))
        csv_dir = os.path.join(parent_dir, 'csv')
        ensure_dir(csv_dir)
        csv_file_name = os.path.splitext(os.path.basename(video_path))[0] + '_ball.csv'
        self.csv_path = os.path.join(csv_dir, csv_file_name)

        # DataFrame-like dict of dicts (index->row dict)
        self.df: Dict[int, Dict] = {}
        self.load_csv_if_any()

        # UI State
        self.scale = 1.0
        self.playback_speed = 1.0
        self.current_frame = 0
        self.brightness = 1.0
        self.brightness_step = 0.1
        self.aimbot_pending = False

        # Widgets
        self.video_view = VideoView(self)
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
        self.show_history = True
        self.history_checkbox = QCheckBox("Show last 4 frames")
        self.history_checkbox.setFont(font)
        self.history_checkbox.setChecked(True)
        self.history_checkbox.stateChanged.connect(self.on_history_toggled)

        # Frame/Status labels
        font.setPointSize(20)
        self.frame_label = QLabel(f"Frame: 0/{self.total_frames-1}")
        self.frame_label.setFont(font)
        self.event_label = QLabel("Event: 0")
        self.event_label.setFont(font)

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
        self.slider.setRange(0, self.total_frames - 1)
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
        left_layout.addSpacing(16)
        left_layout.addWidget(self.event_label)
        left_layout.addStretch(1)
        left_layout.addWidget(self.history_checkbox)

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
        h_layout.addWidget(self.video_view, 12)
        h_layout.addWidget(zoom_widget, 1)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(ctrl_panel)
        self.setCentralWidget(central)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        # Initial draw
        self.update_frame()
        self.video_view.resetTransform()
        self.video_view.fitInView(self.video_view.sceneRect(), Qt.KeepAspectRatio)

    # ---------- Data IO ----------
    def load_csv_if_any(self):
        if os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            if not set(COLUMNS).issubset(df.columns):
                raise ValueError(f"{self.csv_path} is missing columns from {COLUMNS}!")
            # Build dict by index
            self.df = {int(row['Frame']): {
                'Frame': int(row['Frame']),
                'Visibility': int(row['Visibility']),
                'X': float(row['X']),
                'Y': float(row['Y']),
                'Z': float(row.get('Z', 0.0)),
                'Event': int(row.get('Event', 0)),
                'Timestamp': float(row.get('Timestamp', 0.0)),
            } for _, row in df.iterrows()}
        else:
            self.df = {}
        # Ensure at least row 0 exists
        if 0 not in self.df:
            self.df[0] = empty_row(0, self.fps)

    def save_csv(self):
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

    # ---------- UI plumbing ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_view.resetTransform()
        self.video_view.fitInView(self.video_view.sceneRect(), Qt.KeepAspectRatio)

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
        self.update_frame()

    def on_brighten(self):
        self.brightness = min(self.brightness + self.brightness_step, 3.0)
        self.update_frame()

    def on_darken(self):
        self.brightness = max(self.brightness - self.brightness_step, 0.1)
        self.update_frame()

    def play_pause(self):
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
        if self.current_frame + 1 < self.total_frames:
            self.current_frame += 1
            self.update_frame()
        else:
            self.timer.stop()
            self.play_btn.setText("Play")

    def seek_frame(self, idx: int):
        idx = max(0, min(self.total_frames - 1, int(idx)))
        self.current_frame = idx
        self.update_frame()

    # ---------- Label ops ----------
    def ensure_row(self, idx: int):
        if idx not in self.df:
            self.df[idx] = empty_row(idx, self.fps)

    def set_point_at(self, x: int, y: int):
        self.ensure_row(self.current_frame)
        row = self.df[self.current_frame]
        row['Visibility'] = 1
        row['X'] = float(x)
        row['Y'] = float(y)
        self.update_frame()

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
        src = self.frames[self.current_frame].copy()
        display = cv2.convertScaleAbs(src, alpha=self.brightness, beta=0)

        # draw history with fading alpha
        marker_radius = max(1, int(min(self.frame_h, self.frame_w) * 0.002))
        font_scale = max(0.3, marker_radius / 4.0)
        thickness = max(1, marker_radius)

        if self.show_history:
            for h in range(1, HISTORY+1):
                idx = self.current_frame - h
                if idx < 0:
                    break
                row = self.df.get(idx)
                if not row or not row['Visibility']:
                    continue
                alpha = (HISTORY + 1 - h) / (HISTORY + 1)
                overlay = display.copy()
                cv2.circle(overlay, (int(row['X']), int(row['Y'])), marker_radius, LABEL_COLOR, -1)
                cv2.addWeighted(overlay, alpha, display, 1-alpha, 0, display)

        # current
        rowc = self.df.get(self.current_frame)
        if rowc and rowc['Visibility']:
            cv2.circle(display, (int(rowc['X']), int(rowc['Y'])), marker_radius, LABEL_COLOR, -1)

        # push to view
        h, w, ch = display.shape
        qimg = QImage(display.data, w, h, ch*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        self.video_view.pixmap_item.setPixmap(pix)
        self.video_view.scene.setSceneRect(0, 0, self.frame_w, self.frame_h)

        # labels
        ev = rowc['Event'] if rowc else 0
        self.frame_label.setText(f"Frame: {self.current_frame}/{self.total_frames-1}")
        self.event_label.setText(f"Event: {ev}")
        self.slider.setValue(self.current_frame)

    # ---------- Key handling ----------
    def keyPressEvent(self, event):
        k = event.key()
        text = event.text()
        handled = True

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
        elif text == KEY_Save:
            self.save_csv()
            QMessageBox.information(self, "Saved", f"Saved CSV to:\n{self.csv_path}")
        elif text == KEY_Quit:
            self.save_csv()
            self.close()
        elif text in ('0', '1', '2', '3'):
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
            event.accept()
        elif ans == QMessageBox.No:
            event.accept()
        else:
            event.ignore()


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_name = resolve_video_path()
    if not video_name:
        sys.exit(0)

    w = MainWindow(video_name)
    w.showMaximized()
    sys.exit(app.exec_())
