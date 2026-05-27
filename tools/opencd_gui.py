# Copyright (c) Open-CD. All rights reserved.
import os
import re
import signal
import sys

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt5.QtGui import QColor, QTextCharFormat
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog,
                             QFormLayout, QFrame, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QProgressBar,
                             QScrollArea,
                             QSizePolicy, QSpinBox, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CHANGE3D_ROOT = os.path.join(PROJECT_ROOT, 'projects',
                                         'change3d', 'Change3D')
DEFAULT_CHANGEDINO_ROOT = os.path.join(PROJECT_ROOT, 'projects',
                                       'changedino', 'ChangeDINO')


MODEL_PRESETS = {
    'ChangeFormer-B0 (LEVIR-CD)': {
        'type': 'opencd',
        'config':
        'configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py',
        'work_dir': 'results/changeformer/b0',
        'data_root': r'D:\Projects\change_detection\CD_datasets\Levir-CD'
    },
    'ChangeFormer-B1 (LEVIR-CD)': {
        'type': 'opencd',
        'config':
        'configs/changeformer/changeformer_mit-b1_256x256_40k_levircd.py',
        'work_dir': 'results/changeformer/b1',
        'data_root': r'D:\Projects\change_detection\CD_datasets\Levir-CD'
    },
    'Changer-R18 (LEVIR-CD)': {
        'type': 'opencd',
        'config': 'configs/changer/changer_ex_r18_512x512_40k_levircd.py',
        'work_dir': 'results/changer/r18',
        'data_root': r'D:\Projects\change_detection\CD_datasets\Levir-CD'
    },
    'Change3D-BCD': {
        'type': 'change3d',
        'task': 'bcd',
        'config': 'configs/change3d/change3d_bcd.py',
        'work_dir': 'results/change3d/bcd',
        'data_root': r'D:\Projects\change_detection\CD_datasets\CD_mine\11'
    },
    'Change3D-MCD': {
        'type': 'change3d',
        'task': 'mcd',
        'config': 'configs/change3d/change3d_mcd.py',
        'work_dir': 'results/change3d/mcd',
        'data_root': r'D:\Projects\change_detection\CD_datasets\CD_mine\11'
    },
    'ChangeDINO-BCD': {
        'type': 'changedino',
        'task': 'bcd',
        'config': 'configs/changedino/changedino_bcd.py',
        'work_dir': 'results/changedino/bcd',
        'data_root': r'D:\Projects\change_detection\CD_datasets\CD_mine\11'
    },
}


def abs_path(path):
    if not path:
        return ''
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


class PathRow(QWidget):

    def __init__(self, mode='file', placeholder=''):
        super().__init__()
        self.mode = mode
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton('浏览')
        self.button.clicked.connect(self.browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self):
        return self.edit.text().strip()

    def setText(self, text):
        self.edit.setText(text)

    def browse(self):
        if self.mode == 'dir':
            path = QFileDialog.getExistingDirectory(self, '选择文件夹',
                                                   self.text() or PROJECT_ROOT)
        elif self.mode == 'save_dir':
            path = QFileDialog.getExistingDirectory(self, '选择保存目录',
                                                   self.text() or PROJECT_ROOT)
        else:
            path, _ = QFileDialog.getOpenFileName(self, '选择文件',
                                                  self.text() or PROJECT_ROOT)
        if path:
            self.edit.setText(path)


class NoWheelSpinBox(QSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class OpenCDGui(QMainWindow):

    def __init__(self):
        super().__init__()
        self.process = None
        self.process_pid = None
        self.max_iters = 40000
        self.current_task = None
        self.active_log = None

        self.setWindowTitle('OpenCD 变化检测工具')
        self.resize(1280, 760)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 10, 14, 12)
        root_layout.setSpacing(8)

        header = self.build_header()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.make_scroll_tab(self.build_train_tab()), '模型训练')
        self.tabs.addTab(self.make_scroll_tab(self.build_predict_tab()), '模型推理')
        self.tabs.addTab(self.make_scroll_tab(self.build_big_tif_tab()), '大图推理')

        root_layout.addWidget(header)
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.apply_style()
        self.apply_train_preset()
        self.apply_predict_preset()
        self.apply_big_preset()

    def make_scroll_tab(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setWidget(widget)
        return scroll

    def build_header(self):
        header = QFrame()
        header.setObjectName('Header')
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 0, 18, 0)
        title = QLabel('OpenCD 变化检测工具')
        title.setObjectName('HeaderTitle')
        user = QLabel('用户:')
        user.setObjectName('HeaderUser')
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(user)
        return header

    def build_train_tab(self):
        tab = QWidget()
        tab.setMinimumHeight(1120)
        layout = QVBoxLayout(tab)

        env_box = QGroupBox('环境设置')
        env_box.setMinimumHeight(155)
        env_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        env_form = QFormLayout(env_box)
        self.python_row = PathRow('file')
        self.python_row.setText(sys.executable)
        self.root_row = PathRow('dir')
        self.root_row.setText(PROJECT_ROOT)
        self.train_script_row = PathRow('file')
        self.train_script_row.setText(os.path.join(PROJECT_ROOT, 'tools',
                                                   'train.py'))
        env_form.addRow('Python 解释器:', self.python_row)
        env_form.addRow('项目根目录:', self.root_row)
        env_form.addRow('训练脚本:', self.train_script_row)

        train_box = QGroupBox('训练设置')
        train_box.setMinimumHeight(345)
        train_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        train_grid = QGridLayout(train_box)
        train_grid.setContentsMargins(16, 26, 16, 16)
        train_grid.setHorizontalSpacing(10)
        train_grid.setVerticalSpacing(12)
        self.train_preset = QComboBox()
        self.train_preset.addItems(MODEL_PRESETS.keys())
        self.train_preset.currentIndexChanged.connect(self.apply_train_preset)
        self.train_config_row = PathRow('file')
        self.train_workdir_row = PathRow('dir')
        self.train_data_root_row = PathRow('dir')
        self.train_checkpoint_row = PathRow('file')
        self.max_iters_spin = NoWheelSpinBox()
        self.max_iters_spin.setRange(1, 10000000)
        self.max_iters_spin.setValue(40000)
        self.batch_size_spin = NoWheelSpinBox()
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(8)
        self.num_workers_spin = NoWheelSpinBox()
        self.num_workers_spin.setRange(0, 128)
        self.num_workers_spin.setValue(8)
        self.resume_check = QCheckBox('断点续训')
        self.amp_check = QCheckBox('混合精度 (AMP)')

        train_grid.addWidget(QLabel('模型预设:'), 0, 0)
        train_grid.addWidget(self.train_preset, 0, 1, 1, 3)
        train_grid.addWidget(QLabel('配置文件:'), 1, 0)
        train_grid.addWidget(self.train_config_row, 1, 1, 1, 3)
        train_grid.addWidget(QLabel('输出目录:'), 2, 0)
        train_grid.addWidget(self.train_workdir_row, 2, 1, 1, 3)
        train_grid.addWidget(QLabel('数据集目录:'), 3, 0)
        train_grid.addWidget(self.train_data_root_row, 3, 1, 1, 3)
        self.train_checkpoint_label = QLabel('预训练权重:')
        train_grid.addWidget(self.train_checkpoint_label, 4, 0)
        train_grid.addWidget(self.train_checkpoint_row, 4, 1, 1, 3)
        self.train_iters_label = QLabel('迭代次数:')
        train_grid.addWidget(self.train_iters_label, 5, 0)
        train_grid.addWidget(self.max_iters_spin, 5, 1)
        train_grid.addWidget(QLabel('Batch Size:'), 5, 2)
        train_grid.addWidget(self.batch_size_spin, 5, 3)
        train_grid.addWidget(QLabel('Num Workers:'), 6, 0)
        train_grid.addWidget(self.num_workers_spin, 6, 1)
        train_grid.addWidget(self.resume_check, 6, 2)
        train_grid.addWidget(self.amp_check, 6, 3)
        for row in range(7):
            train_grid.setRowMinimumHeight(row, 38)

        self.change3d_train_box = QGroupBox('Change3D 训练参数')
        self.change3d_train_box.setMinimumHeight(195)
        self.change3d_train_box.setSizePolicy(QSizePolicy.Expanding,
                                              QSizePolicy.Fixed)
        self.c3d_train_grid = QGridLayout(self.change3d_train_box)
        c3d_grid = self.c3d_train_grid
        c3d_grid.setContentsMargins(16, 26, 16, 16)
        c3d_grid.setHorizontalSpacing(10)
        c3d_grid.setVerticalSpacing(12)
        self.c3d_root_row = PathRow('dir')
        self.c3d_root_row.setText(DEFAULT_CHANGE3D_ROOT)
        self.c3d_pretrained_row = PathRow('file')
        self.c3d_pretrained_row.setText(
            os.path.join(DEFAULT_CHANGE3D_ROOT, 'model', 'X3D_L.pyth'))
        self.c3d_num_classes_spin = NoWheelSpinBox()
        self.c3d_num_classes_spin.setRange(2, 512)
        self.c3d_num_classes_spin.setValue(31)
        c3d_grid.addWidget(QLabel('Change3D 源码目录:'), 0, 0)
        c3d_grid.addWidget(self.c3d_root_row, 0, 1, 1, 3)
        c3d_grid.addWidget(QLabel('X3D 预训练权重:'), 1, 0)
        c3d_grid.addWidget(self.c3d_pretrained_row, 1, 1, 1, 3)
        self.c3d_num_classes_label = QLabel('MCD 类别数:')
        c3d_grid.addWidget(self.c3d_num_classes_label, 2, 0)
        c3d_grid.addWidget(self.c3d_num_classes_spin, 2, 1)
        for row in range(3):
            c3d_grid.setRowMinimumHeight(row, 38)

        self.changedino_train_box = QGroupBox('ChangeDINO 训练参数')
        self.changedino_train_box.setMinimumHeight(150)
        self.changedino_train_box.setSizePolicy(QSizePolicy.Expanding,
                                                QSizePolicy.Fixed)
        cdino_grid = QGridLayout(self.changedino_train_box)
        cdino_grid.setContentsMargins(16, 26, 16, 16)
        cdino_grid.setHorizontalSpacing(10)
        cdino_grid.setVerticalSpacing(12)
        self.cdino_root_row = PathRow('dir')
        self.cdino_root_row.setText(DEFAULT_CHANGEDINO_ROOT)
        self.cdino_dataset_name_row = QLineEdit()
        self.cdino_dataset_name_row.setText('CD_mine/11')
        self.cdino_epochs_spin = NoWheelSpinBox()
        self.cdino_epochs_spin.setRange(1, 10000)
        self.cdino_epochs_spin.setValue(100)
        cdino_grid.addWidget(QLabel('ChangeDINO 源码目录:'), 0, 0)
        cdino_grid.addWidget(self.cdino_root_row, 0, 1, 1, 3)
        cdino_grid.addWidget(QLabel('数据集名称:'), 1, 0)
        cdino_grid.addWidget(self.cdino_dataset_name_row, 1, 1)
        cdino_grid.addWidget(QLabel('训练 Epoch:'), 1, 2)
        cdino_grid.addWidget(self.cdino_epochs_spin, 1, 3)
        for row in range(2):
            cdino_grid.setRowMinimumHeight(row, 38)

        controls = QHBoxLayout()
        self.start_train_button = QPushButton('开始训练')
        self.start_train_button.setObjectName('StartButton')
        self.stop_button = QPushButton('停止任务')
        self.stop_button.setObjectName('StopButton')
        self.clear_button = QPushButton('清空日志')
        self.auto_scroll_check = QCheckBox('自动滚动')
        self.auto_scroll_check.setChecked(True)
        self.start_train_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_process)
        self.clear_button.clicked.connect(self.clear_log)
        controls.addWidget(self.start_train_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(self.auto_scroll_check)
        controls.addWidget(self.clear_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 40000)
        self.progress.setValue(0)
        self.progress.setFormat('进度 %v / %m   (%p%)')

        self.train_log = self.create_log_text()
        self.active_log = self.train_log

        layout.addWidget(env_box)
        layout.addWidget(train_box)
        layout.addWidget(self.change3d_train_box)
        layout.addWidget(self.changedino_train_box)
        layout.addLayout(controls)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel('训练日志:'))
        layout.addWidget(self.train_log)
        layout.addStretch(1)
        return tab

    def build_predict_tab(self):
        tab = QWidget()
        tab.setMinimumHeight(760)
        layout = QVBoxLayout(tab)

        box = QGroupBox('推理设置')
        form = QFormLayout(box)
        self.pred_preset = QComboBox()
        self.pred_preset.addItems(MODEL_PRESETS.keys())
        self.pred_preset.currentIndexChanged.connect(self.apply_predict_preset)
        self.pred_config_row = PathRow('file')
        self.pred_checkpoint_row = PathRow('file')
        self.pred_data_root_row = PathRow('dir')
        self.pred_workdir_row = PathRow('dir')
        self.pred_showdir_row = PathRow('dir')
        form.addRow('模型预设:', self.pred_preset)
        form.addRow('配置文件:', self.pred_config_row)
        form.addRow('权重文件:', self.pred_checkpoint_row)
        form.addRow('测试文件夹:', self.pred_data_root_row)
        form.addRow('工作目录:', self.pred_workdir_row)
        form.addRow('可视化输出目录:', self.pred_showdir_row)

        controls = QHBoxLayout()
        run = QPushButton('开始推理')
        run.setObjectName('StartButton')
        run.clicked.connect(self.start_prediction)
        stop = QPushButton('停止任务')
        stop.setObjectName('StopButton')
        stop.clicked.connect(self.stop_process)
        controls.addWidget(run)
        controls.addWidget(stop)
        controls.addStretch(1)

        self.pred_progress = QProgressBar()
        self.pred_progress.setRange(0, 100)
        self.pred_progress.setValue(0)
        self.pred_progress.setFormat('进度 %v / %m   (%p%)')

        layout.addWidget(box)
        layout.addLayout(controls)
        layout.addWidget(self.pred_progress)
        layout.addWidget(QLabel('推理日志:'))
        self.pred_log = self.create_log_text()
        layout.addWidget(self.pred_log, 1)
        return tab

    def build_big_tif_tab(self):
        tab = QWidget()
        tab.setMinimumHeight(900)
        layout = QVBoxLayout(tab)
        box = QGroupBox('大图推理')
        grid = QGridLayout(box)
        self.big_preset = QComboBox()
        self.big_preset.addItems(MODEL_PRESETS.keys())
        self.big_preset.currentIndexChanged.connect(self.apply_big_preset)
        self.big_config_row = PathRow('file')
        self.big_checkpoint_row = PathRow('file')
        self.big_tif_a_row = PathRow('file')
        self.big_tif_b_row = PathRow('file')
        self.big_output_row = PathRow('dir')
        self.big_tile_spin = NoWheelSpinBox()
        self.big_tile_spin.setRange(64, 4096)
        self.big_tile_spin.setValue(512)
        self.big_overlap_spin = NoWheelSpinBox()
        self.big_overlap_spin.setRange(0, 2048)
        self.big_overlap_spin.setValue(128)
        self.big_batch_spin = NoWheelSpinBox()
        self.big_batch_spin.setRange(1, 128)
        self.big_batch_spin.setValue(1)
        self.big_threshold_spin = NoWheelDoubleSpinBox()
        self.big_threshold_spin.setRange(0.0, 1.0)
        self.big_threshold_spin.setSingleStep(0.05)
        self.big_threshold_spin.setValue(0.5)
        self.big_hole_spin = NoWheelSpinBox()
        self.big_hole_spin.setRange(0, 100000000)
        self.big_hole_spin.setValue(2000)
        self.big_object_spin = NoWheelSpinBox()
        self.big_object_spin.setRange(0, 100000000)
        self.big_object_spin.setValue(100)
        self.big_fill_holes_check = QCheckBox('填充孔洞')
        self.big_save_shp_check = QCheckBox('保存 Shapefile')

        grid.addWidget(QLabel('模型预设:'), 0, 0)
        grid.addWidget(self.big_preset, 0, 1, 1, 3)
        grid.addWidget(QLabel('配置文件:'), 1, 0)
        grid.addWidget(self.big_config_row, 1, 1, 1, 3)
        grid.addWidget(QLabel('权重文件:'), 2, 0)
        grid.addWidget(self.big_checkpoint_row, 2, 1, 1, 3)
        grid.addWidget(QLabel('第一时相 TIF:'), 3, 0)
        grid.addWidget(self.big_tif_a_row, 3, 1, 1, 3)
        grid.addWidget(QLabel('第二时相 TIF:'), 4, 0)
        grid.addWidget(self.big_tif_b_row, 4, 1, 1, 3)
        grid.addWidget(QLabel('输出目录:'), 5, 0)
        grid.addWidget(self.big_output_row, 5, 1, 1, 3)
        grid.addWidget(QLabel('Tile Size:'), 6, 0)
        grid.addWidget(self.big_tile_spin, 6, 1)
        grid.addWidget(QLabel('Overlap:'), 6, 2)
        grid.addWidget(self.big_overlap_spin, 6, 3)
        grid.addWidget(QLabel('Batch Size:'), 7, 0)
        grid.addWidget(self.big_batch_spin, 7, 1)
        grid.addWidget(QLabel('阈值:'), 7, 2)
        grid.addWidget(self.big_threshold_spin, 7, 3)
        grid.addWidget(QLabel('孔洞面积:'), 8, 0)
        grid.addWidget(self.big_hole_spin, 8, 1)
        grid.addWidget(QLabel('最小目标:'), 8, 2)
        grid.addWidget(self.big_object_spin, 8, 3)
        grid.addWidget(self.big_fill_holes_check, 9, 1)
        grid.addWidget(self.big_save_shp_check, 9, 2)

        controls = QHBoxLayout()
        run = QPushButton('开始大图推理')
        run.setObjectName('StartButton')
        run.clicked.connect(self.start_big_tif_inference)
        stop = QPushButton('停止任务')
        stop.setObjectName('StopButton')
        stop.clicked.connect(self.stop_process)
        controls.addWidget(run)
        controls.addWidget(stop)
        controls.addStretch(1)

        self.big_progress = QProgressBar()
        self.big_progress.setRange(0, 100)
        self.big_progress.setValue(0)
        self.big_progress.setFormat('大图进度 %v / %m   (%p%)')

        layout.addWidget(box)
        layout.addLayout(controls)
        layout.addWidget(self.big_progress)
        layout.addWidget(QLabel('大图推理日志:'))
        self.big_log = self.create_log_text()
        layout.addWidget(self.big_log, 1)
        return tab

    def build_change3d_tab(self):
        tab = QWidget()
        tab.setMinimumHeight(980)
        layout = QVBoxLayout(tab)

        box = QGroupBox('Change3D')
        grid = QGridLayout(box)
        self.c3d_root_row = PathRow('dir')
        self.c3d_root_row.setText(DEFAULT_CHANGE3D_ROOT)
        self.c3d_mode_combo = QComboBox()
        self.c3d_mode_combo.addItems([
            'BCD train',
            'MCD train',
            'BCD big tif inference',
            'MCD big tif inference',
        ])
        self.c3d_dataset_row = PathRow('dir')
        self.c3d_dataset_row.setText(
            r'D:\Projects\change_detection\CD_datasets\CD_mine\11')
        self.c3d_workdir_row = PathRow('dir')
        self.c3d_workdir_row.setText(
            os.path.join(PROJECT_ROOT, 'results', 'change3d'))
        self.c3d_pretrained_row = PathRow('file')
        self.c3d_pretrained_row.setText(
            os.path.join(DEFAULT_CHANGE3D_ROOT, 'model', 'X3D_L.pyth'))
        self.c3d_resume_row = PathRow('file')
        self.c3d_max_steps_spin = NoWheelSpinBox()
        self.c3d_max_steps_spin.setRange(1, 10000)
        self.c3d_max_steps_spin.setValue(120)
        self.c3d_batch_spin = NoWheelSpinBox()
        self.c3d_batch_spin.setRange(1, 512)
        self.c3d_batch_spin.setValue(4)
        self.c3d_workers_spin = NoWheelSpinBox()
        self.c3d_workers_spin.setRange(0, 64)
        self.c3d_workers_spin.setValue(0)
        self.c3d_num_classes_spin = NoWheelSpinBox()
        self.c3d_num_classes_spin.setRange(2, 512)
        self.c3d_num_classes_spin.setValue(31)

        self.c3d_tif1_row = PathRow('file')
        self.c3d_tif2_row = PathRow('file')
        self.c3d_pt_row = PathRow('file')
        self.c3d_outdir_row = PathRow('dir')
        self.c3d_outdir_row.setText(
            os.path.join(PROJECT_ROOT, 'results', 'change3d', 'infer'))
        self.c3d_tile_spin = NoWheelSpinBox()
        self.c3d_tile_spin.setRange(64, 4096)
        self.c3d_tile_spin.setValue(512)
        self.c3d_overlap_spin = NoWheelSpinBox()
        self.c3d_overlap_spin.setRange(0, 2048)
        self.c3d_overlap_spin.setValue(128)
        self.c3d_infer_batch_spin = NoWheelSpinBox()
        self.c3d_infer_batch_spin.setRange(1, 128)
        self.c3d_infer_batch_spin.setValue(8)
        self.c3d_threshold_spin = NoWheelDoubleSpinBox()
        self.c3d_threshold_spin.setRange(0.0, 1.0)
        self.c3d_threshold_spin.setSingleStep(0.05)
        self.c3d_threshold_spin.setValue(0.5)

        grid.addWidget(QLabel('Change3D 源码目录:'), 0, 0)
        grid.addWidget(self.c3d_root_row, 0, 1, 1, 3)
        grid.addWidget(QLabel('任务类型:'), 1, 0)
        grid.addWidget(self.c3d_mode_combo, 1, 1, 1, 3)
        grid.addWidget(QLabel('数据集目录:'), 2, 0)
        grid.addWidget(self.c3d_dataset_row, 2, 1, 1, 3)
        grid.addWidget(QLabel('输出目录:'), 3, 0)
        grid.addWidget(self.c3d_workdir_row, 3, 1, 1, 3)
        grid.addWidget(QLabel('X3D 预训练权重:'), 4, 0)
        grid.addWidget(self.c3d_pretrained_row, 4, 1, 1, 3)
        grid.addWidget(QLabel('恢复训练权重:'), 5, 0)
        grid.addWidget(self.c3d_resume_row, 5, 1, 1, 3)
        grid.addWidget(QLabel('训练 Epoch:'), 6, 0)
        grid.addWidget(self.c3d_max_steps_spin, 6, 1)
        grid.addWidget(QLabel('Batch Size:'), 6, 2)
        grid.addWidget(self.c3d_batch_spin, 6, 3)
        grid.addWidget(QLabel('Workers:'), 7, 0)
        grid.addWidget(self.c3d_workers_spin, 7, 1)
        grid.addWidget(QLabel('MCD 类别数:'), 7, 2)
        grid.addWidget(self.c3d_num_classes_spin, 7, 3)
        grid.addWidget(QLabel('第一时相 TIF:'), 8, 0)
        grid.addWidget(self.c3d_tif1_row, 8, 1, 1, 3)
        grid.addWidget(QLabel('第二时相 TIF:'), 9, 0)
        grid.addWidget(self.c3d_tif2_row, 9, 1, 1, 3)
        grid.addWidget(QLabel('TorchScript .pt:'), 10, 0)
        grid.addWidget(self.c3d_pt_row, 10, 1, 1, 3)
        grid.addWidget(QLabel('推理输出目录:'), 11, 0)
        grid.addWidget(self.c3d_outdir_row, 11, 1, 1, 3)
        grid.addWidget(QLabel('Tile Size:'), 12, 0)
        grid.addWidget(self.c3d_tile_spin, 12, 1)
        grid.addWidget(QLabel('Overlap:'), 12, 2)
        grid.addWidget(self.c3d_overlap_spin, 12, 3)
        grid.addWidget(QLabel('推理 Batch:'), 13, 0)
        grid.addWidget(self.c3d_infer_batch_spin, 13, 1)
        grid.addWidget(QLabel('BCD 阈值:'), 13, 2)
        grid.addWidget(self.c3d_threshold_spin, 13, 3)

        controls = QHBoxLayout()
        run = QPushButton('运行 Change3D')
        run.setObjectName('StartButton')
        run.clicked.connect(self.start_change3d)
        stop = QPushButton('停止任务')
        stop.setObjectName('StopButton')
        stop.clicked.connect(self.stop_process)
        controls.addWidget(run)
        controls.addWidget(stop)
        controls.addStretch(1)

        self.c3d_progress = QProgressBar()
        self.c3d_progress.setRange(0, 100)
        self.c3d_progress.setValue(0)
        self.c3d_progress.setFormat('Change3D %v / %m   (%p%)')

        layout.addWidget(box)
        layout.addLayout(controls)
        layout.addWidget(self.c3d_progress)
        layout.addWidget(QLabel('Change3D 日志:'))
        self.c3d_log = self.create_log_text()
        layout.addWidget(self.c3d_log, 1)
        return tab

    def create_log_text(self):
        log = QTextEdit()
        log.setReadOnly(True)
        log.setObjectName('LogText')
        log.setMinimumHeight(220)
        log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        log.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return log

    def apply_train_preset(self):
        preset = MODEL_PRESETS[self.train_preset.currentText()]
        config = abs_path(preset['config'])
        work_dir = abs_path(preset['work_dir'])
        data_root = abs_path(preset.get('data_root', ''))
        self.train_config_row.setText(config)
        self.train_workdir_row.setText(work_dir)
        self.train_data_root_row.setText(data_root)
        is_change3d = preset.get('type') == 'change3d'
        is_changedino = preset.get('type') == 'changedino'
        if hasattr(self, 'change3d_train_box'):
            self.change3d_train_box.setVisible(is_change3d)
            self.changedino_train_box.setVisible(is_changedino)
            self.amp_check.setEnabled(not is_change3d)
            self.num_workers_spin.setEnabled(not (is_change3d or
                                                  is_changedino))
            if is_change3d or is_changedino:
                self.num_workers_spin.setValue(0)
            self.train_iters_label.setText(
                '训练 Epoch:' if is_change3d else '迭代次数:')
            self.max_iters_spin.setRange(1, 10000 if is_change3d else 1000000)
            self.max_iters_spin.setValue(120 if is_change3d else 40000)
            self.train_checkpoint_label.setText(
                '恢复训练权重:' if (is_change3d or is_changedino) else '预训练权重:')
            self.train_checkpoint_row.setEnabled(True)
            is_mcd = preset.get('task') == 'mcd'
            self.c3d_num_classes_label.setVisible(is_mcd)
            self.c3d_num_classes_spin.setVisible(is_mcd)
            self.c3d_train_grid.setRowMinimumHeight(2, 38 if is_mcd else 0)
            self.change3d_train_box.setMinimumHeight(195 if is_mcd else 150)
            if is_changedino:
                self.cdino_dataset_name_row.setText(self.changedino_dataset_name(data_root))
            self.train_script_row.setText(
                os.path.join(PROJECT_ROOT, 'tools', 'train.py'))
    def apply_predict_preset(self):
        preset = MODEL_PRESETS[self.pred_preset.currentText()]
        config = abs_path(preset['config'])
        work_dir = abs_path(preset['work_dir'])
        data_root = abs_path(preset.get('data_root', ''))
        self.pred_config_row.setText(config)
        self.pred_data_root_row.setText(os.path.join(data_root, 'test'))
        self.pred_workdir_row.setText(work_dir)
        self.pred_showdir_row.setText(os.path.join(work_dir, 'show'))
        self.pred_checkpoint_row.setText('')

    def apply_big_preset(self):
        preset = MODEL_PRESETS[self.big_preset.currentText()]
        config = abs_path(preset['config'])
        work_dir = abs_path(preset['work_dir'])
        self.big_config_row.setText(config)
        self.big_checkpoint_row.setText('')
        self.big_output_row.setText(os.path.join(work_dir, 'big_tif'))
        model_type = preset.get('type')
        if model_type in ('change3d', 'changedino'):
            self.big_batch_spin.setValue(8 if model_type == 'change3d' else 16)
            self.big_threshold_spin.setValue(0.5)
            self.big_hole_spin.setValue(200000)
            self.big_object_spin.setValue(800)
            self.big_fill_holes_check.setChecked(False)
        else:
            self.big_batch_spin.setValue(1)
            self.big_threshold_spin.setValue(0.5)
            self.big_hole_spin.setValue(2000)
            self.big_object_spin.setValue(100)
            self.big_fill_holes_check.setChecked(False)

    def changedino_dataset_name(self, data_root):
        norm = os.path.normpath(data_root)
        marker = os.path.normpath(r'D:\Projects\change_detection\CD_datasets')
        if norm.lower().startswith(marker.lower() + os.sep.lower()):
            return os.path.relpath(norm, marker).replace('\\', '/')
        return os.path.basename(norm)

    def is_changedino_config(self, config):
        return 'changedino' in config.replace('\\', '/').lower()

    def is_change3d_config(self, config):
        return 'change3d' in config.replace('\\', '/').lower()

    def changedino_dataset_parts(self, split_root):
        split_root = os.path.normpath(split_root)
        test_phase = os.path.basename(split_root)
        dataset_root = os.path.dirname(split_root)
        dataset_name = self.changedino_dataset_name(dataset_root)
        dataroot = dataset_root
        if dataset_name:
            suffix = os.path.normpath(dataset_name)
            norm_data = os.path.normpath(dataset_root)
            if norm_data.lower().endswith(suffix.lower()):
                dataroot = norm_data[:-len(suffix)].rstrip('\\/')
        if not dataroot:
            dataroot = os.path.dirname(dataset_root)
            dataset_name = os.path.basename(dataset_root)
        return dataroot, dataset_name, test_phase

    def command_base(self):
        python = self.python_row.text() or sys.executable
        root = self.root_row.text() or PROJECT_ROOT
        return python, root

    def start_training(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, '任务正在运行', '已有任务在运行，请先停止当前任务。')
            return

        preset = MODEL_PRESETS[self.train_preset.currentText()]
        if preset.get('type') == 'change3d':
            self.start_change3d_training(preset)
            return
        if preset.get('type') == 'changedino':
            self.start_changedino_training(preset)
            return

        python, root = self.command_base()
        script = self.train_script_row.text()
        config = self.train_config_row.text()
        work_dir = self.train_workdir_row.text()
        data_root = self.train_data_root_row.text()
        if data_root and not self.validate_train_dataset(data_root):
            return
        self.max_iters = self.max_iters_spin.value()
        self.progress.setMaximum(self.max_iters)
        self.progress.setValue(0)
        self.progress.setFormat('进度 %v / %m   (%p%)')
        self.current_task = 'train'
        self.active_log = self.train_log

        args = [script, config, '--work-dir', work_dir]
        args.extend([
            '--cfg-options',
            f'train_cfg.max_iters={self.max_iters}',
            f'train_cfg.val_interval={min(4000, self.max_iters)}',
            f'default_hooks.checkpoint.interval={min(4000, self.max_iters)}',
            f'train_dataloader.batch_size={self.batch_size_spin.value()}',
            f'train_dataloader.num_workers={self.num_workers_spin.value()}',
            f'train_dataloader.dataset.data_root={data_root}',
            f'val_dataloader.dataset.data_root={data_root}',
            f'test_dataloader.dataset.data_root={data_root}'
        ])
        if self.resume_check.isChecked():
            args.append('--resume')
        if self.amp_check.isChecked():
            args.append('--amp')
        self.start_process(python, args, root)

    def start_change3d_training(self, preset):
        python, root = self.command_base()
        change3d_root = self.c3d_root_row.text()
        if not os.path.isdir(change3d_root):
            QMessageBox.warning(self, 'Change3D',
                                'Change3D 源码目录不存在。')
            return

        data_root = self.train_data_root_row.text()
        if data_root and not self.validate_train_dataset(data_root):
            return

        max_epochs = self.max_iters_spin.value()
        self.max_iters = max_epochs
        self.progress.setMaximum(max_epochs)
        self.progress.setValue(0)
        self.progress.setFormat('Epoch %v / %m')
        self.current_task = 'change3d_train'
        self.active_log = self.train_log

        args = [
            os.path.join(root, 'tools', 'train.py'),
            self.train_config_row.text(),
            '--work-dir',
            self.train_workdir_row.text(),
            '--cfg-options',
            f'change3d.change3d_root={change3d_root}',
            f'change3d.file_root={data_root}',
            f'change3d.pretrained={self.c3d_pretrained_row.text()}',
            'change3d.max_steps=-1',
            f'change3d.max_epochs={max_epochs}',
            f'change3d.batch_size={self.batch_size_spin.value()}',
            'change3d.num_workers=0',
            'change3d.gpu_id=0',
            'change3d.devices=1',
            'change3d.strategy=auto',
        ]
        if preset.get('task') == 'mcd':
            args.append(
                f'change3d.num_class={self.c3d_num_classes_spin.value()}')
        if self.resume_check.isChecked() and self.train_checkpoint_row.text():
            args.append(f'change3d.resume={self.train_checkpoint_row.text()}')
        self.start_process(python, args, root)

    def start_changedino_training(self, preset):
        python, root = self.command_base()
        changedino_root = self.cdino_root_row.text()
        if not os.path.isdir(changedino_root):
            QMessageBox.warning(self, 'ChangeDINO',
                                'ChangeDINO 源码目录不存在。')
            return

        data_root = self.train_data_root_row.text()
        if data_root and not self.validate_train_dataset(data_root):
            return

        dataset_name = self.cdino_dataset_name_row.text().strip()
        dataroot = data_root
        if dataset_name:
            suffix = os.path.normpath(dataset_name)
            norm_data = os.path.normpath(data_root)
            if norm_data.lower().endswith(suffix.lower()):
                dataroot = norm_data[:-len(suffix)].rstrip('\\/')
        if not dataroot:
            dataroot = os.path.dirname(data_root)
            dataset_name = os.path.basename(data_root)

        self.max_iters = self.cdino_epochs_spin.value()
        self.progress.setMaximum(self.max_iters)
        self.progress.setValue(0)
        self.progress.setFormat('Epoch %v / %m')
        self.current_task = 'changedino_train'
        self.active_log = self.train_log

        args = [
            os.path.join(root, 'tools', 'train.py'),
            self.train_config_row.text(),
            '--work-dir',
            self.train_workdir_row.text(),
            '--cfg-options',
            f'changedino.changedino_root={changedino_root}',
            f'changedino.dataroot={dataroot}',
            f'changedino.dataset={dataset_name}',
            f'changedino.checkpoint_dir={self.train_workdir_row.text()}',
            f'changedino.result_dir={os.path.join(self.train_workdir_row.text(), "results")}',
            f'changedino.num_epochs={self.cdino_epochs_spin.value()}',
            f'changedino.batch_size={self.batch_size_spin.value()}',
            'changedino.num_workers=0',
            'changedino.gpu_ids=0',
            'changedino.devices=1',
            'changedino.strategy=auto',
        ]
        if self.resume_check.isChecked() and self.train_checkpoint_row.text():
            args.append(f'changedino.resume={self.train_checkpoint_row.text()}')
        self.start_process(python, args, root)

    def validate_train_dataset(self, data_root):
        required_dirs = []
        for split in ('train', 'val', 'test'):
            for sub_dir in ('A', 'B', 'label'):
                required_dirs.append(os.path.join(data_root, split, sub_dir))
        missing = [path for path in required_dirs if not os.path.isdir(path)]
        if missing:
            QMessageBox.warning(
                self, '训练数据集格式不正确',
                '训练数据集目录需要满足:\n'
                'train/A, train/B, train/label\n'
                'val/A, val/B, val/label\n'
                'test/A, test/B, test/label\n\n'
                '缺失目录:\n' + '\n'.join(missing))
            return False
        return True

    def start_prediction(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, '任务正在运行', '已有任务在运行，请先停止当前任务。')
            return

        python, root = self.command_base()
        test_root = self.pred_data_root_row.text()
        if test_root and not self.validate_test_dataset(test_root):
            return
        self.active_log = self.pred_log
        config = self.pred_config_row.text()
        args = [
            os.path.join(root, 'tools', 'test.py'),
            config,
            self.pred_checkpoint_row.text(),
        ]
        if self.pred_workdir_row.text():
            args.extend(['--work-dir', self.pred_workdir_row.text()])
        if self.pred_showdir_row.text():
            args.extend(['--show-dir', self.pred_showdir_row.text()])
        if test_root:
            if self.is_changedino_config(config):
                dataroot, dataset_name, test_phase = (
                    self.changedino_dataset_parts(test_root))
                args.extend([
                    '--cfg-options',
                    f'changedino.dataroot={dataroot}',
                    f'changedino.dataset={dataset_name}',
                    f'changedino.test_phase={test_phase}',
                    f'changedino.checkpoint_dir={self.pred_workdir_row.text()}',
                    f'changedino.result_dir={self.pred_showdir_row.text()}',
                    f'changedino.batch_size={self.batch_size_spin.value()}',
                    'changedino.num_workers=0',
                    'changedino.gpu_ids=0',
                ])
            else:
                data_root = os.path.dirname(test_root)
                split_name = os.path.basename(os.path.normpath(test_root))
                args.extend([
                    '--cfg-options',
                    f'test_dataloader.dataset.data_root={data_root}',
                    f'test_dataloader.dataset.data_prefix.img_path_from={split_name}/A',
                    f'test_dataloader.dataset.data_prefix.img_path_to={split_name}/B',
                    f'test_dataloader.dataset.data_prefix.seg_map_path={split_name}/label'
                ])
        self.current_task = 'test'
        self.pred_progress.setRange(0, 100)
        self.pred_progress.setValue(0)
        self.start_process(python, args, root)

    def validate_test_dataset(self, test_root):
        required_dirs = [
            os.path.join(test_root, 'A'),
            os.path.join(test_root, 'B'),
            os.path.join(test_root, 'label'),
        ]
        missing = [path for path in required_dirs if not os.path.isdir(path)]
        if missing:
            QMessageBox.warning(
                self, '测试数据集格式不正确',
                '测试数据集目录需要满足:\n'
                'test/A\n'
                'test/B\n'
                'test/label\n\n'
                '缺失目录:\n' + '\n'.join(missing))
            return False

        image_names_a = {
            name for name in os.listdir(required_dirs[0])
            if os.path.isfile(os.path.join(required_dirs[0], name))
        }
        image_names_b = {
            name for name in os.listdir(required_dirs[1])
            if os.path.isfile(os.path.join(required_dirs[1], name))
        }
        if image_names_a != image_names_b:
            QMessageBox.warning(self, '测试数据集格式不正确',
                                'test/A 和 test/B 中的文件名必须一一对应。')
            return False
        return True

    def start_big_tif_inference(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, '任务正在运行', '已有任务在运行，请先停止当前任务。')
            return

        if self.big_overlap_spin.value() >= self.big_tile_spin.value():
            QMessageBox.warning(self, '参数错误', 'Overlap 必须小于 Tile Size。')
            return

        python, root = self.command_base()
        config = self.big_config_row.text()
        if self.is_change3d_config(config):
            command = 'infer-mcd' if 'mcd' in config.replace('\\', '/').lower() else 'infer-bcd'
            args = [
                os.path.join(root, 'projects', 'change3d',
                             'run_change3d.py'),
                command,
                '--change3d-root',
                os.path.join(root, 'projects', 'change3d', 'Change3D'),
                '--tif1',
                self.big_tif_a_row.text(),
                '--tif2',
                self.big_tif_b_row.text(),
                '--pt-path',
                self.big_checkpoint_row.text(),
                '--out-dir',
                self.big_output_row.text(),
                '--out-name',
                'change3d',
                '--tile-size',
                str(self.big_tile_spin.value()),
                '--overlap',
                str(self.big_overlap_spin.value()),
                '--batch-size',
                str(self.big_batch_spin.value()),
                '--gpu-id',
                '0',
                '--device',
                'cuda',
                '--remove-holes-area',
                str(self.big_hole_spin.value()),
                '--remove-objects-min-size',
                str(self.big_object_spin.value()),
                '--min-area-mu',
                '0.1',
            ]
            if command == 'infer-bcd':
                args.extend(['--prob-threshold',
                             str(self.big_threshold_spin.value())])
            else:
                args.extend(['--num-classes', '31'])
        elif self.is_changedino_config(config):
            checkpoint = self.big_checkpoint_row.text()
            checkpoint_dir = os.path.dirname(checkpoint) if checkpoint else self.big_output_row.text()
            args = [
                os.path.join(root, 'tools', 'test.py'),
                config,
                checkpoint or '__auto__',
                '--work-dir',
                self.big_output_row.text(),
                '--show-dir',
                self.big_output_row.text(),
                '--cfg-options',
                f'changedino.infer.tif1={self.big_tif_a_row.text()}',
                f'changedino.infer.tif2={self.big_tif_b_row.text()}',
                f'changedino.infer.checkpoint_dir={checkpoint_dir}',
                f'changedino.infer.out_base_dir={self.big_output_row.text()}',
                f'changedino.infer.tile_size={self.big_tile_spin.value()}',
                f'changedino.infer.overlap={self.big_overlap_spin.value()}',
                f'changedino.infer.batch_size={self.big_batch_spin.value()}',
                f'changedino.infer.prob_threshold={self.big_threshold_spin.value()}',
                f'changedino.infer.logit_threshold={self.big_threshold_spin.value()}',
                f'changedino.infer.remove_holes_area={self.big_hole_spin.value()}',
                f'changedino.infer.remove_objects_min_size={self.big_object_spin.value()}',
                'changedino.infer.gpu_ids=0',
            ]
            if checkpoint:
                args.append(f'changedino.infer.weights=["{checkpoint}"]')
        else:
            args = [
                os.path.join(root, 'tools', 'big_tif_infer.py'),
                '--config',
                config,
                '--checkpoint',
                self.big_checkpoint_row.text(),
                '--tif-a',
                self.big_tif_a_row.text(),
                '--tif-b',
                self.big_tif_b_row.text(),
                '--out-dir',
                self.big_output_row.text(),
                '--tile-size',
                str(self.big_tile_spin.value()),
                '--overlap',
                str(self.big_overlap_spin.value()),
                '--batch-size',
                str(self.big_batch_spin.value()),
                '--threshold',
                str(self.big_threshold_spin.value()),
                '--remove-holes-area',
                str(self.big_hole_spin.value()),
                '--remove-objects-min-size',
                str(self.big_object_spin.value()),
            ]
            if self.big_fill_holes_check.isChecked():
                args.append('--fill-all-holes')
            if self.big_save_shp_check.isChecked():
                args.append('--save-shp')

        self.current_task = 'big_tif'
        self.active_log = self.big_log
        self.big_progress.setRange(0, 100)
        self.big_progress.setValue(0)
        self.start_process(python, args, root)

    def start_change3d(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, '任务正在运行',
                                '已有任务在运行，请先停止当前任务。')
            return

        mode = self.c3d_mode_combo.currentText()
        python, root = self.command_base()
        change3d_root = self.c3d_root_row.text()
        if not os.path.isdir(change3d_root):
            QMessageBox.warning(self, 'Change3D',
                                'Change3D 源码目录不存在。')
            return

        is_mcd = mode.startswith('MCD')
        config = os.path.join(
            root, 'configs', 'change3d',
            'change3d_mcd.py' if is_mcd else 'change3d_bcd.py')
        if 'train' in mode:
            dataset_root = self.c3d_dataset_row.text()
            if not self.validate_train_dataset(dataset_root):
                return
            args = [os.path.join(root, 'tools', 'train.py'), config]
            args.extend([
                '--work-dir', self.c3d_workdir_row.text(),
                '--cfg-options',
                f'change3d.change3d_root={change3d_root}',
                f'change3d.file_root={dataset_root}',
                f'change3d.pretrained={self.c3d_pretrained_row.text()}',
                'change3d.max_steps=-1',
                f'change3d.max_epochs={self.c3d_max_steps_spin.value()}',
                f'change3d.batch_size={self.c3d_batch_spin.value()}',
                f'change3d.num_workers={self.c3d_workers_spin.value()}',
                'change3d.gpu_id=0',
                'change3d.devices=1',
                'change3d.strategy=auto',
            ])
            if is_mcd:
                args.append(
                    f'change3d.num_class={self.c3d_num_classes_spin.value()}')
            if self.c3d_resume_row.text():
                args.append(f'change3d.resume={self.c3d_resume_row.text()}')
            self.c3d_progress.setMaximum(self.c3d_max_steps_spin.value())
        else:
            if self.c3d_overlap_spin.value() >= self.c3d_tile_spin.value():
                QMessageBox.warning(self, 'Change3D',
                                    'Overlap 必须小于 Tile Size。')
                return
            args = [
                os.path.join(root, 'tools', 'test.py'), config,
                self.c3d_pt_row.text()
            ]
            args.extend([
                '--work-dir', self.c3d_outdir_row.text(),
                '--cfg-options',
                f'change3d.change3d_root={change3d_root}',
                f'change3d.infer.tif1={self.c3d_tif1_row.text()}',
                f'change3d.infer.tif2={self.c3d_tif2_row.text()}',
                f'change3d.infer.tile_size={self.c3d_tile_spin.value()}',
                f'change3d.infer.overlap={self.c3d_overlap_spin.value()}',
                f'change3d.infer.batch_size={self.c3d_infer_batch_spin.value()}',
                'change3d.infer.gpu_id=0',
            ])
            if not is_mcd:
                args.append(
                    f'change3d.infer.prob_threshold={self.c3d_threshold_spin.value()}')
            else:
                args.append(
                    f'change3d.infer.num_classes={self.c3d_num_classes_spin.value()}')
            self.c3d_progress.setRange(0, 100)

        self.current_task = 'change3d'
        self.active_log = self.c3d_log
        self.c3d_progress.setValue(0)
        self.start_process(python, args, root)

    def start_process(self, program, args, workdir):
        self.process = QProcess(self)
        self.process.setWorkingDirectory(workdir)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUTF8', '1')
        env.insert('PYTHONIOENCODING', 'utf-8')
        env.insert('NO_COLOR', '1')
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        run_args = ['-u'] + args if os.path.basename(program).lower().startswith(
            'python') else args
        self.append_log('$ ' + ' '.join([program] + run_args))
        self.process.start(program, run_args)
        if not self.process.waitForStarted(3000):
            self.append_log('任务启动失败，请检查 Python 路径或参数。')
        else:
            self.process_pid = int(self.process.processId())
            self.append_log(f'任务已启动，PID: {self.process_pid}')

    def read_process_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode(
            errors='replace')
        self.append_log(text.rstrip())
        self.update_progress_from_log(text)

    def update_progress_from_log(self, text):
        train_match = re.findall(r'Iter\(train\)\s+\[\s*(\d+)/(\d+)\]',
                                 text)
        if train_match:
            current, total = train_match[-1]
            self.progress.setMaximum(int(total))
            self.progress.setValue(int(current))

        test_match = re.findall(r'Iter\(test\)\s+\[\s*(\d+)/(\d+)\]',
                                text)
        if test_match:
            current, total = test_match[-1]
            self.pred_progress.setMaximum(int(total))
            self.pred_progress.setValue(int(current))

        big_match = re.findall(r'\[PROGRESS\]\s+(\d+)/(\d+)\s+tiles',
                               text)
        if big_match:
            current, total = big_match[-1]
            self.big_progress.setMaximum(int(total))
            self.big_progress.setValue(int(current))
            if hasattr(self, 'c3d_progress') and self.current_task == 'change3d':
                self.c3d_progress.setMaximum(int(total))
                self.c3d_progress.setValue(int(current))

        c3d_train_match = re.findall(
            r'\[PROGRESS\]\s+epoch=(\d+)\s+step=(\d+)/(\d+)\s+loss=',
            text)
        if c3d_train_match and self.current_task in ('change3d',
                                                     'change3d_train'):
            _epoch, current, total = c3d_train_match[-1]
            current = int(current)
            total = int(total)
            if self.current_task == 'change3d_train':
                max_epoch = max(1, self.max_iters)
                self.progress.setMaximum(max_epoch)
                self.progress.setValue(min(int(_epoch) + 1, max_epoch))
                self.progress.setFormat(
                    f'Epoch {int(_epoch) + 1} / {max_epoch}   step {current} / {total}')
            else:
                self.c3d_progress.setMaximum(total)
                self.c3d_progress.setValue(current)

    def process_finished(self, exit_code, _status):
        if self.current_task == 'test' and exit_code == 0:
            self.pred_progress.setValue(self.pred_progress.maximum())
        if self.current_task == 'big_tif' and exit_code == 0:
            self.big_progress.setValue(self.big_progress.maximum())
        if self.current_task == 'change3d' and exit_code == 0:
            self.c3d_progress.setValue(self.c3d_progress.maximum())
        if self.current_task == 'change3d_train' and exit_code == 0:
            self.progress.setValue(self.progress.maximum())
        self.append_log(f'任务结束，退出码: {exit_code}')
        self.current_task = None
        self.process_pid = None

    def stop_process(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            pid = self.process_pid
            if pid and os.name == 'nt':
                QProcess.execute('taskkill', ['/PID', str(pid), '/T', '/F'])
            else:
                if pid:
                    os.kill(pid, signal.SIGTERM)
                self.process.terminate()
                if not self.process.waitForFinished(3000):
                    self.process.kill()
            self.append_log('任务已停止')

    def clear_log(self):
        if self.active_log is not None:
            self.active_log.clear()

    def append_log(self, text):
        if not text:
            return
        log = self.active_log or self.train_log
        fmt = QTextCharFormat()
        if 'ERROR' in text or 'Traceback' in text:
            fmt.setForeground(QColor('#c0392b'))
        elif 'WARNING' in text:
            fmt.setForeground(QColor('#c77d00'))
        else:
            fmt.setForeground(QColor('#6b6b6b'))
        cursor = log.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text + '\n', fmt)
        if self.auto_scroll_check.isChecked():
            log.setTextCursor(cursor)
            log.ensureCursorVisible()

    def apply_style(self):
        self.setStyleSheet("""
            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 14px;
                color: #4b4b4b;
            }
            #Header {
                background: #247b38;
                min-height: 42px;
                max-height: 42px;
                border-radius: 4px;
            }
            #HeaderTitle {
                color: white;
                font-size: 18px;
                font-weight: 700;
            }
            #HeaderUser {
                color: #d8efdc;
            }
            QTabWidget::pane {
                border: 1px solid #cfcfcf;
                top: -1px;
            }
            QTabBar::tab {
                min-width: 120px;
                padding: 10px 18px;
                background: #efefef;
                border: 1px solid #d1d1d1;
            }
            QTabBar::tab:selected {
                color: #207a3a;
                background: white;
                border-bottom: 3px solid #2a8a48;
                font-weight: 700;
            }
            QGroupBox {
                border: 1px solid #d6d6d6;
                border-radius: 4px;
                margin-top: 18px;
                padding-top: 18px;
                font-weight: 700;
                color: #258348;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                background: white;
            }
            QLineEdit, QComboBox, QSpinBox {
                min-height: 32px;
                border: 1px solid #d6d6d6;
                border-radius: 3px;
                padding: 3px 8px;
                background: white;
            }
            QCheckBox {
                min-height: 32px;
                spacing: 6px;
            }
            QPushButton {
                min-width: 86px;
                min-height: 30px;
                border: 1px solid #cfcfcf;
                border-radius: 3px;
                background: #eeeeee;
            }
            #StartButton {
                color: white;
                background: #6fbf8c;
                border-color: #6fbf8c;
                font-weight: 700;
            }
            #StopButton {
                color: white;
                background: #c53b2c;
                border-color: #c53b2c;
                font-weight: 700;
            }
            QProgressBar {
                min-height: 24px;
                border: 1px solid #d7d7d7;
                text-align: center;
                background: #efefef;
            }
            QProgressBar::chunk {
                background: #2a9d55;
            }
            #LogText {
                background: white;
                border: 1px solid #d7d7d7;
                font-family: Consolas, "Courier New", monospace;
                font-size: 13px;
            }
        """)


def main():
    app = QApplication(sys.argv)
    window = OpenCDGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
