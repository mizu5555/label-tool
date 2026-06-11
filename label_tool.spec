# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

hiddenimports = ['PyQt5.sip']
excludes = [
    'IPython',
    'PIL',
    'PySide2',
    'PySide6',
    'fsspec',
    'lxml',
    'matplotlib',
    'notebook',
    'openpyxl',
    'pyarrow',
    'pytest',
    'scipy',
    'sklearn',
    'sqlalchemy',
    'tensorflow',
    'torch',
]

a = Analysis(
    ['label_tool_v6.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='label_tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
