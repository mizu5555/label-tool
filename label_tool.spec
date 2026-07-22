# -*- mode: python ; coding: utf-8 -*-

import os
import sys

block_cipher = None
debug_build = os.environ.get('LABEL_TOOL_DEBUG') == '1'
release_macos_app = sys.platform == 'darwin' and not debug_build

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

if release_macos_app:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='label_tool',
        debug=debug_build,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=debug_build,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='label_tool',
    )
    app = BUNDLE(
        coll,
        name='Label Tool.app',
        icon=None,
        bundle_identifier='com.mizu.labeltool',
        info_plist={
            'CFBundleName': 'Label Tool',
            'CFBundleDisplayName': 'Label Tool',
            'CFBundleIdentifier': 'com.mizu.labeltool',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHighResolutionCapable': True,
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='label_tool',
        debug=debug_build,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=debug_build,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
