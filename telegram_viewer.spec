# PyInstaller spec for Telegram Viewer .exe (RU → EN)
# Run: pyinstaller telegram_viewer.spec
# Output: dist/TelegramViewer.exe

a = Analysis(
    ['run_telegram_viewer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config_telegram.yml', '.'),
    ],
    hiddenimports=[
        'scraper',
        'telegram_utils',
        'translate_text',
        'yaml',
        'bs4',
        'lxml',
        'requests',
        'playwright',
        'deep_translator',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['streamlit'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TelegramViewer',
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
