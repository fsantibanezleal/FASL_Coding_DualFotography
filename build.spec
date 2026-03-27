# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for FASL_DualFotography.

Build with:
    pyinstaller build.spec --clean --noconfirm

Or use the PowerShell script:
    .\\Build_PyInstaller.ps1
"""
import pkgutil
from pathlib import Path

APP_NAME = "FASL_DualFotography"
ENTRY_POINT = "run_app.py"

# Hidden imports: dynamically discover all dash submodules
hidden_imports = []
try:
    import dash
    for importer, modname, ispkg in pkgutil.walk_packages(
        dash.__path__, prefix="dash."
    ):
        hidden_imports.append(modname)
except ImportError:
    pass

# Also add dash-bootstrap-components
hidden_imports.extend([
    "dash_bootstrap_components",
    "plotly",
    "plotly.graph_objects",
    "plotly.express",
])

# Data files: Dash assets and SVG documentation
datas = []
assets_dir = Path("src/frontend/assets")
if assets_dir.exists():
    datas.append((str(assets_dir), "src/frontend/assets"))

svg_dir = Path("docs/svg")
if svg_dir.exists():
    datas.append((str(svg_dir), "docs/svg"))

a = Analysis(
    [ENTRY_POINT],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
