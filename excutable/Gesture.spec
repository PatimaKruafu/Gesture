# -*- mode: python ; coding: utf-8 -*-
# Gesture.spec (or main.spec)

block_cipher = None

a = Analysis(
    ['main.py'],  # Your main Python script
    pathex=[], # Or adjust to your project root if spec is there
    binaries=[],
    datas=[('tasks', 'tasks')],  # Ensure 'tasks' folder is relative to this spec file location
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries, # Usually empty for macOS .app when BUNDLE is used
    a.zipfiles,
    a.datas,    # Make sure datas is passed here too
    [],
    name='Gesture', # Name of the executable inside Contents/MacOS/
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,   # UPX might be ignored or problematic on arm64 macOS, False is safer
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # *** IMPORTANT: Set to False for a windowed app ***
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64', # Correct for your M1 Mac
    codesign_identity=None, # For ad-hoc signing, this is fine
    entitlements_file=None,
)

# --- CORRECTED BUNDLE SECTION ---
app = BUNDLE(
    exe,
    name='Gesture.app',        # The name of your .app bundle folder
    icon=None,                # Optional: path to your 'icon.icns' file
    bundle_identifier='com.stillalive.gestureapp', # <<< CORRECTED: REPLACE WITH YOUR UNIQUE ID
                                                 # Example: com.stillalive.gesturecontrol
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSUIElement': '0',      # '0' for normal app with Dock icon, '1' for agent app
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'Gesture', # Often same as the EXE name or .app name without suffix
        'CFBundleDisplayName': 'Gesture Controller', # User-friendly name in Finder/Launchpad
        'CFBundleVersion': '0.1.0',           # Your app's internal version
        'CFBundleShortVersionString': '0.1',  # Version string shown to user
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',        # Often '????' for non-App Store apps

        # --- Essential Permission Keys ---
        'NSCameraUsageDescription': 'This application requires camera access to detect hand gestures for controlling the user interface and ROI.',
        'NSAccessibilityUsageDescription': 'This application requires Accessibility permissions to control the mouse cursor and perform clicks based on your hand gestures.'
    }
)