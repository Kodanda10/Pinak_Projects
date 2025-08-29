from setuptools import setup

APP = ['main.py']
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'Pinak Bridge',
        'CFBundleDisplayName': 'Pinak Bridge',
        'CFBundleIdentifier': 'ai.pinak.bridge',
        'LSMinimumSystemVersion': '12.0',
        'LSUIElement': True,
        'CFBundleIconFile': 'pinak-sync',
    },
    'packages': ['pinak'],
}

setup(
    app=APP,
    options={'py2app': OPTIONS},
    data_files=['../../pinak-sync.png'],
    setup_requires=['py2app'],
)
