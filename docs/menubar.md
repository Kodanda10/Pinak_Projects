Pinak macOS Menu Bar

- Command: `pinak-menubar`
- Platform: macOS

Features
- Status: Shows whether Memory API is up and token is valid.
- Self-heal: Attempts to rotate token and start services if unhealthy.
- Controls: Start/Heal, Rotate Token, Stop Services, Quit.

Icon
- Uses `/Users/abhijita/Pinak_Package/pinak-sync.png` as the app icon.

Security Notes
- No secrets are logged. Token rotation uses project context and env `SECRET_KEY`.
- Honors CI security gates; no changes to service auth.

Troubleshooting
- If the app fails to start, ensure dependencies are installed and run: `pip install -e .` in the repo venv.
