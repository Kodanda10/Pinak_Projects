
from typing import Optional

try:
except Exception:
    rumps = None  # type: ignore


APP_NAME = "Pinak"
# Use provided icon for macOS menu bar
DEFAULT_ICON_PATH = "/Users/abhijita/Pinak_Package/pinak-sync.png"
ICON_OK = DEFAULT_ICON_PATH if os.path.exists(DEFAULT_ICON_PATH) else None
ICON_WARN = ICON_OK


def check_health() -> dict:
    try:

        mm = MemoryManager()
        mem_ok = mm.health()
    except Exception:
        mem_ok = False
    token_expired = None
    token_expires_in = None
    try:


        ctx = ProjectContext.find()
        tok = ctx.get_token() if ctx else None
        if tok:
            try:
                claims = jwt.get_unverified_claims(tok)  # type: ignore[attr-defined]
                exp = int(claims.get("exp")) if claims.get("exp") is not None else None
            except Exception:
                exp = None
            now = int(time.time())
            if exp is not None:
                token_expired = exp <= now
                token_expires_in = max(0, exp - now)
    except Exception:
        pass
    return {
        "memory_api_ok": bool(mem_ok),
        "token_expired": token_expired,
        "token_expires_in": token_expires_in,
    }


def try_self_heal(log_cb=None) -> dict:
    def log(msg: str):
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    info = check_health()
    changed = False
    if info.get("token_expired") is True:
        try:

            ctx = ProjectContext.find()
            if ctx:
                secret = os.getenv("SECRET_KEY", "change-me-in-prod")
                ctx.rotate_token(minutes=240, secret=secret)
                log("Token rotated (4h)")
                changed = True
        except Exception:
            log("Token rotation failed")
    if not info.get("memory_api_ok"):
        try:

            rc = cmd_up(type("NS", (), {})())
            log(f"pinak up: rc={rc}")
            changed = True
        except Exception:
            try:
                subprocess.call(["pinak", "up"])  # last resort
            except Exception:
                log("Failed to start services")
    new_info = check_health()
    new_info["changed"] = changed
    return new_info


class PinakStatusApp(rumps.App):
    def __init__(self):
        super().__init__(APP_NAME, icon=ICON_OK, template=True)
        self.menu = [
            rumps.MenuItem("Status", callback=self.on_status),
            rumps.MenuItem("Start/Heal", callback=self.on_heal),
            rumps.MenuItem("Rotate Token", callback=self.on_rotate),
            rumps.MenuItem("Stop Services", callback=self.on_stop),
            None,
            rumps.MenuItem("Quit", callback=self.on_quit),
        ]
        self._timer = rumps.Timer(self._tick, 60)
        self._timer.start()
        threading.Thread(target=self._initial_probe, daemon=True).start()

    def _initial_probe(self):
        self._set_title("Checking…")
        self.update_status()

    def _set_title(self, text: str):
        self.title = f"{APP_NAME}: {text}"

    def notify(self, title: str, msg: str):
        try:
            rumps.notification(title, "", msg)
        except Exception:
            pass

    def on_quit(self, _):
        rumps.quit_application()

    def on_status(self, _):
        info = check_health()
        self._render_status(info)

    def on_heal(self, _):
        info = try_self_heal(lambda m: None)
        self._render_status(info)
        if info.get("changed"):
            self.notify("Pinak", "Self-heal attempted. Status updated.")

    def on_rotate(self, _):
        try:

            ctx = ProjectContext.find()
            if ctx:
                secret = os.getenv("SECRET_KEY", "change-me-in-prod")
                ctx.rotate_token(minutes=240, secret=secret)
                self.notify("Pinak", "Token rotated (4h)")
        except Exception:
            self.notify("Pinak", "Token rotation failed")
        self.update_status()

    def on_stop(self, _):
        try:

            cmd_down(type("NS", (), {})())
        except Exception:
            try:
                subprocess.call(["pinak", "down"])  # last resort
            except Exception:
                pass
        self.update_status()

    def _tick(self, _):
        self.update_status()

    def update_status(self):
        info = check_health()
        self._render_status(info)

    def _render_status(self, info: dict):
        mem_ok = info.get("memory_api_ok")
        texp = info.get("token_expired")
        if mem_ok and (texp is False or texp is None):
            self._set_title("Up ✔")
        elif not mem_ok:
            self._set_title("Down ✖")
        elif texp is True:
            self._set_title("Token Expired ⚠")
        else:
            self._set_title("Degraded ⚠")


def main():
    if sys.platform != "darwin":
        print("pinak-menubar is supported on macOS only.")
        return 1
    if rumps is None:
        print("Missing dependency: rumps. Please reinstall Pinak or pip install rumps.")
        return 2
    app = PinakStatusApp()
    app.run()
    return 0
