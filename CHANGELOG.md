## 1.1.1 (2025-08-28)

- Fix: Correct JWT `exp` calculation using timezone-aware UTC to prevent immediate expiry (401 issues resolved).
- Bridge: `pinak-bridge verify` now reports token expiry and seconds to expiry.
- UX: Memory client prints guidance on 401 to rotate token.
- New: macOS menu bar app `pinak-menubar` with status + self-healing.

