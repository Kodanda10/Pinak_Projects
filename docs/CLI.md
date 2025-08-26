# Pinak CLI (One-Click)

Use the top-level `pinak` CLI for one-click setup and health checks.

Examples:

```
# One-click: Bridge init + services up + health
pinak quickstart --name "MyApp" --url http://localhost:8011 --tenant default

# Security baseline + environment check
pinak doctor

# Token helper (mints dev JWT and sets via Bridge)
pinak token --set

# Passthrough to sub-CLIs
pinak bridge status --json
pinak memory health
```
