package pinak.policy

default allow = true

# Example strict rule from enterprise reference
allow {
  input.claims.role == "editor"
  lower(input.headers["x-pinak-project"]) == lower(input.claims.pid)
  input.request.path == "/api/v1/memory/episodic/add"
}

