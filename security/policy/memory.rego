package memory

default allow = false

allow {
  input.user.sub == "analyst"
  not input.item.pii_sensitive
  input.item.scope == "project"
}

