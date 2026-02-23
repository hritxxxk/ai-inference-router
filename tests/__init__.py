"""Pytest package settings."""

import os


# Direct telemetry writes to a throwaway sqlite DB for tests
os.environ.setdefault("ROUTER_TELEMETRY_DB_PATH", "tests/.tmp/router-test.db")
