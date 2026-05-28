#!/usr/bin/env python3
"""
Ensure CVAT E2E user exists with staff/superuser privileges.

Reads CVAT_E2E_USER and CVAT_E2E_PASSWORD from .env file.
Executes Django management commands via docker compose exec.

Usage:
    python3 scripts/e2e/sam2/ensure_user.py [--run-dir DIR]
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def load_env(env_path: str = ".env") -> dict:
    """Parse a simple .env file (KEY=VALUE lines, no quoting)."""
    env = {}
    p = Path(env_path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def docker_exec_python(script: str, compose_project: str = "cvat") -> str:
    """Run a Python script inside cvat_server container."""
    cmd = [
        "docker", "compose", "-p", compose_project,
        "-f", "docker-compose.yml",
        "exec", "-T", "cvat_server",
        "python3", "-c", script,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"docker exec failed (rc={result.returncode})")
    return result.stdout.strip()


def ensure_user(username: str, password: str, compose_project: str = "cvat") -> dict:
    """Create or update CVAT user with staff/superuser privileges."""
    # Django script to create/update user
    django_script = f"""
import json
from django.contrib.auth.models import User

username = {username!r}
password = {password!r}

try:
    user = User.objects.get(username=username)
    user.set_password(password)
    user.is_staff = True
    user.is_superuser = True
    user.save()
    created = False
except User.DoesNotExist:
    user = User.objects.create_superuser(
        username=username,
        password=password,
        email=f"{{username}}@e2e.local",
    )
    user.is_staff = True
    user.is_superuser = True
    user.save()
    created = True

print(json.dumps({{
    "username": user.username,
    "is_staff": user.is_staff,
    "is_superuser": user.is_superuser,
    "is_active": user.is_active,
    "created": created,
}}))
"""
    # Wrap in manage.py shell context
    wrapper = f"""
import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cvat.settings.production')
django.setup()
{django_script}
"""
    output = docker_exec_python(wrapper, compose_project)
    # Find the JSON line
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from Django script. Output: {output}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensure CVAT E2E user")
    parser.add_argument("--run-dir", default=None, help="Directory to save user info JSON")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    args = parser.parse_args()

    env = load_env(args.env_file)
    username = env.get("CVAT_E2E_USER")
    password = env.get("CVAT_E2E_PASSWORD")

    if not username or not password:
        print("ERROR: CVAT_E2E_USER and CVAT_E2E_PASSWORD must be set in .env", file=sys.stderr)
        sys.exit(1)

    print(f"Ensuring user: {username}")
    result = ensure_user(username, password)
    print(json.dumps(result, indent=2))

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "e2e_user.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved to: {out_path}")

    if not result.get("is_staff") or not result.get("is_superuser"):
        print("WARNING: User is missing staff/superuser privileges!", file=sys.stderr)
        sys.exit(1)

    print("OK: User is ready for E2E testing")


if __name__ == "__main__":
    main()
