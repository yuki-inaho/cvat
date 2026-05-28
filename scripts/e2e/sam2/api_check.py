#!/usr/bin/env python3
"""
Check E2E task/job API endpoints for CVAT SAM2 testing.

Verifies:
  - job 180 / task 181 (512x512 square image)
  - job 181 / task 182 (640x360 non-square image)

Saves results to run artifact directory.

Usage:
    python3 scripts/e2e/sam2/api_check.py [--run-dir DIR]
"""
import json
import http.cookiejar
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


def load_env(env_path: str = ".env") -> dict:
    """Parse a simple .env file."""
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


class CvatApiClient:
    """Minimal CVAT API client using only stdlib."""

    def __init__(self, host: str, username: str, password: str):
        self.host = host.rstrip("/")
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cj)
        )
        self._login(username, password)

    def _login(self, username: str, password: str):
        """Login and store session cookies."""
        body = json.dumps({"username": username, "password": password}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/auth/login",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = self.opener.open(req, timeout=15)
        resp.read()  # consume response
        # Extract CSRF token for subsequent requests
        self.csrf_token = None
        for cookie in self.cj:
            if cookie.name == "csrftoken":
                self.csrf_token = cookie.value

    def get(self, path: str) -> dict:
        """GET an API endpoint, return parsed JSON."""
        url = f"{self.host}{path}"
        req = urllib.request.Request(url, method="GET")
        if self.csrf_token:
            req.add_header("X-CSRFToken", self.csrf_token)
        resp = self.opener.open(req, timeout=15)
        return {
            "status": resp.status,
            "url": url,
            "data": json.loads(resp.read().decode()),
        }

    def check_endpoint(self, path: str) -> dict:
        """Check an endpoint and return status + summary."""
        try:
            result = self.get(path)
            return {
                "path": path,
                "status": result["status"],
                "ok": True,
                "data": result["data"],
            }
        except urllib.error.HTTPError as e:
            return {
                "path": path,
                "status": e.code,
                "ok": False,
                "error": str(e),
            }
        except Exception as e:
            return {
                "path": path,
                "status": None,
                "ok": False,
                "error": str(e),
            }


# Define check targets
CHECKS = [
    # Job 180 (task 181, 512x512)
    {"name": "job_180_detail", "path": "/api/jobs/180"},
    {"name": "job_180_meta", "path": "/api/jobs/180/data/meta"},
    {"name": "job_180_annotations", "path": "/api/jobs/180/annotations", "optional": True},
    {"name": "task_181_detail", "path": "/api/tasks/181"},
    {"name": "task_181_meta", "path": "/api/tasks/181/data/meta"},
    # Job 181 (task 182, 640x360)
    {"name": "job_181_detail", "path": "/api/jobs/181"},
    {"name": "job_181_meta", "path": "/api/jobs/181/data/meta"},
    {"name": "job_181_annotations", "path": "/api/jobs/181/annotations", "optional": True},
    {"name": "task_182_detail", "path": "/api/tasks/182"},
    {"name": "task_182_meta", "path": "/api/tasks/182/data/meta"},
]


def extract_image_size(meta_data: dict) -> dict:
    """Extract image dimensions from task/job meta response."""
    frames = meta_data.get("frames", [])
    if frames:
        frame = frames[0]
        return {"width": frame.get("width"), "height": frame.get("height")}
    return {"width": None, "height": None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check E2E task/job API endpoints")
    parser.add_argument("--run-dir", default=None, help="Directory to save results")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    args = parser.parse_args()

    env = load_env(args.env_file)
    host = env.get("CVAT_E2E_HOST", "http://localhost:8080")
    username = env.get("CVAT_E2E_USER")
    password = env.get("CVAT_E2E_PASSWORD")

    if not username or not password:
        print("ERROR: CVAT_E2E_USER and CVAT_E2E_PASSWORD must be set in .env", file=sys.stderr)
        sys.exit(1)

    # Create run dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(f"temp/e2e_sam2/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"API check against {host}")
    print(f"Results will be saved to: {run_dir}")

    client = CvatApiClient(host, username, password)

    results = {}
    all_ok = True
    for check in CHECKS:
        name = check["name"]
        path = check["path"]
        optional = check.get("optional", False)
        print(f"  Checking {name} ... ", end="", flush=True)
        result = client.check_endpoint(path)
        results[name] = result
        if result["ok"]:
            print(f"HTTP {result['status']} OK")
            # Save individual result
            out_path = run_dir / f"{name}.json"
            out_path.write_text(json.dumps(result["data"], indent=2) + "\n")
        else:
            suffix = " (optional, ignored)" if optional else ""
            print(f"FAILED: {result.get('error', 'unknown')}{suffix}")
            if not optional:
                all_ok = False

    # Extract and verify image sizes
    print("\n--- Image size verification ---")
    expected_sizes = {
        "job_180_meta": {"width": 512, "height": 512},
        "job_181_meta": {"width": 640, "height": 360},
    }
    for meta_name, expected in expected_sizes.items():
        if meta_name in results and results[meta_name]["ok"]:
            actual = extract_image_size(results[meta_name]["data"])
            match = actual["width"] == expected["width"] and actual["height"] == expected["height"]
            status = "OK" if match else "MISMATCH"
            print(f"  {meta_name}: {actual['width']}x{actual['height']} (expected {expected['width']}x{expected['height']}) [{status}]")
            if not match:
                all_ok = False
        else:
            print(f"  {meta_name}: SKIPPED (endpoint failed)")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "host": host,
        "all_ok": all_ok,
        "checks": {name: {"ok": r["ok"], "status": r["status"]} for name, r in results.items()},
    }
    summary_path = run_dir / "api_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary saved to: {summary_path}")

    if all_ok:
        print("OK: All API checks passed")
    else:
        print("FAILED: Some API checks failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
