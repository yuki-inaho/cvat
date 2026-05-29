#!/usr/bin/env bash
# Collect CVAT / Nuclio / SAM2 diagnostic logs.
#
# Usage:
#   scripts/e2e/sam2/collect_logs.sh [RUN_DIR]
#
# If RUN_DIR is not specified, creates temp/e2e_sam2/run_<timestamp>/
#
# Collects:
#   - docker ps (container states + ports)
#   - cvat_server logs (last 300 lines)
#   - nuclio dashboard logs (last 300 lines)
#   - SAM2 function container logs (last 300 lines)
#   - just status output
#   - just sam2-ps output
#   - just fn-list output
#   - SAM2 function port mapping
#   - Nuclio function health check (via nuctl)
#   - CVAT health API
#
# All output is saved to RUN_DIR for post-mortem analysis of 500 errors.
set -euo pipefail

RUN_DIR="${1:-temp/e2e_sam2/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

echo "=== Collecting diagnostic logs ==="
echo "Run dir: $RUN_DIR"
echo ""

# Container state
echo "  Collecting docker ps..."
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' > "$RUN_DIR/docker_ps.txt" 2>&1 || true

# CVAT server logs
echo "  Collecting cvat_server logs..."
docker logs --tail 300 cvat_server > "$RUN_DIR/cvat_server.log" 2>&1 || echo "WARN: cvat_server logs not available" > "$RUN_DIR/cvat_server.log"

# Nuclio dashboard logs
echo "  Collecting nuclio dashboard logs..."
docker logs --tail 300 nuclio > "$RUN_DIR/nuclio.log" 2>&1 || echo "WARN: nuclio logs not available" > "$RUN_DIR/nuclio.log"

# SAM2 function container logs
echo "  Collecting SAM2 function logs..."
docker logs --tail 300 nuclio-nuclio-ort-facebookresearch-sam2-hiera-large > "$RUN_DIR/sam2_function.log" 2>&1 || echo "WARN: SAM2 function logs not available" > "$RUN_DIR/sam2_function.log"

# CVAT worker annotation logs (relevant for serverless invoke path)
echo "  Collecting cvat_worker_annotation logs..."
docker logs --tail 300 cvat_worker_annotation > "$RUN_DIR/cvat_worker_annotation.log" 2>&1 || echo "WARN: cvat_worker_annotation logs not available" > "$RUN_DIR/cvat_worker_annotation.log"

# just status
echo "  Collecting just status..."
just status > "$RUN_DIR/just_status.txt" 2>&1 || echo "WARN: just status failed" > "$RUN_DIR/just_status.txt"

# just sam2-ps
echo "  Collecting just sam2-ps..."
just sam2-ps > "$RUN_DIR/just_sam2_ps.txt" 2>&1 || echo "WARN: just sam2-ps failed" > "$RUN_DIR/just_sam2_ps.txt"

# just fn-list
echo "  Collecting just fn-list..."
just fn-list > "$RUN_DIR/just_fn_list.txt" 2>&1 || echo "WARN: just fn-list failed" > "$RUN_DIR/just_fn_list.txt"

# SAM2 port mapping
echo "  Collecting SAM2 port..."
docker port nuclio-nuclio-ort-facebookresearch-sam2-hiera-large 8080/tcp > "$RUN_DIR/sam2_port.txt" 2>&1 || echo "WARN: SAM2 port not available" > "$RUN_DIR/sam2_port.txt"

# Nuclio function health via nuctl
echo "  Collecting nuctl function details..."
nuctl get function ort-facebookresearch-sam2-hiera-large --platform local --namespace nuclio > "$RUN_DIR/nuctl_function.txt" 2>&1 || echo "WARN: nuctl get failed" > "$RUN_DIR/nuctl_function.txt"

# CVAT health API
echo "  Collecting CVAT health..."
curl -fsS "http://localhost:8080/api/server/health/?format=json&org=" > "$RUN_DIR/cvat_health.json" 2>&1 || echo '{"error": "health check failed"}' > "$RUN_DIR/cvat_health.json"

# Docker network info for nuclio
echo "  Collecting Docker network info..."
docker network inspect cvat_cvat > "$RUN_DIR/docker_network_cvat.json" 2>&1 || echo "WARN: network inspect failed" > "$RUN_DIR/docker_network_cvat.json"

# Search for 500 errors in cvat_server logs
echo "  Searching for 500 errors..."
grep -i "500\|error\|exception\|traceback" "$RUN_DIR/cvat_server.log" | tail -50 > "$RUN_DIR/cvat_server_errors.txt" 2>&1 || echo "No 500/error patterns found" > "$RUN_DIR/cvat_server_errors.txt"

# Search for errors in SAM2 function logs
grep -i "error\|exception\|traceback\|fail" "$RUN_DIR/sam2_function.log" | tail -50 > "$RUN_DIR/sam2_function_errors.txt" 2>&1 || echo "No error patterns found" > "$RUN_DIR/sam2_function_errors.txt"

# Search for errors in nuclio dashboard logs
grep -i "error\|fail\|500" "$RUN_DIR/nuclio.log" | tail -50 > "$RUN_DIR/nuclio_errors.txt" 2>&1 || echo "No error patterns found" > "$RUN_DIR/nuclio_errors.txt"

# Timestamp
date "+%Y-%m-%d %H:%M:%S %Z%z" > "$RUN_DIR/collected_at.txt"

echo ""
echo "=== Collection complete ==="
echo "Files:"
ls -lh "$RUN_DIR/"
echo ""
echo "To analyze 500 errors, check:"
echo "  $RUN_DIR/cvat_server_errors.txt"
echo "  $RUN_DIR/sam2_function_errors.txt"
echo "  $RUN_DIR/nuclio_errors.txt"
echo ""
echo "Run dir: $RUN_DIR"
