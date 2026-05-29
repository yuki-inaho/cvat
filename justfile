set shell := ["bash", "-euo", "pipefail", "-c"]

# =============================================================================
# CVAT + SAM2 半自動アノテーション 運用 justfile
# =============================================================================
# このファイルは CVAT 本体・serverless(Nuclio)基盤・SAM2 アノテーション関数
# (PyTorch版 / ONNX Runtime GPU版) ・E2E テストを一括操作するための入口。
#
# レシピ一覧は `just --list`。以下は「何をしたいか」別のワークフロー。
#
# -----------------------------------------------------------------------------
# 0. 全体構成 (3 レイヤー)
# -----------------------------------------------------------------------------
#   [レイヤー1] CVAT 本体          : up / down / status / logs / superuser
#   [レイヤー2] serverless基盤(Nuclio): aa-up / aa-down / aa-ps / fn-list
#   [レイヤー3] モデル関数(Nuclio fn): sam2-up-cpu / sam2-up-gpu / sam2-ort-up-gpu / *-down / *-ps
#   ※ レイヤーは下に行くほど上位に依存する。aa-down するとレイヤー3の関数は消える。
#
# -----------------------------------------------------------------------------
# 1. はじめての起動 (SAM2 を GPU で動かす - 推奨ルート)
# -----------------------------------------------------------------------------
#   just up && just wait           # CVAT 本体を起動し health 200 まで待つ
#   just aa-up                     # serverless/Nuclio 基盤を起動
#   just sam2-ort-up-gpu           # SAM2.1 large encoder を ONNX Runtime GPU でdeploy
#   just status && just sam2-ort-ps  # CVAT と ORT 関数(CUDAExecutionProvider)の確認
#   → ブラウザ http://localhost:8080 で SAM2 interactor が使える
#
# -----------------------------------------------------------------------------
# 2. 再起動 (restart) — 対象別
# -----------------------------------------------------------------------------
#   ● SAM2 ORT 関数だけ作り直す (基盤は維持・最速):
#       just sam2-ort-down && just sam2-ort-up-gpu && just sam2-ort-ps
#   ● CVAT 画面がおかしい (本体だけ):
#       just down && just up && just wait
#   ● 基盤ごと入れ直す (Nuclio関数は消えるので最後に再deploy):
#       just aa-down && just aa-up && just wait && just sam2-ort-up-gpu
#
# -----------------------------------------------------------------------------
# 3. SAM2 バックエンドの選択 (3 種類、関数名で排他/共存)
# -----------------------------------------------------------------------------
#   ● sam2-up-cpu      : PyTorch CPU 版 (pth-...-large)。GPU不要だが遅い。
#   ● sam2-up-gpu      : PyTorch GPU 版 (pth-...-large)。GTX1070(sm_61)では
#                        torch wheel が非対応でクラッシュする既知問題あり (非推奨)。
#   ● sam2-ort-up-gpu  : ONNX Runtime GPU 版 (ort-...-large)。GTX1070 で
#                        CUDAExecutionProvider 動作 (本worktreeの主力)。
#   ※ pth版とort版は metadata.name が別なので同時deploy可。UI plugin が呼ぶ
#     関数は cvat-ui/plugins/sam2/src/ts/index.tsx の modelID で決まる
#     (現状 `ort-facebookresearch-sam2-hiera-large`)。
#
# -----------------------------------------------------------------------------
# 4. E2E テスト (Playwright CLI, headless)
# -----------------------------------------------------------------------------
#   just e2e-user                  # .env の E2E ユーザーを作成/更新
#   just e2e-login                 # headless ログインし auth-state 保存
#   just e2e-sam2-positive-point   # 左クリック positive point
#   just e2e-sam2-negative-point   # 右クリック negative point
#   just e2e-sam2-bbox             # BBox プロンプト
#   just e2e-sam2-non-square       # 640x360 非正方形画像の座標検証
#   just e2e-sam2-all              # 上記を連続実行 (login→job→bbox→pos→neg→non-square)
#   → 成果物は temp/e2e_sam2/run_*/ に screenshot/network/console を保存
#
# -----------------------------------------------------------------------------
# 5. 開発ループ (ORT core のコード変更時)
# -----------------------------------------------------------------------------
#   just sam2-ort-lint             # cd nuclio && uv run ruff + pytest (型/shape契約検証)
#   (handler変更を反映) just sam2-ort-down && just sam2-ort-up-gpu
#   (UI plugin変更を反映) just sam2-ui-up   # cvat_ui を plugin込みで再ビルド
#
# -----------------------------------------------------------------------------
# 6. トラブルシュート
# -----------------------------------------------------------------------------
#   ● Bad Gateway / 502        : just status → just logs cvat_server
#   ● SAM2 が 500 / mask 出ない : just sam2-ort-ps (provider確認) → just sam2-ort-logs
#                                → just e2e-logs (CVAT/Nuclio/関数ログ一括収集)
#   ● ORT関数が ready にならない : just sam2-ort-logs で CUDAExecutionProvider と
#                                model IO を確認。CUDA不在なら fail-fast で起動失敗(設計通り)。
#   ● mask が空/ずれる          : encoder/decoder の variant 不一致を疑う(7節)。
#
# -----------------------------------------------------------------------------
# 7. ONNX Runtime GPU 版の重要な前提 (sam2-ort-*)
# -----------------------------------------------------------------------------
#   ● モデル配置: encoder ONNX (848MB) は git に入れず host dir を volume mount。
#     既定 SAM2_ORT_MODEL_DIR=temp/sam2_models_export/models。このファイルを消すと
#     sam2-ort-up-gpu は exit 2 で停止する (暗黙fallbackなし)。
#   ● variant 一致必須: CVAT UI の decoder は large 固定 (sam2.1_hiera_large.decoder.onnx)。
#     encoder も large でないと shape は通っても空マスクになる。
#     → 採用 encoder は系統A(no_mem_embed加算済) の large (export_onnx.py 由来)。
#   ● GPU 必須: SAM2_ORT_REQUIRE_GPU=true。CUDAExecutionProvider が無ければ
#     CPU に落とさず明示的に起動失敗する (config/provider/encoder の3層 fail-fast)。
#
# -----------------------------------------------------------------------------
# 8. 補足
# -----------------------------------------------------------------------------
# - `COMPOSE_PROJECT_NAME` 未指定時は `cvat` を使う。別worktreeから既存CVATを操作するため。
# - 管理者作成: `just superuser`。状態確認: `just status`/`just ps`/`just aa-ps`/`just sam2-ps`。
# - CVATからNuclio関数を叩く経路はdashboard経由。direct呼び出しはこの環境だと
#   host.docker.internal:関数port で詰まる。
# - GPU版(pth/ort)はDocker/NVIDIA runtimeが利用可能な環境でのみ使う。
# =============================================================================

compose_project := env_var_or_default("COMPOSE_PROJECT_NAME", "cvat")
host := env_var_or_default("CVAT_HOST", "localhost")
cvat_url := "http://" + host + ":8080"

# SAM2 ONNX Runtime GPU function (sam2-ort-*) settings.
# モデルONNX(848MB, SAM2.1 hiera large)はimageに焼かず、host dirをvolume mountする。env で上書き可。
# large encoder を採用する理由: CVAT UI plugin が large decoder asset を使うため、
# encoder/decoder variant 一致 (large) が必須 (variant 不一致は空マスクになる)。
sam2_ort_model_dir := env_var_or_default("SAM2_ORT_MODEL_DIR", "/home/inaho-omen/Project/cvat-feature-sam2/temp/sam2_models_export/models")
sam2_ort_model_file := env_var_or_default("SAM2_ORT_MODEL_FILE", "sam2.1_hiera_large_encoder.onnx")
sam2_ort_function := "ort-facebookresearch-sam2-hiera-large"
sam2_ort_container := "nuclio-nuclio-ort-facebookresearch-sam2-hiera-large"
sam2_ort_dir := "serverless/onnxruntime/facebookresearch/sam2/nuclio"
compose := "docker compose -p " + compose_project
base_files := "-f docker-compose.yml"
aa_files := "-f docker-compose.yml -f components/serverless/docker-compose.serverless.yml"
aa_dev_files := "-f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml"
ui_files := "-f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.dev.yml"

_default:
    @just --list

default: _default

help: _default

# 通常のCVATだけを起動する。serverless/Nuclioは含めない。
up:
    {{compose}} {{base_files}} up -d

# 通常のCVATだけをbuild込みで起動する。
up-build:
    {{compose}} {{base_files}} up -d --build

# 通常のCVATだけを停止する。serverless込みで起動した場合は `just aa-down` を使う。
down:
    {{compose}} {{base_files}} down

# 通常のCVAT compose状態を見る。
ps:
    {{compose}} {{base_files}} ps

# 通常のCVATの状態をまとめて見る。Bad GatewayやCannot connect時はまずこれを見る。
status:
    @echo "== url =="
    @echo "{{cvat_url}}"
    @echo
    @echo "== containers =="
    {{compose}} {{base_files}} ps
    @echo
    @echo "== health =="
    curl -fsS "{{cvat_url}}/api/server/health/?format=json&org=" | python3 -m json.tool || true

# CVAT URLを表示する。CVAT_HOSTを指定した場合はその値を使う。
# 例: CVAT_HOST=192.168.1.10 just url
url:
    @echo "{{cvat_url}}"

# ローカルブラウザでCVATを開く。GUI環境がない場合はURLだけ表示する。
open:
    @echo "{{cvat_url}}"
    @xdg-open "{{cvat_url}}" >/dev/null 2>&1 || true

# CVAT backendのhealth APIを見る。502/Bad GatewayやCannot connect時の一次確認に使う。
health:
    curl -fsS "{{cvat_url}}/api/server/health/?format=json&org=" | python3 -m json.tool

# CVATが応答するまで待つ。起動直後のmigrationやjob同期待ちに使う。
# 例: just wait 300
wait timeout="180":
    #!/usr/bin/env bash
    set -euo pipefail
    deadline=$((SECONDS + {{timeout}}))
    until curl -fsS "{{cvat_url}}/api/server/health/?format=json&org=" >/dev/null; do
        if (( SECONDS >= deadline )); then
            echo "CVAT health check timed out: {{cvat_url}}" >&2
            exit 1
        fi
        sleep 3
    done
    echo "CVAT is healthy: {{cvat_url}}"

# Docker/Nuclio/CVATの基本状態をまとめて見る。
doctor:
    @echo "== docker compose =="
    docker compose version
    @echo
    @echo "== nuctl =="
    nuctl version || true
    @echo
    @echo "== containers =="
    {{compose}} {{aa_files}} ps
    @echo
    @echo "== cvat health =="
    curl -fsS "{{cvat_url}}/api/server/health/?format=json&org=" | python3 -m json.tool || true
    @echo
    @echo "== nuclio functions =="
    nuctl get function --platform local --namespace nuclio || true

# composeサービスのログを見る。service未指定時はcvat_serverを見る。
# 例: just logs cvat_ui 100
logs service="cvat_server" lines="200":
    {{compose}} {{aa_files}} logs --tail={{lines}} -f {{service}}

# 全サービスの直近ログを流す。量が多いので必要なときだけ使う。
logs-all lines="200":
    {{compose}} {{aa_files}} logs --tail={{lines}} -f

# CVAT backendコンテナでmanage.pyを実行する。
# 例: just manage "showmigrations"
manage command="check":
    {{compose}} {{base_files}} exec cvat_server bash -ic 'python3 ~/manage.py {{command}}'

# 初期管理者ユーザーを対話的に作成する。
superuser:
    {{compose}} {{base_files}} exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

# backendコンテナに入る。手動調査用。
server-shell:
    {{compose}} {{base_files}} exec cvat_server bash

# CVAT + serverless/Nuclio基盤を起動する。SAM/SAM2等のモデル関数はdeployしない。
aa-up:
    {{compose}} {{aa_files}} up -d

# CVAT + serverless/Nuclio基盤をbuild込みで起動する。SAM/SAM2等のモデル関数はdeployしない。
aa-up-build:
    {{compose}} {{aa_files}} up -d --build

# serverless/Nuclio基盤込みのcomposeを停止する。
aa-down:
    {{compose}} {{aa_files}} down

# serverless/Nuclio基盤込みのcompose状態を見る。
aa-ps:
    {{compose}} {{aa_files}} ps

# dev compose + serverless込みで起動する。UI pluginをDocker buildに含めるときに使う。
# 例: just aa-dev-build "plugins/sam2"
aa-dev-build plugins="plugins/sam2":
    CLIENT_PLUGINS={{plugins}} CVAT_HOST={{host}} {{compose}} {{aa_dev_files}} up -d --build

# dev compose + serverless込みで起動する。build済みイメージを使いたいとき用。
aa-dev-up plugins="plugins/sam2":
    CLIENT_PLUGINS={{plugins}} CVAT_HOST={{host}} {{compose}} {{aa_dev_files}} up -d

# dev compose + serverless込みで停止する。
aa-dev-down:
    {{compose}} {{aa_dev_files}} down

# 任意のUI pluginを入れてcvat_uiだけ再ビルドする。既存DBやworkerは触らない。
ui-build plugins="plugins/sam2":
    CLIENT_PLUGINS={{plugins}} {{compose}} {{ui_files}} build cvat_ui

# 任意のUI pluginを入れてcvat_uiだけ再起動する。既存DBやworkerは触らない。
ui-up plugins="plugins/sam2":
    CLIENT_PLUGINS={{plugins}} {{compose}} {{ui_files}} up -d --no-deps --build cvat_ui

# SAM2.1 plugin込みUIのショートカット。
sam2-ui-build:
    CLIENT_PLUGINS=plugins/sam2 {{compose}} {{ui_files}} build cvat_ui

# SAM2.1 plugin込みUIを再ビルド/再起動する。
sam2-ui-up:
    CLIENT_PLUGINS=plugins/sam2 {{compose}} {{ui_files}} up -d --no-deps --build cvat_ui

# Nuclio client versionを見る。
fn-version:
    nuctl version

# Nuclio関数一覧を見る。
fn-list namespace="nuclio":
    nuctl get function --platform local --namespace {{namespace}}

# 各Nuclio関数のNODE PORTを確認する。
# 実際のDocker公開port確認は `just fn-port <function>` または `just sam2-port` を優先する。
fn-ports namespace="nuclio":
    nuctl get function --platform local --namespace {{namespace}}
    @echo
    @echo "Ubuntu firewall例: sudo ufw allow <NODE PORT>/tcp"

# 任意のNuclio関数をCPU版としてdeployする。
# 例: just fn-cpu serverless/pytorch/facebookresearch/sam/nuclio
fn-cpu path="serverless/pytorch/facebookresearch/sam/nuclio":
    ./serverless/deploy_cpu.sh {{path}}

# 任意のNuclio関数をGPU版としてdeployする。
# 例: just fn-gpu serverless/pytorch/facebookresearch/sam/nuclio
fn-gpu path="serverless/pytorch/facebookresearch/sam/nuclio":
    ./serverless/deploy_gpu.sh {{path}}

# 任意のNuclio関数を削除する。
fn-rm function:
    nuctl delete function {{function}} --platform local --namespace nuclio --force

# 任意のNuclio関数を削除する。関数が無い場合も失敗扱いにしない。
fn-rm-safe function:
    nuctl delete function {{function}} --platform local --namespace nuclio --force || true

# Nuclio関数コンテナの公開portを確認する。
fn-port function="pth-facebookresearch-sam2-hiera-large":
    docker port nuclio-nuclio-{{function}} 8080/tcp

# Nuclio関数コンテナのログを見る。
fn-logs function="pth-facebookresearch-sam2-hiera-large" lines="200":
    docker logs --tail={{lines}} -f nuclio-nuclio-{{function}}

# SAM v1をCPU版でdeployする。serverless基盤も自動で起動する。
sam-up: aa-up
    ./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam/nuclio

# SAM v1をGPU版でdeployする。
sam-up-gpu: aa-up
    ./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/nuclio

# SAM v1関数を削除する。serverless基盤自体は停止しない。
sam-down:
    nuctl delete function pth-facebookresearch-sam-vit-h --platform local --namespace nuclio --force

# SAM v1関数を削除する。関数が無い場合も失敗扱いにしない。
sam-down-safe:
    nuctl delete function pth-facebookresearch-sam-vit-h --platform local --namespace nuclio --force || true

# SAM v1状態を見る。
sam-ps:
    nuctl get function --platform local --namespace nuclio | grep -E 'NAME|pth-facebookresearch-sam-vit-h' || true

# SAM2.1をCPU版でdeployする。UI pluginは含めない。
sam2-cpu: aa-up
    ./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam2/nuclio

# SAM2.1をGPU版でdeployする。UI pluginは含めない。
sam2-gpu: aa-up
    ./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam2/nuclio

# SAM2.1のUI plugin、serverless基盤、CPU版Nuclio関数をまとめて起動する。
sam2-up-cpu:
    just sam2-ui-up
    just sam2-cpu
    just sam2-ps

# SAM2.1のUI plugin、serverless基盤、GPU版Nuclio関数をまとめて起動する。
sam2-up-gpu:
    just sam2-ui-up
    just sam2-gpu
    just sam2-ps

# SAM2.1関数を削除する。UIやserverless基盤自体は停止しない。
sam2-down:
    nuctl delete function pth-facebookresearch-sam2-hiera-large --platform local --namespace nuclio --force

# SAM2.1関数を削除する。関数が無い場合も失敗扱いにしない。
sam2-down-safe:
    nuctl delete function pth-facebookresearch-sam2-hiera-large --platform local --namespace nuclio --force || true

# SAM2.1関連の現在状態を見る。
sam2-ps:
    @echo "== nuctl functions =="
    nuctl get function --platform local --namespace nuclio | grep -E 'NAME|pth-facebookresearch-sam2-hiera-large' || true
    @echo
    @echo "== sam2 container =="
    docker ps --filter name=nuclio-nuclio-pth-facebookresearch-sam2-hiera-large || true
    @echo
    @echo "== cvat_ui plugin assets =="
    docker exec cvat_ui sh -lc 'ls -lh /usr/share/nginx/html/assets | grep -E "sam2|decoder|plugin_" || true' || true

# SAM2.1関数のDocker公開portを見る。
sam2-port:
    docker port nuclio-nuclio-pth-facebookresearch-sam2-hiera-large 8080/tcp

# SAM2.1関数コンテナのログを見る。
sam2-logs lines="200":
    docker logs --tail={{lines}} -f nuclio-nuclio-pth-facebookresearch-sam2-hiera-large

# SAM2.1 Nuclio関数の最小推論疎通。64x64 PNGを生成してencoder応答のshape相当サイズを見る。
sam2-test:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 <<'PY'
    import base64
    import io
    import json
    import subprocess
    import urllib.request
    from PIL import Image, ImageDraw

    port_line = subprocess.check_output([
        "docker", "port", "nuclio-nuclio-pth-facebookresearch-sam2-hiera-large", "8080/tcp",
    ], text=True).strip().splitlines()[0]
    port = port_line.rsplit(":", 1)[1]

    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle([16, 16, 48, 48], fill="red")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    body = json.dumps({"image": base64.b64encode(buf.getvalue()).decode()}).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())

    print("sam2_nuclio_port", port)
    for key in ("high_res_feats_0", "high_res_feats_1", "image_embed"):
        print(key, len(base64.b64decode(payload[key])))
    PY

# CVAT APIでlambda関数一覧を見る。ログインが必要なので環境変数を渡す。
# 例: CVAT_USERNAME=... CVAT_PASSWORD=... just aa-test
aa-test username=env_var_or_default("CVAT_USERNAME", "") password=env_var_or_default("CVAT_PASSWORD", ""):
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "{{username}}" || -z "{{password}}" ]]; then
        echo "CVAT_USERNAME と CVAT_PASSWORD を指定してください。" >&2
        exit 2
    fi
    cookie="$(mktemp)"
    payload="$(mktemp)"
    trap 'rm -f "$cookie" "$payload"' EXIT
    CVAT_USERNAME='{{username}}' CVAT_PASSWORD='{{password}}' python3 <<'PY' > "$payload"
    import json
    import os
    print(json.dumps({
        "username": os.environ["CVAT_USERNAME"],
        "password": os.environ["CVAT_PASSWORD"],
    }))
    PY
    curl -fsS -c "$cookie" -H "Content-Type: application/json" -d @"$payload" "{{cvat_url}}/api/auth/login" >/dev/null
    curl -fsS -b "$cookie" "{{cvat_url}}/api/lambda/functions?org=" | python3 -m json.tool

# CVAT APIでSAM2.1関数だけが見えることを確認する。
# 例: CVAT_USERNAME=... CVAT_PASSWORD=... just sam2-api-test
sam2-api-test username=env_var_or_default("CVAT_USERNAME", "") password=env_var_or_default("CVAT_PASSWORD", ""):
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "{{username}}" || -z "{{password}}" ]]; then
        echo "CVAT_USERNAME と CVAT_PASSWORD を指定してください。推論だけ見るなら: just sam2-test" >&2
        exit 2
    fi
    cookie="$(mktemp)"
    payload="$(mktemp)"
    trap 'rm -f "$cookie" "$payload"' EXIT
    CVAT_USERNAME='{{username}}' CVAT_PASSWORD='{{password}}' python3 <<'PY' > "$payload"
    import json
    import os
    print(json.dumps({
        "username": os.environ["CVAT_USERNAME"],
        "password": os.environ["CVAT_PASSWORD"],
    }))
    PY
    curl -fsS -c "$cookie" -H "Content-Type: application/json" -d @"$payload" "{{cvat_url}}/api/auth/login" >/dev/null
    curl -fsS -b "$cookie" "{{cvat_url}}/api/lambda/functions?org=" | python3 -m json.tool | grep -E "Segment Anything 2\\.1|pth-facebookresearch-sam2-hiera-large"

# --- E2E harness recipes ---

# E2Eユーザーを作成/更新する。.envからCVAT_E2E_USER/PASSWORDを読む。
e2e-user:
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_DIR="temp/e2e_sam2/run_$(date +%Y%m%d_%H%M%S)"
    python3 scripts/e2e/sam2/ensure_user.py --run-dir "$RUN_DIR"

# Playwright headlessでCVATにログインし、auth-stateを保存する。
e2e-login:
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_PATH="${HOME}/temp/playwright-cli/node_modules" node scripts/e2e/sam2/playwright_login.js

# E2E task/jobのAPI疎通を確認する。
e2e-job-check:
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_DIR="temp/e2e_sam2/run_$(date +%Y%m%d_%H%M%S)"
    python3 scripts/e2e/sam2/api_check.py --run-dir "$RUN_DIR"

# Playwright headlessでjob画面をロードし、screenshot/console/networkを保存する。
e2e-job-open job_url="http://localhost:8080/tasks/181/jobs/180":
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_PATH="${HOME}/temp/playwright-cli/node_modules" node scripts/e2e/sam2/playwright_job_check.js "{{job_url}}"

# SAM2 BBox baseline E2E。API lambda invokeとPlaywright network captureの2段階で検証する。
e2e-sam2-bbox:
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_PATH="${HOME}/temp/playwright-cli/node_modules" node scripts/e2e/sam2/playwright_sam2_bbox.js

# SAM2 Positive Point E2E。AIツール→Interactor→point modeで左クリック→mask生成を検証する。
e2e-sam2-positive-point:
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_PATH="${HOME}/temp/playwright-cli/node_modules" node scripts/e2e/sam2/playwright_sam2_positive_point.js

# SAM2 Negative Point E2E。positive point後にright-clickでnegative pointを追加し、maskの変化を検証する。
e2e-sam2-negative-point:
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_PATH="${HOME}/temp/playwright-cli/node_modules" node scripts/e2e/sam2/playwright_sam2_negative_point.js

# SAM2 Non-Square E2E。640x360画像でbbox/point座標のスケーリングが正しいことを検証する。
e2e-sam2-non-square:
    #!/usr/bin/env bash
    set -euo pipefail
    NODE_PATH="${HOME}/temp/playwright-cli/node_modules" node scripts/e2e/sam2/playwright_sam2_non_square.js

# SAM2 E2E全テストを順に実行する。login→job check→bbox→positive→negative→non-squareを実行し、同一run dirに成果物を集約する。
e2e-sam2-all:
    #!/usr/bin/env bash
    set -euo pipefail
    RUN_DIR="temp/e2e_sam2/run_all_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RUN_DIR"
    echo "=== SAM2 E2E All ===" | tee "$RUN_DIR/summary.txt"
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RUN_DIR/summary.txt"

    # 各ステップを順に実行。失敗しても全体を止めず記録する。
    steps=("e2e-login" "e2e-job-check" "e2e-sam2-bbox" "e2e-sam2-positive-point" "e2e-sam2-negative-point" "e2e-sam2-non-square")
    pass=0; fail=0
    for step in "${steps[@]}"; do
        echo "--- $step ---" | tee -a "$RUN_DIR/summary.txt"
        rc=0
        just $step > "$RUN_DIR/${step}.log" 2>&1 || rc=$?
        cat "$RUN_DIR/${step}.log"
        if [ "$rc" -eq 0 ]; then
            echo "PASS: $step" | tee -a "$RUN_DIR/summary.txt"
            pass=$((pass + 1))
        else
            echo "FAIL: $step (exit $rc)" | tee -a "$RUN_DIR/summary.txt"
            fail=$((fail + 1))
        fi
    done

    echo "End: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RUN_DIR/summary.txt"
    echo "Pass: $pass / Fail: $fail / Total: $((pass+fail))" | tee -a "$RUN_DIR/summary.txt"
    echo "Run dir: $RUN_DIR" | tee -a "$RUN_DIR/summary.txt"
    [ "$fail" -eq 0 ] || exit 1

# CVAT/Nuclio/SAM2の診断ログを一括収集する。500エラー発生時の事後分析用。
e2e-logs run_dir="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "{{run_dir}}" ]]; then
        bash scripts/e2e/sam2/collect_logs.sh "{{run_dir}}"
    else
        bash scripts/e2e/sam2/collect_logs.sh
    fi

# --- SAM2 ONNX Runtime GPU function recipes (sam2-ort-*) ---
# 注意: これらは PyTorch 版 `sam2-up-cpu`/`sam2-up-gpu`/`sam2-down`/`sam2-logs` とは
# 別系統。ONNX Runtime GPU (CUDAExecutionProvider 必須・CPU fallback 禁止) の
# function `ort-facebookresearch-sam2-hiera-large` を扱う。PyTorch 版とは
# metadata.name が別なので同時 deploy 可能。
# モデルONNX(848MB, large)は image に焼かず host dir (SAM2_ORT_MODEL_DIR) を volume mount する。

# SAM2 ORT GPU 関数を deploy する (モデルを /opt/nuclio/models へ volume mount, CUDA必須, aa-up前提)。
sam2-ort-up-gpu: aa-up
    #!/usr/bin/env bash
    set -euo pipefail
    model_path="{{sam2_ort_model_dir}}/{{sam2_ort_model_file}}"
    if [[ ! -f "$model_path" ]]; then
        echo "encoder ONNX が見つかりません: $model_path" >&2
        echo "SAM2_ORT_MODEL_DIR / SAM2_ORT_MODEL_FILE で配置先を指定してください。" >&2
        exit 2
    fi
    echo "Deploying {{sam2_ort_function}} (model mount: $model_path -> /opt/nuclio/models)"
    nuctl create project cvat --platform local 2>/dev/null || true
    nuctl deploy --project-name cvat \
        --path "{{sam2_ort_dir}}" \
        --file "{{sam2_ort_dir}}/function-ort-gpu.yaml" \
        --platform local \
        --volume "{{sam2_ort_model_dir}}:/opt/nuclio/models" \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
    nuctl get function --platform local --namespace nuclio | grep -E 'NAME|{{sam2_ort_function}}' || true

# SAM2 ORT GPU 関数を削除する。serverless基盤やPyTorch版関数は止めない。
sam2-ort-down:
    nuctl delete function {{sam2_ort_function}} --platform local --namespace nuclio --force

# SAM2 ORT 関数の状態と provider を確認する。logsからCUDAExecutionProviderをgrepする。
sam2-ort-ps:
    @echo "== nuctl function =="
    nuctl get function --platform local --namespace nuclio | grep -E 'NAME|{{sam2_ort_function}}' || true
    @echo
    @echo "== container =="
    docker ps --filter name={{sam2_ort_container}} || true
    @echo
    @echo "== provider (CUDAExecutionProvider) =="
    docker logs {{sam2_ort_container}} 2>&1 | grep -E "Provider|CUDAExecutionProvider|SAM2-ORT" | tail -20 || true

# SAM2 ORT 関数コンテナのログを見る。起動時 provider self-test (model IO, provider) を確認できる。
sam2-ort-logs lines="200":
    docker logs --tail={{lines}} -f {{sam2_ort_container}}

# SAM2 ORT 関数の最小推論疎通。64x64 PNGを生成しencoder応答3出力のサイズを見る。
sam2-ort-test:
    #!/usr/bin/env bash
    set -euo pipefail
    python3 <<'PY'
    import base64
    import io
    import json
    import subprocess
    import urllib.request
    from PIL import Image, ImageDraw

    container = "{{sam2_ort_container}}"
    port_line = subprocess.check_output(
        ["docker", "port", container, "8080/tcp"], text=True
    ).strip().splitlines()[0]
    port = port_line.rsplit(":", 1)[1]

    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle([16, 16, 48, 48], fill="red")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    body = json.dumps({"image": base64.b64encode(buf.getvalue()).decode()}).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())

    print("sam2_ort_nuclio_port", port)
    for key in ("high_res_feats_0", "high_res_feats_1", "image_embed"):
        print(key, len(base64.b64decode(payload[key])))
    PY

# SAM2 ORT core の品質ゲート。uv環境で ruff lint/format check と pytest を実行する。
sam2-ort-lint:
    cd {{sam2_ort_dir}} && uv run ruff check . && uv run ruff format --check . && uv run pytest
