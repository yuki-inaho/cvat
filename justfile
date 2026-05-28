set shell := ["bash", "-euo", "pipefail", "-c"]

# 使い方:
# - 利用可能なコマンド一覧: `just --list`
# - 通常のCVAT起動/停止: `just up` / `just down`
# - 半自動アノテーション基盤の起動/停止: `just aa-up` / `just aa-down`
# - SAM2.1のCPU起動/停止: `just sam2-up-cpu` / `just sam2-down`
# - SAM2.1のGPU起動: `just sam2-up-gpu`
# - 状態確認: `just status` / `just ps` / `just aa-ps` / `just sam2-ps`
# - ログ確認: `just logs cvat_server` / `just sam2-logs`
# - 管理者作成: `just superuser`
#
# 注意:
# - `COMPOSE_PROJECT_NAME` 未指定時は `cvat` を使う。別worktreeから既存CVATを操作するため。
# - `aa-*` はCVATのserverless/Nuclio基盤だけを起動・停止する。
# - `sam*` / `sam2-*` は対象モデルのNuclio関数をdeploy/deleteする。
# - `sam2-up-cpu` はUI pluginも必要なので、`sam2-ui-up` とSAM2 CPU関数deployをまとめて行う。
# - GPU版はDocker/NVIDIA runtimeが利用可能な環境でのみ使う。
# - CVATからNuclio関数を叩く経路はdashboard経由。direct呼び出しはこの環境だとhost.docker.internal:関数portで詰まる。

compose_project := env_var_or_default("COMPOSE_PROJECT_NAME", "cvat")
host := env_var_or_default("CVAT_HOST", "localhost")
cvat_url := "http://" + host + ":8080"
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
