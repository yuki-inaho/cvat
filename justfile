set shell := ["bash", "-euo", "pipefail", "-c"]

# 使い方:
# - このリポジトリはRTK運用なので、Codexから実行するときは `rtk just <recipe>` で呼ぶ。
# - 利用可能なコマンド一覧: `rtk just --list`
# - 通常のCVAT起動: `rtk just cvat-up`
# - 半自動/自動アノテーション基盤込みの起動: `rtk just cvat-aa-up`
# - SAM1 Nuclio interactor起動: `rtk just sam-cpu-up` / `rtk just sam-gpu-up`
# - SAM2.1 Nuclio interactor起動: `rtk just sam2-cpu-up` / `rtk just sam2-gpu-up`
# - SAM2.1用UI plugin込みでUIを再ビルド/再起動: `rtk just sam2-ui-up`
# - 状態確認: `rtk just cvat-aa-ps` と `rtk just aa-functions`
# - 停止: `rtk just cvat-down` または `rtk just cvat-aa-down`
#
# 注意:
# - `COMPOSE_PROJECT_NAME` 未指定時は `cvat` を使う。別worktreeから既存CVATを操作するため。
# - 最新developでは `ai-models/agents_deployment/sam2` は削除済み。SAM2 agent系はcvat-modelsへ移動済み。
# - このjustfileのSAM2は、hashJoe系SAM2.1をベースにしたNuclio interactor + UI pluginの検証用。
# - serverless起動時はCVATからNuclioをdashboard経由で呼ぶ。direct呼び出しはこの環境だとhost.docker.internal:関数portで詰まる。
# - GPU版はDocker/NVIDIA runtimeが利用可能な環境でのみ使う。

compose_project := env_var_or_default("COMPOSE_PROJECT_NAME", "cvat")
compose := "docker compose -p " + compose_project
serverless_files := "-f docker-compose.yml -f components/serverless/docker-compose.serverless.yml"
ui_plugin_files := "-f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.dev.yml"

_default:
    @just --list

default: _default

help: _default

# 基本のCVAT操作
cvat-up:
    {{compose}} -f docker-compose.yml up -d

cvat-up-build:
    {{compose}} -f docker-compose.yml up -d --build

cvat-down:
    {{compose}} -f docker-compose.yml down

cvat-ps:
    {{compose}} -f docker-compose.yml ps

# CVAT + Serverlessプラットフォーム（半自動/自動アノテーション向け共通基盤）
cvat-aa-up:
    {{compose}} {{serverless_files}} up -d

cvat-aa-up-build:
    {{compose}} {{serverless_files}} up -d --build

cvat-aa-down:
    {{compose}} {{serverless_files}} down

cvat-aa-ps:
    {{compose}} {{serverless_files}} ps

# UI pluginを入れてcvat_uiだけ再ビルド/再起動する。既存DBやworkerは触らない。
cvat-ui-plugin-build plugins="plugins/sam2":
    CLIENT_PLUGINS={{plugins}} {{compose}} {{ui_plugin_files}} build cvat_ui

cvat-ui-plugin-up plugins="plugins/sam2":
    CLIENT_PLUGINS={{plugins}} {{compose}} {{ui_plugin_files}} up -d --no-deps --build cvat_ui

# SAM2.1 plugin込みUIのショートカット。
sam2-ui-build:
    CLIENT_PLUGINS=plugins/sam2 {{compose}} {{ui_plugin_files}} build cvat_ui

sam2-ui-up:
    CLIENT_PLUGINS=plugins/sam2 {{compose}} {{ui_plugin_files}} up -d --no-deps --build cvat_ui

# Nuclio / 関数管理
aa-nuctl-version:
    nuctl version

aa-functions:
    nuctl get function --platform local

aa-deploy-cpu path="serverless/pytorch/facebookresearch/sam/nuclio":
    ./serverless/deploy_cpu.sh {{path}}

aa-deploy-gpu path="serverless/pytorch/facebookresearch/sam/nuclio":
    ./serverless/deploy_gpu.sh {{path}}

aa-remove function="pth-facebookresearch-sam-vit-h":
    nuctl delete function {{function}} --platform local --namespace nuclio --force

# プリセット: Segment Anything v1をCPU/GPUで起動
sam-cpu-up: cvat-aa-up
    ./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam/nuclio

sam-gpu-up: cvat-aa-up
    ./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/nuclio

sam-stop:
    nuctl delete function pth-facebookresearch-sam-vit-h --platform local --namespace nuclio --force

# プリセット: Segment Anything 2.1をCPU/GPUで起動
# UI側のONNX decoder pluginも必要なので、通常は `sam2-ui-up` と組み合わせる。
sam2-cpu-up: cvat-aa-up
    ./serverless/deploy_cpu.sh serverless/pytorch/facebookresearch/sam2/nuclio

sam2-gpu-up: cvat-aa-up
    ./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam2/nuclio

sam2-stop:
    nuctl delete function pth-facebookresearch-sam2-hiera-large --platform local --namespace nuclio --force

# SAM2.1の最小API疎通。CVATへログインして関数一覧にSegment Anything 2.1が出ることを見る。
# 例: CVAT_USERNAME=... CVAT_PASSWORD=... rtk just sam2-smoke-api
sam2-smoke-api username=env_var_or_default("CVAT_USERNAME", "") password=env_var_or_default("CVAT_PASSWORD", ""):
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "{{username}}" || -z "{{password}}" ]]; then
        echo "CVAT_USERNAME と CVAT_PASSWORD を指定してください。推論だけ見るなら: rtk just sam2-smoke-nuclio" >&2
        exit 2
    fi
    cookie="$(mktemp)"
    trap 'rm -f "$cookie"' EXIT
    CVAT_USERNAME='{{username}}' CVAT_PASSWORD='{{password}}' python3 <<'PY' > /tmp/cvat_sam2_login.json
    import json
    import os
    print(json.dumps({
        "username": os.environ["CVAT_USERNAME"],
        "password": os.environ["CVAT_PASSWORD"],
    }))
    PY
    curl -fsS -c "$cookie" -H "Content-Type: application/json" -d @/tmp/cvat_sam2_login.json http://localhost:8080/api/auth/login >/dev/null
    curl -fsS -b "$cookie" http://localhost:8080/api/lambda/functions | python3 -m json.tool | grep -E "Segment Anything 2\\.1|pth-facebookresearch-sam2-hiera-large"

# SAM2.1 Nuclio関数の最小推論疎通。64x64 PNGを生成してencoder応答のshape相当サイズを見る。
sam2-smoke-nuclio:
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

# 旧SAM2 agent系コマンド。最新developでは本体がcvat-modelsへ移動済みのため、このbranchではNuclio版を使う。
sam2-agent-up-cpu:
    @echo "ai-models/agents_deployment/sam2 は最新developから削除済みです。代わりに: rtk just sam2-cpu-up"
    @exit 1

sam2-agent-up-gpu:
    @echo "ai-models/agents_deployment/sam2 は最新developから削除済みです。代わりに: rtk just sam2-gpu-up"
    @exit 1

sam2-agent-up:
    @echo "ai-models/agents_deployment/sam2 は最新developから削除済みです。代わりに: rtk just sam2-cpu-up または rtk just sam2-gpu-up"
    @exit 1

sam2-agent-down:
    @echo "このbranchではSAM2 agent composeは管理しません。Nuclio版の停止は: rtk just sam2-stop"

sam2-agent-deregister:
    @echo "このbranchではSAM2 agent登録解除は不要です。Nuclio版の停止は: rtk just sam2-stop"

sam2-agent-logs:
    @echo "Nuclio版SAM2のログはDocker上の関数コンテナまたはNuclio dashboardで確認してください。"
