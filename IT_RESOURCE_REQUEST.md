# 统一氛围灯服务——部署拓扑与服务器资源申请（IT 版）

本文档面向 IT/运维/基础设施团队，用于申请并部署「统一氛围灯服务」（FastAPI）。内容包含：部署拓扑、网络依赖、端口、存储、资源规格建议与运维要点。

---

## 1. 服务概述

- 服务形态：HTTP API（FastAPI + Uvicorn）
- 默认监听：`0.0.0.0:8000`
- 核心能力：
  - 接收自然语言指令
  - 通过外部 AI 服务进行：意图规划（Planner）与矩阵生图（Image Generation）
  - 将结果落盘为本地文件（供硬件侧拉取）
  - 可选：WebSocket 推送、MQTT 广播

代码入口与关键文件：
- HTTP 服务：`main.py`
- 核心逻辑：`api_core.py`
- 矩阵生图：`matrix_service.py`、`image_processor.py`
- 灯带配色：`strip_service.py`
- MCP（可选独立进程）：`mcp_server.py`

---

## 2. 部署拓扑（Deployment Topology）

### 2.1 逻辑拓扑图（draw.io）

- 文件：`IT_DEPLOYMENT_TOPOLOGY.drawio`
- 用法：在 draw.io / diagrams.net 里选择 Import 打开即可编辑。

### 2.2 纯文本拓扑（不依赖渲染）

```text
[语音云/业务服务] --HTTP--> [反向代理/Ingress] --> [统一氛围灯服务(FastAPI:8000)]
                                          |                   |
                                          |                   +--> 出网到 LLM Planner / Image Gen
                                          |                   +--> 本地落盘 latest_*_data.json
                                          |
[设备网关/MCU] <---------HTTP 拉取----------+  (GET /api/data/matrix/raw, GET /api/data/strip)
[设备网关/MCU] ----驱动----> [LED 矩阵] & [LED 灯带]

(可选)
[统一氛围灯服务] --WebSocket(/ws)--> [调试UI/面板]
[统一氛围灯服务] --MQTT--> [MQTT Broker] --MQTT--> [设备网关/MCU]
```

---

## 3. 网络与端口

### 3.1 入站（Inbound）

- HTTP/HTTPS
  - 建议：对外只暴露 `443`（TLS 终止在 SLB/Ingress/Nginx）
  - 服务端口：`8000/TCP`（仅在内网被反代访问）

- WebSocket
  - 路径：`/ws`
  - 复用同一 HTTPS 入口（通过反向代理升级连接）

### 3.2 出站（Outbound）

服务需要可访问外部 API（按实际配置放行域名/IP）：
- `https://aihubmix.com`（Planner/部分模型调用）
- `https://api.bfl.ai`（BFL/FLUX 异步生图）
- 如启用 Google Imagen：需放行相应 Google API 域名（由业务方提供清单）

> 建议 IT：按“域名白名单 + 443”放行，避免全量出网。

### 3.3 MQTT（可选）

- 若启用 MQTT：
  - Broker 端口通常为 `1883/TCP`（明文）或 `8883/TCP`（TLS）
  - Broker 可与本服务同机部署或独立部署（推荐独立）

---

## 4. 存储与数据持久化

服务使用本地 JSON 文件做“当前灯效状态”落盘：
- 矩阵：`latest_led_data.json`
- 灯带：`latest_strip_data.json`

部署要求：
- 容器化部署时，需要挂载**可写的持久化卷**（PVC/hostPath/云盘）
- 备份需求：一般不需要长期备份（可视为缓存/最新状态），但需保障重启后可读取最新状态

容量需求：
- 非图片存储（仅 16×16 像素与 RGB 列表），磁盘占用很小（MB 级别）。

---

## 5. 资源规格建议（服务器/容器）

> 说明：本服务的 CPU/内存占用与“并发请求数 + 生图耗时 + PIL 图像处理”相关。矩阵生图主要耗时在外部网络调用，但本地仍需要处理图片并下采样。

### 5.1 生产环境（建议起步）

- 实例数：`2`（高可用，避免单点）
- 每实例：
  - CPU：`2 vCPU`
  - 内存：`4 GB`
  - 磁盘：`20 GB`（主要用于系统/日志/容器层；业务数据很小）

### 5.2 测试/预发环境（建议起步）

- 实例数：`1`
- CPU：`1~2 vCPU`
- 内存：`2~4 GB`
- 磁盘：`10~20 GB`

### 5.3 需要业务侧补充的容量参数（用于精确评估）

- 预估 QPS / 峰值并发
- 矩阵生图调用频率（每分钟/每小时）
- 是否要求 WebSocket 长连接数（面板/运营端数量）

---

## 6. 运行方式（IT 可执行）

### 6.1 Python 方式（适合 VM）

- 安装依赖：`pip install -r requirements.txt`
- 启动：`uvicorn main:app --host 0.0.0.0 --port 8000`

### 6.2 容器方式（推荐）

- 将服务打包为镜像
- 通过 Deployment/Service/Ingress 暴露
- 挂载持久化卷用于 `latest_*_data.json`

---

## 7. 配置与密钥管理（必须项）

本服务依赖外部 AI API Key（不得硬编码、不得写入镜像/仓库）。建议：

- 通过环境变量注入（K8s Secret / Vault / 配置中心）：
  - `API_KEY`（LLM/中转服务 key）
  - `BFL_API_KEY`（如使用 BFL/FLUX）
  - 以及其它模型相关 key（若启用）

> 当前代码中存在硬编码 key 的历史遗留，生产部署前应由研发侧完成“环境变量化”。

---

## 8. 监控与日志（运维要点）

建议接入：
- 进程存活探针：HTTP `GET /docs` 或自定义 `/healthz`（如后续补充）
- 指标：QPS、P95 延迟、5xx、外部 API 超时率
- 日志：建议采集 stdout/stderr

关键风险点：
- 外部 AI 服务不可用/慢：会导致生成延迟或失败（应在网关侧做超时与兜底）
- 出网限制：如未放行域名，将导致核心能力不可用

---

## 9. 对接清单（IT 交付项）

- 服务器/容器资源：按第 5 节
- 入口：Ingress/SLB + TLS（建议）
- 出网策略：按第 3.2 节域名白名单
- 可写持久化卷：用于 `latest_*_data.json`
- （可选）MQTT Broker：独立部署并提供连接信息
