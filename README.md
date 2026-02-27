# 车载氛围灯 AI Agent Backend (Go)

基于 Go 的稳定后端服务，面向 LED 矩阵和灯带场景控制，支持 HTTP API 与 WebSocket 实时推送。

## 已实现能力

- 意图理解与场景规划（中英文关键词启发式 + A/B 模板分流 + 用户偏好记忆）
- LED 矩阵：图片下采样、静态画面生成、动画脚本生成与受限沙箱执行
- LED 灯带：颜色推荐（结合上次状态 + 文本知识库）、效果模式、亮度/速度参数
- 编码输出：RGB24 / RGB565 / RGB111
- 帧元数据：时间戳、帧索引、分辨率、编码
- 持久化：最新矩阵、最新灯带命令、收藏脚本、模板、用户偏好

## 快速启动

```bash
cp .env.example .env
go mod tidy
python3 -m pip install -r requirements-animation.txt
go run ./cmd/server
```

动画模型响应较慢时，可在 `.env` 调整：

```bash
AIHUBMIX_TIMEOUT_SEC=12
ANIMATION_AI_TIMEOUT_SEC=90
```

健康检查：`GET /healthz`

调试前端：`GET /debug/`

## 主要接口（奥卡姆剃刀版）

- 语音入口（提交即生效）
  - `GET/POST /v1/voice/command`
  - 支持 `prompt` 或 `transcript`，收到后立即生成并下发硬件灯效
- App 统一入口（手动文本 + 矩阵能力）
  - `GET/POST /v1/app/command`
  - `operation=scene`：文本场景，立即应用并下发
  - `operation=matrix_latest`：读取已缓存的最新矩阵
  - `operation=matrix_static`：文本生成静态矩阵
  - `operation=matrix_animate`：文本生成矩阵动画
  - `operation=matrix_upload`：图片下采样（multipart 上传 `image`）
- 硬件入口（取包点亮）
  - `GET /v1/hardware/ws?vehicle_id=...`（推荐，实时）
  - `GET /v1/hardware/pull?vehicle_id=...`（轮询）

## 安全与稳定策略（当前实现）

- 上传体积限制 + 扩展名白名单
- 脚本沙箱：临时目录、`python3 -I` 隔离模式、超时强制终止、最小环境变量
- WebSocket：ping/pong 保活、写超时、背压断连

## 后续建议

- 将启发式意图/静态图/动画生成替换为真实 AIHubMix 调用
- 脚本执行升级为容器级沙箱（gVisor/nsjail/firecracker）
- 增加鉴权、限流、审计日志和指标监控
