# 公司 Windows → 小网 Linux：OpenCode 与 SSH 反向隧道操作手册

记录在公司环境里用自有大模型 API，先在 Windows 落地 OpenCode，再 SSH 连小网 Linux，并用「只跟我这条 SSH」的反向端口转发访问大网 LLM 的全套做法。

相关简述见仓库根目录的 `markdown` 文件（OpenCode vs Codex 选型）。本文是可照着做的操作清单。

技术约定：

- **大网**：公司办公网 / 能访问大模型 API 的那侧。Windows 公司机通常在这边。
- **小网**：只有内网 IP 的 Linux（实验机 / GPU 机）。
- **反向隧道**：这里指 `ssh -R`，不是 nginx 常驻反代。

正文里的主机名、IP、端口、模型 ID 都是占位符，换成你们实际值。**不要把 API key 写进仓库。**

---

## 0. 先做哪个判断

| 问 | 怎么验 |决定什么 |
|---|---|---|
| Windows 能不能调公司 LLM？ | 浏览器或 `curl` 公司 `baseURL` | 不能就别往下走，先要网关 / 证书 |
| Windows 能不能 SSH 到小网 Linux？ | `ssh user@10.x.x.x` | 不能就先要跳板 / VPN / 白名单 |
| 小网 Linux 能不能直接访问 LLM？ | 在 Linux 上 `curl` 同一 `baseURL` | **能** → 情况 A；**不能** → 情况 B，要 SSH `-R` |

**情况 A（小网也能打 API）**
Windows 只当跳板。SSH 进 Linux，在 Linux 上跑 OpenCode。不要做反向隧道。

**情况 B（只有 Windows 能打 API）**
用 Windows 发起 `ssh -R`，把大网 LLM 倒挂到 Linux 的 `127.0.0.1:端口`。只有这条 SSH 还活着时，Linux 上的 agent 才能调模型。

不要让 Windows 上的 agent 隔空改 Linux 工程。代码、GPU、文件都在 Linux 上时，agent 也应该跑在 Linux 上。

---

## 1. 为什么选 OpenCode，不选 Codex

公司 Windows + 自有 API key + 以后还要装到小网 Linux → **先上 OpenCode**。

Codex 默认走 ChatGPT 登录和 OpenAI Responses API。接公司网关常见坑：

- 网关只有 `/v1/chat/completions`，Codex 却走 `/v1/responses` 或 WebSocket
- 部分版本会忽略 `OPENAI_BASE_URL`，必须写 `~/.codex/config.toml` 的 `model_providers`
- 自定义网关 HTTPS fallback 丢过 `Authorization` 的前例
- Windows 官方更推 Win11 原生 App 或 WSL，公司机常常两样都不好装

只有同时满足再考虑 Codex：

1. key 是 ChatGPT Enterprise / 官方 OpenAI，或网关文档写明兼容 Codex `responses`
2. Windows 11，IT 允许官方 Codex App 或 WSL
3. 更在意 Codex 在 GPT 系列上的编码能力，而不是「任意公司 key 都能接」

OpenCode 对这个场景：

- 公司 key 是一等公民：`/connect` 或 `opencode.json` 里 `@ai-sdk/openai-compatible` + `baseURL`
- Windows：Scoop / Chocolatey / npm / 桌面安装包；Scoop 通常不需要管理员
- Linux：`curl -fsSL https://opencode.ai/install | bash`，同一份配置拷过去
- MIT 开源，key 只存本机 `auth.json`，不绑 OpenAI 账号

官方仍建议 Windows 用 WSL。公司机不准 WSL 就走 **Scoop 原生 CLI + Windows Terminal**。

---

## 2. Windows 上安装 OpenCode

优先 Scoop（当前用户、常免管理员）。在 **普通用户** PowerShell 里（不要「以管理员身份运行」）：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop install git
scoop bucket add extras
scoop install extras/opencode
opencode --version
```

若已有 Node.js：

```powershell
npm i -g opencode-ai
opencode --version
```

Chocolatey（通常要管理员）：

```powershell
choco install opencode
```

也可从 [opencode.ai/download](https://opencode.ai/download) 下 Windows 桌面安装包。

若 IT 允许 WSL，体验更接近 Linux：

```powershell
wsl --install
```

重启后进 WSL：

```bash
curl -fsSL https://opencode.ai/install | bash
```

---

## 3. 接公司大模型 API（Windows 本机先验证）

把下面的 URL、模型名换成你们的。文件放：

`%USERPROFILE%\.config\opencode\opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider" {
    "company": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Company LLM",
      "options": {
        "baseURL": "https://llm.internal.example.com/v1"
      },
      "models": {
        "your-model-id": {
          "name": "Company Coder",
          "tool_call": true
        }
      }
    }
  }
}
```

启动：

```powershell
cd D:\path\to\some-small-repo
opencode
```

TUI 里：

1. `/connect` → 选公司 provider 或 Other → 粘 API key
2. `/models` → 选 `your-model-id`
3. 先让它在 Windows 本地改一个小仓库，确认 **tool call + 流式** 都通

key 会写到 `%USERPROFILE%\.local\share\opencode\auth.json`，不要把这个文件提交到 git。

### TLS / 公司中间人证书

连不上时先怀疑证书，不要怀疑模型。

- 把公司根证加进 Windows「信任的根证书预发机构」
- 或给当前进程设：

```powershell
$env:SSL_CERT_FILE = "C:\path\to\company-root.pem"
$env:NODE_EXTRA_CA_CERTS = "C:\path\to\company-root.pem"
```

Codex 对应变量是 `CODEX_CA_CERTIFICATE`。OpenCode 走 Node / 系统证书。

---

## 4. 从 Windows SSH 进小网 Linux

```powershell
ssh user@10.x.x.x
```

建议 `%USERPROFILE%\.ssh\config`：

```sshconfig
Host xiaowang
    HostName 10.x.x.x
    User youruser
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

之后：

```powershell
ssh xiaowang
```

登录后先在 **Linux 上** 测 API：

```bash
curl -sS https://llm.internal.example.com/v1/models \
  -H "Authorization: Bearer $KEY"
```

- 通 → 情况 A，跳到第 6 节直接装 OpenCode
- 不通 → 情况 B，做第 5 节反向隧道

---

## 5. 只让我这条 SSH 走「反向代理」（`ssh -R`）

小网机若路由打不到大网 LLM，**不能**在小网上跑 nginx 去反代大网。能做的是：
Windows（能访问 API）把大网能力经 SSH **倒挂** 进 Linux 的 localhost。

```
Windows（大网，能调 LLM）
        ssh -R   ← 只有这条会话
                ↓
小网 Linux  127.0.0.1:8080 或 9443
        → 钻回 Windows
        → 大网 LLM
```

- 只在你这条 SSH 还活着时端口才存在
- 不改 Linux 默认路由、`/etc/environment`、系统 `http_proxy`
- 断开 Windows 侧 `ssh -N` 之后，Linux 上这个端口立刻没了

**不要改** Linux `/etc/ssh/sshd_config` 的 `GatewayPorts yes`。默认 `GatewayPorts no` 正是只给本机 localhost。

### 5.1 HTTP 网关（最干净，优先）

公司 API 若是 `http://llm.internal.example.com:8080/v1`：

Windows PowerShell（专门挂隧道的窗口，不要关）：

```powershell
ssh -N -R 127.0.0.1:8080:llm.internal.example.com:8080 xiaowang
```

或写进 `~/.ssh/config` 的 `Host xiaowang`：

```sshconfig
Host xiaowang
    HostName 10.x.x.x
    User youruser
    RemoteForward 127.0.0.1:8080 llm.internal.example.com:8080
    ExitOnForwardFailure yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

```powershell
ssh -N xiaowang
```

另开窗口登录 Linux，只在你的 shell 里验证：

```bash
curl -sS http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer $KEY"
```

OpenCode（**只** 写你的 `~/.config/opencode/opencode.json`，不要写 `/etc`）：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "company": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Company LLM via SSH",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "your-model-id": {
          "name": "Company Coder",
          "tool_call": true
        }
      }
    }
  }
}
```

没有证书、没有 `/etc/hosts` 污染。

### 5.2 HTTPS 网关

Windows：

```powershell
ssh -N -R 127.0.0.1:9443:llm.internal.example.com:443 xiaowang
```

Linux 上用 `--resolve` 验证（不要改 `/etc/hosts`，那会影响整机）：

```bash
curl -sS https://llm.internal.example.com:9443/v1/models \
  --resolve llm.internal.example.com:9443:127.0.0.1 \
  -H "Authorization: Bearer $KEY"
```

OpenCode / Node 不读 curl 的 `--resolve`。能用 HTTP 就用 5.1。必须 HTTPS 时，只在**当前 tmux 窗口**里配 `baseURL`，并处理证书主机名；不要把域名写进 `/etc/hosts`。

### 5.3 更严：Unix socket（同机其他账号不能用 localhost 端口）

`127.0.0.1:8080` 同机其他登录用户也能连。要更严可绑到 home 下 socket：

Windows：

```powershell
ssh -N -R /home/youruser/.ssh/llm.sock:llm.internal.example.com:8080 xiaowang
```

Linux `sshd` 需要 `StreamLocalBindUnlink yes`（这是服务端能力，不是开全局代理）。socket 权限 `700`。OpenCode 要 TCP 时，在**你的会话**里：

```bash
socat TCP-LISTEN:8080,bind=127.0.0.1,fork UNIX-CONNECT:$HOME/.ssh/llm.sock
```

### 5.4 每天用法

1. Windows 挂着 `ssh -N xiaowang`（隧道）
2. 再开一个窗口 `ssh xiaowang`，进 tmux，只在这个会话里跑 OpenCode
3. 下班关掉 Windows 上的 `ssh -N`，小网这边自动没入口

---

## 6. 小网 Linux 上装 OpenCode

情况 A：直接 `curl` 公司 `baseURL` 成功后装。
情况 B：先挂好第 5 节隧道，`curl 127.0.0.1:...` 成功后再装。

```bash
curl -fsSL https://opencode.ai/install | bash
mkdir -p ~/.config/opencode
```

把 Windows 上的 `opencode.json` 拷到 `~/.config/opencode/`（情况 B 把 `baseURL` 改成 `http://127.0.0.1:8080/v1`）。

用 `/connect` 再录一次 key，不要用 U 盘明文到处拷 `auth.json`。

```bash
cd /path/to/project
tmux new -s agent
opencode
```

之后从公司 Windows `ssh xiaowang` 进去 `tmux attach -t agent` 就能接着干。

---

## 7. 不要做的事

| 做法 | 为什么不要 |
|---|---|
| nginx / Caddy 常驻反代 | 谁都能打，不是「只跟我的 SSH」 |
| `export http_proxy=...` 写进 `/etc/profile` | 整机、所有用户 |
| iptables 把 443 全重定向 | 整机劫持 |
| `GatewayPorts yes` + 监听 `0.0.0.0` | 小网里别人能用你的隧道打大网 API |
| 改 `/etc/hosts` 把 LLM 域名指到 127.0.0.1 | 所有进程解析都变 |
| 把 API key 写进 git / 本仓库 | 泄密 |
| Windows agent 跟期隔空改 Linux 工程 | 沙箱、路径、GPU 都不对 |

---

## 8. 排障

| 现象 | 先查 |
|---|---|
| Windows `opencode` 连不上模型 | `baseURL` 是否带 `/v1`；key；公司根证 |
| `ssh -N` 报 `remote port forwarding failed` | 远端 8080/9443 已被占；或 `AllowTcpForwarding` 被关 |
| Linux `curl 127.0.0.1:8080` Connection refused | Windows 上 `ssh -N` 断了 |
| HTTPS 证书主机不匹配 | 用 `--resolve`，别用 `https://127.0.0.1` 直接打 |
| tool call 不生效 | 模型 / 网关不支持 `tools`；`opencode.json` 里 `tool_call: true` |
| Scoop 报管理员 | 换普通用户 PowerShell，别用管理员窗口 |

确认远端允许转发（读即可，不要随便改成 `GatewayPorts yes`）：

```bash
sshd -T | grep -E 'allowtcpforwarding|gatewayports'
```

期望：`allowtcpforwarding yes`，`gatewayports no`。

---

## 9. 一句话流程

1. 公司 Windows 装 OpenCode，`opencode.json` 接公司 `baseURL`，`/connect` 贴 key，本地小仓库验证 tool call。
2. `ssh xiaowang`，在 Linux 上 `curl` 公司 API。
3. 能直连 → Linux 装同一份 OpenCode，tmux 里跑，Windows 只当 SSH 入口。
4. 不能直连 → Windows `ssh -N -R 127.0.0.1:8080:...`，Linux OpenCode 的 `baseURL` 指 `http://127.0.0.1:8080/v1`。
5. 不要 nginx、不要系统代理、不要 `GatewayPorts yes`。隧道只跟你这条 SSH。
