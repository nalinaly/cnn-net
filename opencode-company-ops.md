# 公司 Windows → 小网 Linux：OpenCode 与 SSH 反向隧道操作手册

记录在公司环境里用自有大模型 API：Windows 已能直连公司大网网关（IP:端口）+ API key 跑 OpenCode；SSH 过去的小网 Linux **curl 不了** 同一地址。因此 Linux 上的 agent 必须经 `ssh -R` 把大网 LLM 倒挂到本机 `127.0.0.1`。

技术约定：

- **大网**：公司办公网 / 能访问大模型 API 的那侧。本 Windows 公司机在这边。
- **小网**：SSH 过去的 Linux（实验机 / GPU 机）。路由打不到公司 LLM。
- **反向隧道**：这里指 `ssh -R`，不是 nginx 常驻反代。

全程只用 **IP:端口** 和 **127.0.0.1**。不要域名、不要 DNS、不要 `/etc/hosts`、不要 `curl --resolve`。正文里的 IP、端口、模型 ID 都是占位符，换成你们实际值。**不要把 API key 写进仓库。**

---

## 0. 当前已核实的条件（按这个走，不要再猜）

| 条件 | 现状 | 含义 |
|---|---|---|
| 本 Windows 能否调公司 LLM | **能**。浏览器 / `curl` / OpenCode 已用公司网关 + API key 跑通 | Windows 是大网侧，OpenCode 本机配置可当模板 |
| Windows 能否 SSH 到小网 Linux | 需要你自己保证 `ssh` 通 | 不通就先要跳板 / VPN / 白名单 |
| SSH 过去的机器能否 `curl` 公司网关 | **不能** | **不要**在 Linux 上直连那个 IP；必须做第 5 节 `ssh -R` |

因此默认路径是原来的 **情况 B**：

- Windows：OpenCode 直连公司网关 `http://10.x.x.x:8080/v1`（已完成；把 `10.x.x.x:8080` 换成你本机已通的 IP:端口）
- Linux：OpenCode 的 `baseURL` 只能是隧道倒挂出来的 `http://127.0.0.1:8080/v1`
- 代码、GPU、文件在 Linux 上时，agent 也跑在 Linux 上；不要用 Windows agent 隔空改 Linux 工程

若以后某台 Linux **能** `curl` 通公司网关 IP，才走文末「例外」——当前环境不要按那个做。

---

## 1. 为什么选 OpenCode，不选 Codex

公司 Windows + 自有 API key + 还要装到小网 Linux → **用 OpenCode**。

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
- Linux：`curl -fsSL https://opencode.ai/install | bash`（安装包本身走公网；**调模型**必须走第 5 节隧道）
- MIT 开源，key 只存本机 `auth.json`，不绑 OpenAI 账号

官方仍建议 Windows 用 WSL。公司机不准 WSL 就走 **Scoop 原生 CLI + Windows Terminal**。

---

## 2. Windows 上安装 OpenCode

当前条件默认 **Windows 已经能用公司网关 + key 跑 OpenCode**。下面仅在新机重装时用。

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

若 IT 允许 WSL：

```powershell
wsl --install
```

重启后进 WSL：

```bash
curl -fsSL https://opencode.ai/install | bash
```

---

## 3. Windows 本机接公司大模型（已通，作模板）

当前条件：本 Windows **已经**可以用公司网关和 API key 跑 OpenCode。本节当作「配置长什么样」，给 Linux 侧对照，不要再当成未完成步骤。

`baseURL` 写成 **IP:端口**，不要写域名。若你现在配置里还是域名，先在 Windows 上改成对应 IP，确认本机仍能通，再去做隧道。

文件：

`%USERPROFILE%\.config\opencode\opencode.json`

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "company": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Company LLM",
      "options": {
        "baseURL": "http://10.x.x.x:8080/v1"
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

本机验证（已做过就跳过）：

```powershell
cd D:\\path\\to\\some-small-repo
opencode
```

TUI：`/connect` 贴 key，`/models` 选模型，确认 **tool call + 流式**。

key 在 `%USERPROFILE%\.local\share\opencode\auth.json`，不要提交 git。

网关若是 HTTPS 且绑在 IP 上，Windows 已通则不必再折腾证书。新机连不上时再查公司根证：

- 进 Windows「信任的根证书颁发机构」
- 或：

```powershell
$env:SSL_CERT_FILE = "C:\\path\\to\\company-root.pem"
$env:NODE_EXTRA_CA_CERTS = "C:\\path\\to\\company-root.pem"
```

不要用改 hosts、不要为证书去配域名。隧道侧默认走 **HTTP + IP**（第 5 节），Linux 上就不会碰到证书主机名。

---

## 4. 从 Windows SSH 进小网 Linux

SSH 也写 IP，不要写需要解析的主机名：

```powershell
ssh user@10.y.y.y
```

建议 `%USERPROFILE%\.ssh\config`（`Host` 只是本机别名，`HostName` 必须是 IP）：

```sshconfig
Host xiaowang
    HostName 10.y.y.y
    User youruser
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

之后：

```powershell
ssh xiaowang
```

**不要指望 Linux 上这条能通**（把 IP:端口换成你 Windows 已通的网关）：

```bash
curl -sS http://10.x.x.x:8080/v1/models \\
  -H "Authorization: Bearer $KEY"
```

当前条件就是：**SSH 机器 curl 不了公司网关。** 测一下只为确认「还是不通」，然后直接做第 5 节。不要在 Linux 上把 OpenCode 的 `baseURL` 写成那个 IP。

---

## 5. 只让我这条 SSH 走「反向代理」（`ssh -R`）——当前默认路径

小网机路由打不到大网 LLM，**不能**在小网上跑 nginx 去反代。能做的是：Windows（能访问 API）把大网能力经 SSH **倒挂** 进 Linux 的 `127.0.0.1`。

```
Windows（大网，OpenCode 已直连 http://10.x.x.x:8080）
        ssh -R   ← 只有这条会话
                ↓
小网 Linux  127.0.0.1:8080
        → 钻回 Windows
        → 10.x.x.x:8080
```

- 只在你这条 SSH 还活着时端口才存在
- 不改 Linux 默认路由、`/etc/environment`、系统 `http_proxy`
- 不改 DNS、不改 `/etc/hosts`
- 断开 Windows 侧 `ssh -N` 之后，Linux 上这个端口立刻没了

**不要改** Linux `/etc/ssh/sshd_config` 的 `GatewayPorts yes`。默认 `GatewayPorts no` 正是只给本机 localhost。

### 5.1 HTTP + IP（唯一默认做法）

右侧写 Windows 已通的网关 **IP:端口**，左侧写 Linux 的 `127.0.0.1:8080`。

Windows PowerShell（专门挂隧道的窗口，不要关）：

```powershell
ssh -N -R 127.0.0.1:8080:10.x.x.x:8080 xiaowang
```

或写进 `~/.ssh/config`：

```sshconfig
Host xiaowang-llm
    HostName 10.y.y.y
    User youruser
    RemoteForward 127.0.0.1:8080 10.x.x.x:8080
    ExitOnForwardFailure yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

```powershell
ssh -N xiaowang-llm
```

直连 Linux 用另一个 Host（不要把 `-R` 绑在你日常改代码的那条 SSH 上，隧道断了不影响登录）：

```sshconfig
Host xiaowang
    HostName 10.y.y.y
    User youruser
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

另开窗口登录 Linux，只在你的 shell 里验证（验的是 **localhost 倒挂**，不是公司网关 IP）：

```bash
curl -sS http://127.0.0.1:8080/v1/models \\
  -H "Authorization: Bearer $KEY"
```

这里通了，才说明隧道可用。`curl 10.x.x.x:8080` 失败是预期，不要据此改回直连。

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

和 Windows 那份的差别：**只有 `baseURL`**。Windows 用 `http://10.x.x.x:8080/v1`；Linux 用 `http://127.0.0.1:8080/v1`。

没有证书、没有域名、没有 DNS。

### 5.2 更严：Unix socket（同机其他账号不能用 localhost 端口）

`127.0.0.1:8080` 同机其他登录用户也能连。要更严可绑到 home 下 socket。右侧仍然是 IP，不是域名：

Windows：

```powershell
ssh -N -R /home/youruser/.ssh/llm.sock:10.x.x.x:8080 xiaowang
```

Linux `sshd` 需要 `StreamLocalBindUnlink yes`（这是服务端能力，不是开全局代理）。socket 权限 `700`。OpenCode 要 TCP 时，在**你的会话**里：

```bash
socat TCP-LISTEN:8080,bind=127.0.0.1,fork UNIX-CONNECT:$HOME/.ssh/llm.sock
```

然后 Linux OpenCode 的 `baseURL` 仍是 `http://127.0.0.1:8080/v1`。

### 5.3 每天用法

1. Windows 挂着 `ssh -N xiaowang-llm`（倒挂公司 LLM，目这个窗口）
2. 再开窗口 `ssh xiaowang`（或 VS Code Remote-SSH），进 tmux，只在这个会话里跑 OpenCode
3. 下班关掉 Windows 上的 `ssh -N`，小网这边 `127.0.0.1:8080` 立刻没了

隧道窗口掉了，Linux 上 `curl 127.0.0.1:8080` 会 Connection refused，OpenCode 调模型失败。直连 SSH / VS Code 可以还在。

---

## 6. 小网 Linux 上装 OpenCode

当前条件必须先挂好第 5 节隧道，并且：

```bash
curl -sS http://127.0.0.1:8080/v1/models \\
  -H "Authorization: Bearer $KEY"
```

成功后再装（或先装二进制、等隧道通了再配 `baseURL`）。

安装包来自 `opencode.ai`（公网）。若小网也 curl 不了公网，把 Windows 上的 `opencode` 二进制拷过去，或走你们已有的内网软件源。**不要**把「装得上 OpenCode」和「调得了公司模型」混为一谈。

```bash
curl -fsSL https://opencode.ai/install | bash
mkdir -p ~/.config/opencode
```

把 Windows 的 `opencode.json` 拷到 `~/.config/opencode/`，**把 `baseURL` 改成** `http://127.0.0.1:8080/v1`。不要继续用公司网关 IP，更不要写域名。

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
| 写域名、配 DNS、改 `/etc/hosts`、`curl --resolve` | 本文全程 IP + `127.0.0.1`，不搞解析 |
| 在 Linux 上把 OpenCode `baseURL` 写成公司网关 IP | 当前 SSH 机不了那个 IP，只会连超时 |
| nginx / Caddy 常驻反代 | 谁都能打，不是「只跟我的 SSH」 |
| `export http_proxy=...` 写进 `/etc/profile` | 整机、所有用户 |
| iptables 把 443 全重定向 | 整机劫持 |
| `GatewayPorts yes` + 监听 `0.0.0.0` | 小网里别人能用你的隧道打大网 API |
| 把 API key 写进 git / 本仓库 | 泄密 |
| Windows agent 长期隔空改 Linux 工程 | 沙箱、路径、GPU 都不对 |

---

## 8. 排障

| 现象 | 先查 |
|---|---|
| Windows `opencode` 连不上模型 | 与当前条件不符；查 `baseURL` 是否是 `http://IP:端口/v1`、key、公司根证 |
| Linux `curl` 公司网关 IP 失败 | **预期**。应 `curl 127.0.0.1:8080` |
| `ssh -N` 报 `remote port forwarding failed` | 远端 8080 已被占；或 `AllowTcpForwarding` 被关 |
| Linux `curl 127.0.0.1:8080` Connection refused | Windows 上 `ssh -N` 断了或假活，重开隧道 |
| 证书 / 主机名报错 | 你还在走 HTTPS 域名。改回 HTTP + IP，Linux 只用 `127.0.0.1` |
| tool call 不生效 | 模型 / 网关不支持 `tools`；`opencode.json` 里 `tool_call: true` |
| Scoop 报管理员 | 换普通用户 PowerShell，别用管理员窗口 |

确认远端允许转发（读即可，不要随便改成 `GatewayPorts yes`）：

```bash
sshd -T | grep -E 'allowtcpforwarding|gatewayports'
```

期望：`allowtcpforwarding yes`，`gatewayports no`。

---

## 9. 一句话流程（按当前条件）

1. **Windows 已通**：OpenCode + `http://10.x.x.x:8080/v1` + API key，本机小仓库验证过 tool call。
2. SSH 进小网 Linux。Linux **`curl` 公司网关 IP 失败是正常的**。
3. Windows 另开窗口：`ssh -N -R 127.0.0.1:8080:10.x.x.x:8080 xiaowang`，目这条。
4. Linux 上 `curl http://127.0.0.1:8080/v1/models` 成功后，装 OpenCode，`baseURL` 用 `http://127.0.0.1:8080/v1`，tmux 里跑。
5. 不要域名、不要 DNS、不要 nginx、不要系统代理、不要 `GatewayPorts yes`。隧道只跟你这条 SSH。

---

## 附录：例外——某台 Linux 以后能直连公司网关 IP

只有在那台机器上确认：

```bash
curl -sS http://10.x.x.x:8080/v1/models \\
  -H "Authorization: Bearer $KEY"
```

**成功** 时，才可以：不建 `ssh -R`，Linux OpenCode 的 `baseURL` 用同一个 `http://10.x.x.x:8080/v1`（与 Windows 同一份）。当前环境不要用这条。
