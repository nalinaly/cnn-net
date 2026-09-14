先上 OpenCode，不要先上 Codex。 公司 Windows + 自有大模型 API key + 以后还要装到小网 Linux，这三件事叠在一起，OpenCode 更合适。

Codex 更强的前提是：官方 ChatGPT / OpenAI，或网关完整兼容它的 Responses / WebSocket 路径。公司 key 通常不是这条路。

───

为什么不是 Codex

Codex 的默认世界是 ChatGPT 登录和 OpenAI Responses API。接到公司网关时常见坑：

• 只提供 /v1/chat/completions，Codex 却走 /v1/responses 或 WebSocket
• OPENAI_BASE_URL 在部分版本会被忽略，必须写 ~/.codex/config.toml 的 model_providers
• 自定义网关有过 HTTPS fallback 丢 Authorization 的问题
• Windows 上官方更推荐 Win11 原生 App 或 WSL；公司机常常两样都不好装

只有在这些成立时才选 Codex：key 是 ChatGPT Enterprise / 官方 OpenAI，或网关明确宣称兼容 Codex，并且 IT 允许装官方 Windows App。

───

为什么选 OpenCode

┌─────────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
│ 你的约束                │ OpenCode                                                                      │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ 公司 API key            │ 一等公民：/connect 或 opencode.json 里用 @ai-sdk/openai-compatible 填 baseURL │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Windows 先落地          │ Scoop / Chocolatey / npm / 桌面安装包都能装；Scoop 通常不需要管理员           │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ 以后小网 Linux 再装一份 │ curl -fsSL https://opencode.ai/install | bash，同一份配置拷过去               │
├─────────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ 公司合规                │ MIT 开源，key 只存在本机 auth.json，不绑 OpenAI 账号                          │
└─────────────────────────┴───────────────────────────────────────────────────────────────────────────────┘

官方文档仍建议 Windows 用 WSL 体验最好。公司机若不准开 WSL，就走 Scoop 原生 CLI，终端用 Windows Terminal。

───

先分清网络，再决定 agent 放哪

Windows 能访问公司 LLM API，小网 Linux 只有内网 IP。真正决定架构的是第二台机器能不能打到那个 API。

情况 A：小网 Linux 也能访问公司 API（内网网关）
Windows 只做跳板。SSH 上去在 Linux 里跑 agent。GPU / 代码 / 文件都在那台机器上，agent 才有完整能力。Windows 上的 OpenCode 用来练手、改文档、验证 key。

情况 B：小网 Linux 出不去，只有 Windows 能调 API
agent 必须留在 Windows。对 Linux 只能 SSH 下命令、scp 文件。Linux 上再装一份 agent 会变成“能改文件但调不到模型”，除非你从 Windows 做反向代理 / SSH 隧道把 API 转进去。

多数公司内网网关是情况 A。先用 Windows 测通 API，再 SSH 到 Linux 测 curl 那个 baseURL。通了就把 OpenCode 装到 Linux，Windows 只当入口。

───

Windows 上最小落地

公司机优先 Scoop（当前用户、常免管理员）：

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop install git
scoop bucket add extras
scoop install extras/opencode

若已有 Node：

npm i -g opencode-ai

连公司网关（把 URL / 模型名换成你们的）：

{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
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

文件放 %USERPROFILE%\.config\opencode\opencode.json。启动后 /connect 贴 key，/models 选模型。先让它在 Windows 本地改一个小仓库，确认 tool call + 流式 都通，再谈远程 Linux。

公司 HTTPS 中间人证书经常把 TLS 弄挂。把公司根证加进系统信任，或给进程设 SSL_CERT_FILE / NODE_EXTRA_CA_CERTS。Codex 有 CODEX_CA_CERTIFICATE，OpenCode 走 Node/系统证书，证书不对时表现为连不上而不是“模型笨”。

连小网 Linux：

ssh user@10.x.x.x

能登录之后，不要让 Windows 上的 agent 隔空改 Linux 工程。长期应在 Linux 上跑同一份 OpenCode，Windows 只 SSH + tmux。

───

小网 Linux 第二步

确认 curl 公司 API 成功后：

curl -fsSL https://opencode.ai/install | bash

把 Windows 上的 opencode.json 拷到 ~/.config/opencode/，用 /connect 再录一次 key（不要用 U 盘明文到处拷）。在项目目录开 tmux，里面跑 opencode。之后你从公司 Windows ssh 进去就能接着干。

───

什么时候回头选 Codex

同时满足再换：

1. 公司给的是 ChatGPT / OpenAI 官方额度，或网关文档写明兼容 Codex responses
2. Windows 11，IT 允许官方 Codex App 或 WSL
3. 你更在意 Codex 在 GPT 系列上的编码能力，而不是“任意公司 key 都能接”

否则 Codex 会在 Windows 安装和自定义网关上先耗掉一周。

───

一句话： 公司 Windows 先装 OpenCode 接现有 API key；用它 SSH 到小网 Linux。Linux 也能打到 API 就把同一套 OpenCode 装到 Linux 上当主力。Codex 留给官方 OpenAI 环境。
