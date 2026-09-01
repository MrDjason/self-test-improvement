# Git 与 GitHub 协作完全指南（含 SSH 端口故障排查）

## 一、准备工作
- 安装 [Git](https://git-scm.com/downloads)。
- 注册 [GitHub](https://github.com) 账号。
- （可选）安装 GitHub Desktop 等图形客户端，但本笔记以命令行操作为主。

---

## 二、本地仓库初始化与首次提交

```bash
# 进入你的项目根目录
cd /path/to/your/project

# 初始化本地 Git 仓库
git init

# 添加所有文件到暂存区（. 表示当前目录所有文件）
git add .

# 提交，创建第一个本地提交（-m 后跟提交说明）
git commit -m "首次提交"

# 将默认分支名改为 main（与 GitHub 统一）
git branch -M main
```

> **注意**：提交之前可以用 `git status` 查看状态。

---

## 三、连接 GitHub 远程仓库

### 3.1 在 GitHub 上创建空仓库
- 登录 GitHub，点击右上角 “+” → “New repository”。
- **不要勾选** “Initialize this repository with a README” 等选项（保持完全空）。
- 复制仓库地址，例如：
  - SSH 地址：`git@github.com:用户名/仓库名.git`
  - HTTPS 地址：`https://github.com/用户名/仓库名.git`

### 3.2 添加远程仓库（任选一种协议）

#### 方式 A：使用 SSH（推荐，免密推送）
1. **bash 生成 SSH 密钥**（如果没有）：
   ```bash
   ssh-keygen -t ed25519 -C "你的邮箱@example.com"
   ```
   一路回车（可设密码，也可留空）。

2. **bash 复制公钥**：
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   复制输出的全部内容（以 `ssh-ed25519` 开头）。

3. **添加到 GitHub**：
   - 点击右上角头像 → **Settings** → **SSH and GPG keys** → **New SSH Key**。
   - Title 任意，**Key type 选择 `Authentication Key`**（认证密钥），粘贴公钥，保存。

4. **测试连接**（默认走 22 端口）：
   ```bash
   ssh -T git@github.com
   ```
   若显示 `Hi 用户名! You've successfully authenticated...` 则成功。

5. **添加远程地址**：
   ```bash
   git remote add origin git@github.com:用户名/仓库名.git
   ```

#### 方式 B：使用 HTTPS（简单但需输密码/Token）
```bash
git remote add origin https://github.com/用户名/仓库名.git
```
推送时会提示输入用户名和密码（密码需用 **Personal Access Token**，生成方法见下文“常见问题”）。

---

## 四、首次推送并建立跟踪关系

```bash
git push -u origin main
```
- **`-u`**（`--set-upstream`）会建立本地 `main` 分支与远程 `origin/main` 的跟踪关系，之后只需 `git push` 即可。

---

## 五、日常更新流程

```bash
# 修改代码后
git add .
git commit -m "本次修改说明"
git push    # 已建立跟踪，无需再写 origin main
```

如果要拉取远程最新代码：
```bash
git pull
```

---

## 六、常见网络问题：SSH 端口 22 被屏蔽

很多公司、校园网络或 ISP 会封锁 SSH 默认的 **22 端口**，导致连接失败（报错 `Connection refused`）。解决方案有两种：

### 方案 1：修改远程地址，强制使用 443 端口（推荐，最稳）
```bash
git remote set-url origin ssh://git@ssh.github.com:443/用户名/仓库名.git
git push -u origin main
```
这样直接指定了主机、端口，不依赖任何配置文件。

### 方案 2：配置 `~/.ssh/config` 文件，让 SSH 自动走 443 端口
在用户目录下的 `.ssh` 文件夹（Windows：`C:\Users\你的用户名\.ssh`）创建 `config` 文件（无扩展名），内容为：
```
Host github.com
    Hostname ssh.github.com
    User git
    Port 443
```
保存后，`git push` 会自动走 443 端口，无需修改远程 URL。

> **⚠️ Windows 注意事项**：
> - 不要用记事本默认保存（会加 BOM），推荐用 VS Code 或 Notepad++ 保存为 **UTF-8 无 BOM** 格式。
> - 或者在 PowerShell 中用命令生成无 BOM 文件：
>   ```bash
>   cd ~/.ssh
>   @"
>   Host github.com
>       Hostname ssh.github.com
>       User git
>       Port 443
>   "@ | Out-File -FilePath config -Encoding ascii
>   ```
>   使用 `ascii` 编码可完全避免 BOM 问题。

如果配置文件出现 `Bad configuration option` 报错，通常就是 BOM 或格式错误，直接删掉 config 文件，改用方案 1 即可。

---

## 七、其他常见故障排除

| 错误信息 | 原因 | 解决办法 |
|--------|------|--------|
| `Permission denied (publickey)` | SSH 公钥未添加到 GitHub，或密钥不匹配 | 检查 `~/.ssh/id_ed25519.pub` 内容是否完整复制，重新添加；或重新生成密钥。 |
| `src refspec main does not match any` | 本地还没有任何 commit | 先执行 `git add .` 和 `git commit -m "..."`。 |
| `Connection refused (port 22)` | 22 端口被屏蔽 | 改用 443 端口（见第六节）。 |
| 推送时提示输入密码但登录密码无效 | GitHub 已禁用密码登录 | 使用 Personal Access Token。生成方法：GitHub Settings → Developer settings → Personal access tokens → Generate new token，勾选 `repo` 权限，复制 token 作为密码。 |
| `fatal: remote origin already exists` | 已添加过远程地址 | 用 `git remote set-url origin 新地址` 修改，或用 `git remote rm origin` 删除后重新添加。 |

---

## 八、推荐 `.gitignore` 文件（避免上传无用文件）

在项目根目录创建 `.gitignore`，根据你的项目类型添加忽略规则。  
Godot 示例（常见忽略项）：
```
.godot/
export/
*.import
*.godot
*.tmp
.DS_Store
```
GitHub 官方提供各类语言的模板，可参考 [gitignore](https://github.com/github/gitignore)。

---

## 九、总结

| 阶段 | 关键命令 |
|------|--------|
| 初始化 | `git init` |
| 添加 | `git add .` |
| 提交 | `git commit -m "..."` |
| 改分支名 | `git branch -M main` |
| 添加远程 | `git remote add origin <地址>` |
| 首次推送 | `git push -u origin main` |
| 后续推送 | `git push` |
| 拉取更新 | `git pull` |

遇到 SSH 端口问题，优先使用 `ssh://git@ssh.github.com:443/用户名/仓库名.git` 格式的远程地址，可以绕过所有配置文件干扰。