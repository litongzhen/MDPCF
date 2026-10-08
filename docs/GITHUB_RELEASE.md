# GitHub 发布清单

## 作者需要准备

1. GitHub 账号与新仓库。建议先建 private 仓库审查，名称可用 MDPCE 或 MDPCF；创建空仓库，不预生成 README/license，避免首次 push 合并冲突。
2. 选择第一方代码许可证，并检查第三方代码许可证/归属。尚未自动添加 LICENSE，避免替作者决定授权。
3. 确认是否公开 `Mathematics/` 论文源码、作者信息和模板文件；数据与模型默认不公开。
4. 确认各数据集获取方式。当前下载脚本仅自动获取 Adult；其他数据必须按元数据配置准备，不能宣称五个数据集都一键下载。
5. 在独立 Python 3.10 环境验证安装和最小流程。当前环境导入检查不等于全新安装验证。

## 已整理的文件

- `.gitignore`：环境、凭据、数据、模型、日志、结果和历史调参输出不提交。
- `requirements.txt`、`requirements-baselines.txt`、`requirements-dev.txt`：主方法、可选 baseline 和测试依赖。
- `scripts/setup_environment.ps1`：在用户选定环境安装，不创建/重置环境。
- `scripts/check_environment.py`：导入检查，不启动实验。
- `release/environment-reference.json`：实际实验环境版本快照，不含本地路径。
- `release/summaries/` 和 `release/configs/`：正式结果和去除本地输出路径的配置，不包含探索结果。
- README：安装、数据、运行、恢复和正式协议说明。

`.venv` 目录不上传。依赖固定仍不保证跨硬件逐位一致；CUDA 驱动、数据划分、黑盒 checkpoint、随机种子与生成配置都影响复现。

## 提交前检查

不要现在直接 `git add .`：工作区包含大量历史修改、删除和第三方源码，必须查看暂存内容。
`.gitignore` 只约束未跟踪文件，不会自动移除已经提交的敏感文件或清理 Git 历史。
发布前应扫描当前暂存区与可达历史；扫描结果仅覆盖定义的规则，不构成绝对安全保证。
`scripts/audit_git_release.py` 检查常见凭据模式、禁止发布的文件以及大于等于 100 MB 的对象。
本版是仅代码发布，论文草稿和授权未确认的 DiCoFlex 上游源码不打包。

```powershell
git status --short
git ls-files
git diff --stat
git diff --cached --stat
git diff --cached
```

确认必要的 runtime 源码均在暂存区，包括 `baselines/dice/DiCE-official`，保留许可证。
DiCoFlex 上游源码另行获取，获取方式见 README；不要提交根目录的重复下载或其中附带的数据。
大模型或原始数据另行放到 Release/独立归档服务，只有具备分享权限时才发布。
如有凭据曾经提交，必须先撤销/轮换凭据并处理历史，不能只添加忽略规则。

## 创建仓库之后

用 GitHub Desktop 登录并发布，或通过 Git 命令推送。HTTPS 登录使用 Git Credential Manager 或 GitHub 的 token 登录流程，不把 token 写进远程 URL、脚本或 README。
审阅并提交选定文件后，添加新仓库地址：

```powershell
git remote add origin git@github.com:litongzhen/MDPCF.git
git push -u origin HEAD
```

如果已经有 origin，不要盲目覆盖，先 `git remote -v`。完成后创建版本 tag/release；正式论文引用确定的版本或归档 DOI，而不是只引用不断变化的分支。
把创建后的仓库地址给 Codex，即可继续检查提交范围、完善论文中的代码链接；不能在仓库尚不存在时填造链接。
