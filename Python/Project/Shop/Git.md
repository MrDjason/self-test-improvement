# Git

## 工作区
- 对于`添加`、`修改`、`删除`文件的操作，都发生在工作区中

## 暂存区
- 暂存区指将工作区中的操作完成小阶段的存储，是版本库中的一部分

## 仓库区
- 仓库区表示个人开发的一个小阶段的完成
  - 仓库区中记录的个版本是可以查看并回退的
  - 但是在暂存区中的版本一旦提交就再也没有了

# Git单人本地仓库操作

- 1.安装Git
```
sudo apt-get install git
```

- 2.查看git安装结果
```
git
```

- 3.创建项目
```
Desktop/test/
```

- 4.创建本地仓库
  - a.进入到`test`，并创建本地仓库`.git`
  - b.新创建的本地仓中`.git`是个空仓
```
cd Desktop/test/
git init 
```

- 5.初始化配置
```
git config user.name xxx
git config user.email xxx@xx.com
```

- 6.创建一个初始文件
```
touch xxx.py
```

- 7.查看工作区域
  - 红色的文件名一般在工作区
```
git status
```

- 8.将工作区的文件提交到暂存区
  - 绿色文件名一般在暂存区
```
git add xxx.py
# 提交当前目录下所有文件
git add .
```

- 9.将暂存区的文件提交到仓库区
```
git commit -m '注释信息'  # 仅提交已经在暂存区的文件
git commit -am '注释信息' # 一步完成暂存+提交
```

- 10.检查信息
  - 使用`git status`。此时文件，已被提交至仓库
```
git status
```
```
位于分支 master
无文件要提交，干净的工作区
```

# Git的日志以及版本管理
- 11.查看历史版本
```
git log    # 看提交历史
git reflog # 看操作日志（本地仓库的所有 HEAD 变动，包括误删 / 回滚的提交）
```

```
commit eda0276ce5cdeb5c007ac1b2c7e6c5421d7dbcc5 (HEAD -> master)
Author: MrDjason <13218798893@163.com>
Date:   Wed Dec 10 16:25:40 2025 +0800

    注释
```
- 注释  
  - `commit eda0276ce5cdeb5c007ac1b2c7e6c5421d7dbcc5`：版本号
  - `Author: MrDjason <13218798893@163.com>`：谁提交的代码
  - `Date:   Wed Dec 10 16:25:40 2025 +0800`：提交代码时间
  - `注释`：版本描述

- 12.回退版本  
使用`git reflog`显示
```
39c8420 (HEAD -> master) HEAD@{0}: commit: 版本2
eda0276 HEAD@{1}: commit (initial): 版本1
```
```
git reset --hard HEAD^ # 回到当前最新版本前一个版本
```
  - 方案一：
    - `HEAD`表示当前最新的版本
    - `HEAD^`表示当前最新版本的前一个版本
    - `HEAD^^`表示当前最新版本的前两个版本，以此类推...
    - `HEAD~1`表示当前最新版本的前一个版本
    - `HEAD~10`表示当前最新版本的前10个版本，以此类推...
  - 方案二：
    - 通过每个版本的版本号回退到指定版本
```
git reset --hard 版本号
git reset --hard eda0276 # 回到版本1
```

- 13.撤销操作
  - 撤销暂存区代码
```
git reset HEAD 文件名 # 把已 git add 到暂存区的修改撤回至工作区
git checkout 文件名 # 用暂存区 / 版本库的文件覆盖工作区的修改
```

# Git远程仓库
- 复制仓库 
```
git clone 仓库名 
# 同时下载到本地已经有.git文件，不需要执行git init
# 必须到相关目录下执行git命令
```
- 关联远程仓库
```
git remote add origin https://github.com/名字/仓库.git
git remote -v # 验证关联成功
```

- 推送到远程仓库
```
git push # 将仓库区全部存储到Github
git push -u origin main 
# 将本地main分支的所有代码上传到远程origin仓库的main分支，同时让Git记住
```
- `-u`：-u是--set-upstream的缩写，核心作用是建立本地ubuntu分支↔远程 origin/ubuntu分支的追踪关系
- `origin`：Git自动把冗长的仓库URL简化成了origin这个短名字

- 强制拉取远程代码覆盖本地
- 先拉取远程最新代码到本地仓库
```
git fetch origin
```

- 强制将本地代码重置至远程main分支
```
git reset --hard origin/main
```