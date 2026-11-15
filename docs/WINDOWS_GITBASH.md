# Windows + Git Bash：从项目包到 GitHub 和本地训练

以下命令均在**项目根目录**的 Git Bash 运行。这个目录应含 `README.md`、`pyproject.toml`、`src/`；不要在 ZIP 文件内部运行命令。

## 1. 把项目文件放入你的 GitHub 仓库

先在 Windows 安装 Python 3.12 和 Git for Windows（已有则跳过）。从本次交付下载 `Oxygen-Adsorption-ML-Starter.zip`，用文件资源管理器解压。

你的 GitHub 仓库已有 README。找一个本地工作目录，在那里右键选择 **Open Git Bash Here**，执行：

```bash
git clone https://github.com/Bowen-Bob-Zhang/Oxygen-Adsorption-Energies-on-Bimetallic-Alloy-Catalysts.git
```

用文件资源管理器打开刚克隆的文件夹，把项目 ZIP 解压出的文件夹**里面的内容**复制进去；如果询问是否覆盖 README，选择覆盖。不要把外层项目文件夹整体作为一个子文件夹复制进去。随后在克隆文件夹里右键打开 Git Bash，检查：

```bash
git remote -v
git status --short
```

`git remote -v` 应显示上面的 GitHub 地址。`git status` 应列出新增或修改的项目代码。

## 2. 安装依赖并检查代码

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

如果本机只有 Python 3.11，第一行用 `py -3.11 -m venv .venv`。以后重新打开 Git Bash，先在项目目录执行 `source .venv/Scripts/activate`。安装 PyTorch 时如有问题，按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择 Windows、Pip、Python、CPU，执行其命令后重试依赖安装。

## 3. 把代码上传 GitHub

在仓库根目录运行：

```bash
git add README.md pyproject.toml requirements.txt .gitignore src tests examples docs
git diff --cached --stat
git status --short
git commit -m "Add oxygen adsorption ML pipeline"
git push -u origin HEAD
```

提交前查看暂存列表：应只有代码、说明、示例候选输入和测试；`data/raw/`、`data/processed/`、`artifacts/`、`.venv/`、proposal 文档及密钥不应出现在其中。首次推送可能要求在浏览器登录 GitHub；按 Git 的提示在**你的电脑**完成认证。推送后刷新仓库网页，确认 `src/` 和 `docs/` 已显示。

`git commit` 如果提示不知道姓名和邮箱，在本机设置后重新执行：

```bash
git config --global user.name "你的 GitHub 显示姓名"
git config --global user.email "你的 GitHub 提交邮箱"
```

## 4. 下载并整理实际数据

在 [Catalysis-Hub 登录页](https://api-catalysis-hub.slac.stanford.edu/auth/login)使用 ORCID 登录，获取自己的 API key。不要把 key 发到聊天里或提交到 GitHub。保持当前 Git Bash 窗口打开，在其中执行：

```bash
read -rs -p 'CATHUB_API_KEY: ' CATHUB_API_KEY; echo
export CATHUB_API_KEY
python -m oxygen_ml.data fetch --output data/raw/cathub_o.json
unset CATHUB_API_KEY
python -m oxygen_ml.data prepare --input data/raw/cathub_o.json --output data/processed/surfaces.csv
```

第一个命令会等待你粘贴 key 并按 Enter，输入时不会显示字符。若你已经有同一出版物的 CatHub JSON，可以跳过 `fetch`，对那个文件运行 `prepare`。检查 `data/processed/surfaces.summary.json` 的 `surface_labels` 是否大于零，以及过滤计数是否符合预期。

## 5. 训练并用于筛选

```bash
python -m oxygen_ml.train --data data/processed/surfaces.csv --output artifacts/run1
python -m oxygen_ml.predict --model-dir artifacts/run1 --candidates examples/candidates.csv --output artifacts/run1/predictions.csv
```

`artifacts/run1/metrics.json` 给出验证集和测试集误差；`model.pt`、`preprocessor.joblib` 和 `metadata.json` 是一起使用的模型文件。把你自己的未知材料写成 UTF-8 CSV，每行一个 `surface_formula`，可加 `facet`，再把预测命令中的 `--candidates` 改为你的 CSV 路径。查看 `predictions.csv` 的预测能量和范围提示，挑选候选做独立 DFT 验证。示例文件仅用于确认命令和输入格式。

真实数据、模型文件和预测结果默认留在本机；继续提交代码时先运行 `git status --short`，核对暂存内容。
