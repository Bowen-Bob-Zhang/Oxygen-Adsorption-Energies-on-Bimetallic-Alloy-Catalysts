# Oxygen adsorption energy prediction on alloy surfaces

这是 MASC 520 项目的第一版可运行架构。目标是预测**理想纯金属或双金属表面上，原子 O 在有效位点中的最低吸附能**（eV），用于初步筛选后续值得做 DFT 的材料。这里的能量采用原数据集的参考反应：

`H2O(g) - H2(g) + * -> O*`

一条训练记录代表一个表面，而不是一次位点计算。输入只包括计算 O 吸附之前已知的合金成分、理想结构、晶面，以及组成元素的原子序数、族和周期。模型不会使用弛豫后的 O 位点或目标能量作为输入。

## 需要的资源

| 项目 | 第一版所需 |
| --- | --- |
| 电脑 | Windows/macOS/Linux 均可；普通笔记本 CPU、约 8 GB RAM 足以开始；无需显卡 |
| 软件 | Python 3.11 或 3.12、Git、Git Bash（Windows 可用）、本仓库的 Python 依赖 |
| 数据 | Catalysis-Hub 的 `MamunHighT2019`，通过在线接口取得已解析的反应能；无需下载约 25 GiB 的原始 DFT 输出 |
| 访问凭证 | 从 [Catalysis-Hub 的 ORCID 登录页](https://api-catalysis-hub.slac.stanford.edu/auth/login)获得 `CATHUB_API_KEY`，只存在本机环境变量中 |
| 训练时间 | MLP 很小，设计为 CPU 训练；实际耗时取决于电脑与获取的数据量 |

数据来源：Mamun et al., *Scientific Data* **6**, 76 (2019), [DOI](https://doi.org/10.1038/s41597-019-0080-z)；[Catalysis-Hub 数据页](https://www.catalysis-hub.org/publications/MamunHighT2019)。数据库论文：Winther et al., *Scientific Data* **6**, 75 (2019), [DOI](https://doi.org/10.1038/s41597-019-0081-y)。

## 1. 安装

在项目根目录打开 Git Bash。Windows 示例：

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你装的是 Python 3.11，把 `-3.12` 改成 `-3.11`。macOS/Linux 则使用 `python3 -m venv .venv` 和 `source .venv/bin/activate`。先从 CPU 开始；如果以后需要 NVIDIA GPU，再按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择对应的 CUDA 安装命令。

## 2. 下载并整理数据

先在 [Catalysis-Hub 登录页](https://api-catalysis-hub.slac.stanford.edu/auth/login)用 ORCID 登录，取得 API key。Git Bash 中可通过下面的无回显提示输入密钥，不要把密钥写在 Python 文件、`.env` 文件或 GitHub 提交里：

```bash
read -rs -p 'CATHUB_API_KEY: ' CATHUB_API_KEY; echo
export CATHUB_API_KEY
python -m oxygen_ml.data fetch --output data/raw/cathub_o.json
python -m oxygen_ml.data prepare --input data/raw/cathub_o.json --output data/processed/surfaces.csv
```

下载器按 200 条一页遍历，页面之间默认暂停 6.5 秒，因此下载可能需要几分钟；整理器再次本地核对 O 产物、参考反应、覆盖度和有限的能量值，然后将同一表面的有效位点取最低值。运行后检查控制台输出的原始条数、保留条数和表面数。每次整理会同时保存处理摘要，便于追溯筛选规则。已有 CatHub JSON 时，可以跳过 `fetch`，把文件路径传给 `prepare`。不要把原始数据或私人 API key 提交到 GitHub。

## 3. 训练和比较

```bash
python -m oxygen_ml.train --data data/processed/surfaces.csv --output artifacts/run1
```

训练会按**无序金属元素对**分开训练、验证和测试。数值缺失填补、标准化、类别编码及目标能量标准化都只从训练部分学习。模型为小型 PyTorch MLP：`输入 → 64 → 32 → 1`，ReLU、dropout 0.1、AdamW、验证集早停。相同划分上还训练 Ridge、随机森林和直方图梯度提升作为比较。输出：

- `artifacts/run1/model.pt`：训练好的神经网络权重；
- `artifacts/run1/preprocessor.joblib`：输入特征的处理步骤；
- `artifacts/run1/metadata.json`：特征顺序、目标标准化参数、已见元素等；
- `artifacts/run1/metrics.json`：四种模型在验证/测试集上的 MAE、RMSE（eV）和 R²；
- `artifacts/run1/splits.csv`：每条表面数据属于哪个划分。

`model.pt` 必须和同一次训练保存的 `preprocessor.joblib`、`metadata.json` 一起使用。预测命令始终使用 MLP；控制台显示的“验证集误差最低模型”只是四种方法的对照结果。先看测试集 MAE，再判断这个神经网络是否足以辅助 DFT 筛选；在这种样本规模下，简单基线可能优于神经网络。

## 4. 对候选材料做预测

准备 CSV，至少一列 `surface_formula`（例如 `Pt3Ni`、`PtNi`、`Pt`），可加 `facet`；省略晶面时按理想结构推断。示例见 `examples/candidates.csv`。

```bash
python -m oxygen_ml.predict --model-dir artifacts/run1 --candidates examples/candidates.csv --output artifacts/run1/predictions.csv
```

输出是每个表面的预测 O 吸附能（eV）、是否超出训练时见过的元素/结构/晶面，以及候选表面是否已在原数据集中有标签。示例 CSV 只用于演示输入格式，真正筛选时请提供未计算过的候选。若已根据**具体反应和参考材料**确定希望接近的目标能量，可附加 `--target-ev <数值>`，按与该数值的距离排序。吸附能不是催化活性的直接预测值；候选材料还需用 DFT 检查，尤其是超出训练范围的材料。

## 5. 本地检查与 GitHub

Windows 上从下载项目包、复制到已有仓库、安装环境、训练到推送代码的逐步说明见 [Git Bash 操作清单](docs/WINDOWS_GITBASH.md)。

```bash
python -m unittest discover -s tests -v
```

已有 GitHub 仓库中若已经有 README 提交，先在电脑上 `git clone <仓库 HTTPS 地址>`，再把这个项目包的文件复制到**克隆得到的仓库文件夹**。在该文件夹运行 `git status`，确认 `.gitignore` 生效，然后按需 `git add README.md pyproject.toml requirements.txt .gitignore src tests examples`、`git commit -m "Add oxygen adsorption ML pipeline"`、`git push`。不要把含学号的 proposal、API key、原始数据库或训练产物传到公开仓库。

## 现在的状态和下一步

这是一套项目代码与示例候选输入，**不含真实数据和训练好的权重**。取得 API key 后，依次运行下载、整理、训练、预测四步，检查真实的过滤计数和验证误差。下一阶段再根据学习曲线决定是否增加更有物理意义的元素描述符或位点级模型。
