# 航班运行数据智能分析与延误预测系统 V1.0

FlightDelayDecisionSystem 是使用 Python 和 PySide6 开发的本地桌面软件，用于航班 CSV 管理、数据质量分析、历史运行统计、特征检查和模型训练管理。

## 当前功能

- 导入、字段映射与分页预览：识别 BTS 常见字段，检查来源文件变化。
- 数据质量分析：缺失值、重复记录、日期时间、取消状态与延误标签检查，导出 JSON 质量报告。
- 历史运行统计：航司、机场、航线等实体统计，以及严格使用目标日期之前数据的历史特征查询。
- 特征工程：A/T/H/C 特征组合、字段来源检查和泄漏审计。
- 模型工程：Random Forest / XGBoost 训练、时间划分、指标查看、模型登记、加载和归档，以及内部推理接口。

当前版本保留 15 项导航，部分页面仍为占位页。单航班与批量预测的完整 GUI、预测记录、SHAP、阈值优化和完整导出工作流尚未开放。内部推理接口不等于这些页面已经实现。DEMO 或 REAL_SMOKE 验证不能视作论文最终结果或部署效果。

## Windows 安装与启动

使用 Python 3.11，在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

依赖版本沿用本地已验证环境；本次发布准备没有重新验证从网络安装依赖。`requirements-ml.txt` 为后续可选依赖清单，当前运行以 `requirements.txt` 为准。

首次启动会创建本地数据库、缓存、日志和输出目录。仓库不携带已有数据库、历史缓存或训练模型，因此初始数据集列表和模型列表为空，需要自行导入及训练。

## 使用合成示例数据

1. 在“航班数据管理”中导入 `data/demo/demo_bts_flights.csv`，查看预览与质量分析。
2. 导入 `data/demo/demo_historical_flights.csv`，在“历史运行统计”中构建缓存并查询统计。
3. `data/demo/demo_schedule_flights.csv` 提供合成计划表，供需要计划上下文的特征流程使用。
4. 进入特征工程或模型训练页面，按实际字段、历史缓存及日期划分完成检查和训练。合成数据只用于验证操作流程。

示例文件均为软件测试用合成数据，不是真实 BTS 观测数据。`tests/fixtures/` 也是测试样例。真实数据需使用者自行准备，不自动下载。

## 项目结构

```text
app/             桌面窗口、页面、控件和后台任务
core/            数据、历史统计、特征及机器学习逻辑
config/          可共享的默认配置
database/        数据库管理源码（不含数据库文件）
data/demo/       三份合成示例 CSV
models/          模型资产说明（不含模型包）
resources/       界面样式
scripts/         数据生成、验证及开发工具
tests/           自动化测试与测试样例
main.py          启动入口
```

`scripts/` 保留开发验证工具；历史报告生成器与部分 GUI 验证脚本依赖本地训练资产或历史报告，不能把它们视为首次安装步骤。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

## 本地文件与模型

航班 CSV、数据库、缓存和日志在本地处理。导入默认登记源文件路径；导出报告可能包含本机路径，分享前需检查。`.gitignore` 排除真实数据、用户配置、数据库、缓存、模型包、日志、全部运行输出及软著申请材料。

只加载自己生成或来源可信的模型资产。模型注册及哈希检查不构成安全签名，不能保证不可信 joblib/pickle 文件安全。

## 许可状态

当前版本暂不添加开源许可证，保留现有 `LICENSE_NOTICE.txt` 权属说明。第三方依赖按各自许可证使用。
