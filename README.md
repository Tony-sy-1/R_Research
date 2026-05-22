# 家禽关键遗传位点筛选与育种值预测系统

## 功能
- VCF 数据解析、MAF/缺失率过滤、LD 标签编码。
- 基于深度学习的 SNP 重要性评分与育种值回归预测。
- Streamlit 可视化应用（训练曲线、关键位点 Top 排名、交互参数配置）。
- Windows 桌面版可执行程序（双击运行，无需命令行）。

## 快速启动（源码方式）
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Windows 打包为桌面版 EXE（双击可运行）
1. 安装 Python 3.10+（勾选 Add to PATH）。
2. 在项目根目录双击运行：`scripts/build_windows.bat`。
3. 打包成功后，在 `dist/PoultryBreedingDesktop/` 目录得到可执行程序 `PoultryBreedingDesktop.exe`。
4. 双击 `PoultryBreedingDesktop.exe` 即可启动系统（自动拉起本地服务并打开界面）。

> 打包依赖见 `requirements-desktop.txt`，采用 PyInstaller 规范文件 `build_windows_exe.spec`。

## 输入文件说明
- `VCF`：标准变异文件。
- `表型文件`：最后一列为目标表型值（可含样本ID等前置列）。

## 输出
- `encoded_sequences.txt`
- `encoded_sequences_sample_ids.txt`
- `encoded_sequences_stats.csv`
- `best_finetuned_model.pt`
- `train_history.csv`
- `val_predictions.csv`
- `selected_top_positions.npy`

## 桌面版运行机制说明
- 桌面启动器入口：`desktop_launcher.py`。
- 启动器会自动：
  - 查找空闲本地端口。
  - 启动内置 Streamlit 服务。
  - 自动打开浏览器访问系统页面。
  - 关闭程序时自动结束后台服务。
