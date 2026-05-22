import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from src.genomics_pipeline import summarize_selected_snps, train_finetune, vcf_to_encoded_sequences

st.set_page_config(page_title="家禽关键遗传位点筛选与育种值预测", page_icon="🐔", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
[data-testid="stMetricValue"] {font-size: 1.4rem;}
</style>
""", unsafe_allow_html=True)

st.title("🐔 家禽关键遗传位点筛选与育种值预测系统")
st.caption("软著版可视化应用：VCF预处理 → 深度学习训练 → 关键位点/预测结果展示与导出")

with st.sidebar:
    st.header("⚙️ 参数设置")
    min_maf = st.slider("最小 MAF", 0.0, 0.5, 0.05, 0.01)
    max_missing = st.slider("最大缺失率", 0.0, 0.5, 0.10, 0.01)
    epochs = st.slider("训练轮次", 1, 100, 10)
    batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
    lr = st.select_slider("学习率", options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3], value=1e-3)
    hidden_size = st.selectbox("隐藏层维度", [64, 128, 256], index=1)
    num_layers = st.selectbox("Transformer 层数", [1, 2, 3, 4], index=1)
    nhead = st.selectbox("注意力头数", [2, 4, 8], index=1)
    max_seq_len = st.slider("最大序列长度", 64, 1024, 256, 64)

c1, c2 = st.columns(2)
with c1:
    vcf_file = st.file_uploader("上传VCF文件", type=["vcf", "gz"])
with c2:
    pheno_file = st.file_uploader("上传表型文件（最后一列为表型值）", type=["txt", "csv", "tsv"])

run_btn = st.button("🚀 开始训练并分析", type="primary", use_container_width=True)

if run_btn:
    if not vcf_file or not pheno_file:
        st.error("请先同时上传 VCF 与表型文件。")
        st.stop()

    with tempfile.TemporaryDirectory() as td:
        vcf_path = os.path.join(td, vcf_file.name)
        pheno_path = os.path.join(td, pheno_file.name)
        with open(vcf_path, "wb") as f:
            f.write(vcf_file.read())
        with open(pheno_path, "wb") as f:
            f.write(pheno_file.read())

        out_dir = os.path.join(td, "outputs")
        seq_file = os.path.join(out_dir, "encoded_sequences.txt")

        with st.status("任务执行中...", expanded=True) as status:
            st.write("Step1/2：VCF预处理与编码")
            pre = vcf_to_encoded_sequences(vcf_path, seq_file, min_maf=min_maf, max_missing=max_missing)
            st.write("Step2/2：模型训练与预测")
            train_res = train_finetune(
                pre["sequence_file"], pheno_path, out_dir,
                epochs=epochs, batch_size=batch_size, lr=lr,
                hidden_size=hidden_size, num_layers=num_layers, nhead=nhead,
                max_seq_len=max_seq_len,
            )
            status.update(label="执行完成", state="complete")

        st.subheader("📊 处理统计")
        s = pre["stats"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("扫描记录", s["records_scanned"])
        m2.metric("保留SNP", s["retained_snps"])
        m3.metric("缺失率过滤", s["filtered_missing"])
        m4.metric("MAF过滤", s["filtered_maf"])

        history_df = pd.DataFrame(train_res["history"])
        st.subheader("📈 训练过程")
        st.plotly_chart(px.line(history_df, x="epoch", y=["train_loss", "corr", "rmse"], markers=True), use_container_width=True)

        if os.path.exists(train_res["pred_file"]):
            pred_df = pd.read_csv(train_res["pred_file"])
            st.subheader("🎯 验证集预测散点图")
            fig = px.scatter(pred_df, x="y_true", y="y_pred", trendline="ols", title="真实值 vs 预测值")
            st.plotly_chart(fig, use_container_width=True)

        if os.path.exists(train_res["snp_file"]):
            top_df = summarize_selected_snps(train_res["snp_file"], top_n=30)
            st.subheader("🧬 Top30 关键遗传位点")
            st.dataframe(top_df, use_container_width=True)
            st.plotly_chart(px.bar(top_df, x="snp_pos", y="freq", title="关键位点选择频次"), use_container_width=True)

        st.subheader("⬇️ 结果下载")
        st.download_button("下载训练历史CSV", open(train_res["history_file"], "rb").read(), file_name="train_history.csv")
        st.download_button("下载预测结果CSV", open(train_res["pred_file"], "rb").read(), file_name="val_predictions.csv")
        st.download_button("下载编码序列TXT", open(pre["sequence_file"], "rb").read(), file_name="encoded_sequences.txt")

        st.success(f"完成！最佳验证相关系数: {train_res['best_corr']:.4f}")
