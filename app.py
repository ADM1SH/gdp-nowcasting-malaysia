import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.nowcaster import run_gdp_nowcast_model

st.set_page_config(page_title="Malaysia GDP Nowcasting", layout="wide")
st.title("GDP NOWCASTING MODEL")
st.caption("Nowcasting from Services Index (3.2) & IPI (3.3) vs MOF Official 4.2% - 5.2% Range")

c1, c2 = st.columns(2)
with c1:
    s_in = st.slider("2027 Index of Services Growth (%)", 2.0, 8.0, 5.2, 0.1)
with c2:
    i_in = st.slider("2027 Industrial Production Index Growth (%)", 1.0, 8.0, 4.3, 0.1)

results = run_gdp_nowcast_model(s_in, i_in)
df_res = pd.DataFrame([r.model_dump() for r in results])
r2027 = results[-1]

k1, k2, k3 = st.columns(3)
k1.metric("2027 GDP Nowcast", f"{r2027.nowcast_point_pct:.2f}%")
k2.metric("95% Confidence Band", f"{r2027.ci_lower_pct:.2f}% to {r2027.ci_upper_pct:.2f}%")
k3.metric("MOF 2027 Official Target (4.2% - 5.2%)", "WITHIN RANGE" if r2027.within_official_range else "OUTSIDE RANGE")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_res["year"], y=df_res["ci_upper_pct"], mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df_res["year"], y=df_res["ci_lower_pct"], mode="lines", fill="tonexty", fillcolor="rgba(29, 78, 216, 0.15)", line=dict(width=0), name="95% CI"))
fig.add_trace(go.Scatter(x=df_res["year"], y=df_res["nowcast_point_pct"], mode="lines+markers", name="Nowcast (%)", line=dict(color="#D92B2B", width=3)))
fig.add_shape(type="rect", x0=2026.7, x1=2027.3, y0=4.2, y1=5.2, fillcolor="rgba(234, 179, 8, 0.35)", line=dict(color="#111111", width=2))
fig.update_layout(title="GDP Nowcast vs Official Target Box (4.2% - 5.2%)", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_res, use_container_width=True)
