import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iowa Liquor Sales Dashboard",
    page_icon="🥃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #1a0a0a; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #2d1010, #3a1515);
        border: 1px solid #5c2020;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label { color: #c9a0a0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
    .metric-value { color: #fde8e8; font-size: 1.7rem; font-weight: 700; }
    .metric-delta { font-size: 0.8rem; margin-top: 0.2rem; }
    .metric-delta.pos { color: #e8735a; }
    .metric-delta.neg { color: #ff6b6b; }
    .section-title {
        color: #f5d0d0; font-size: 1.1rem; font-weight: 600;
        border-left: 3px solid #c0392b; padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stSidebar"] { background-color: #150808; border-right: 1px solid #2d1010; }
    .stSelectbox label, .stMultiSelect label { color: #c9a0a0 !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("iowa_liquor_clean_2026.csv", parse_dates=["date"])
    df["month_period"] = df["date"].dt.to_period("M").astype(str)
    df["month_label"] = df["date"].dt.strftime("%b %Y")
    return df

df_raw = load_data()


# ─────────────────────────────────────────────
# SIDEBAR – FILTER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥃 Iowa Liquor Sales")
    st.markdown("**Dashboard Analisis 2026**")
    st.markdown("---")

    st.markdown("### Filter Data")

    # Bulan
    all_months = sorted(df_raw["month_period"].unique())
    selected_months = st.multiselect(
        "Pilih Bulan",
        options=all_months,
        default=all_months,
    )

    # Kategori
    all_cats = sorted(df_raw["category_name"].unique())
    selected_cats = st.multiselect(
        "Kategori Liquor",
        options=all_cats,
        default=all_cats,
    )

    # Kota
    all_cities = sorted(df_raw["city"].unique())
    selected_cities = st.multiselect(
        "Kota",
        options=all_cities,
        default=all_cities,
    )

    st.markdown("---")
    st.markdown("### Pengaturan Visualisasi")
    top_n = st.slider("Jumlah Top N", min_value=5, max_value=20, value=10)
    st.markdown("---")
    st.caption("Data: BigQuery `iowa_liquor_sales.sales`  \nPeriode: Jan–Apr 2026")

# Apply filter
df = df_raw[
    df_raw["month_period"].isin(selected_months) &
    df_raw["category_name"].isin(selected_cats) &
    df_raw["city"].isin(selected_cities)
].copy()

if df.empty:
    st.error("Tidak ada data dengan filter yang dipilih. Silakan sesuaikan filter.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🥃 Iowa Liquor Sales — Dashboard Analisis 2026")
st.markdown(
    f"Menampilkan **{len(df):,}** transaksi · "
    f"**{df['city'].nunique()}** kota · "
    f"**{df['category_name'].nunique()}** kategori · "
    f"**{df['store_name'].nunique()}** toko"
)
st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 1 – KPI METRICS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">Ringkasan Kinerja</div>', unsafe_allow_html=True)

total_sales = df["sale_dollars"].sum()
total_bottles = int(df["bottles_sold"].sum())
total_transaksi = len(df)
avg_sales_per_tx = df["sale_dollars"].mean()
total_volume = df["volume_sold_liters"].sum()
avg_margin = df["margin_pct"].mean()

# MoM growth (bulan terakhir vs sebelumnya)
monthly_sales = df.groupby("month_period")["sale_dollars"].sum().sort_index()
if len(monthly_sales) >= 2:
    last_m = monthly_sales.iloc[-1]
    prev_m = monthly_sales.iloc[-2]
    mom_growth = (last_m - prev_m) / prev_m * 100
else:
    mom_growth = None

col1, col2, col3, col4, col5, col6 = st.columns(6)

def metric_card(col, label, value, delta=None):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="metric-delta {cls}">{sign} {abs(delta):.1f}% MoM</div>'
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

metric_card(col1, "Total Revenue", f"${total_sales/1e6:.2f}M", mom_growth)
metric_card(col2, "Total Transaksi", f"{total_transaksi:,}")
metric_card(col3, "Total Botol Terjual", f"{total_bottles:,}")
metric_card(col4, "Avg Sales / Transaksi", f"${avg_sales_per_tx:,.0f}")
metric_card(col5, "Total Volume (Liter)", f"{total_volume:,.0f}L")
metric_card(col6, "Avg Margin", f"{avg_margin:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 2 – DISTRIBUSI SALE DOLLARS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">1. Distribusi Sale Dollars</div>', unsafe_allow_html=True)

col_hist, col_box = st.columns(2)

with col_hist:
    clip_val = df["sale_dollars"].quantile(0.99)
    sale_clipped = df["sale_dollars"].clip(upper=clip_val)
    fig_hist = px.histogram(
        sale_clipped, x=sale_clipped,
        nbins=60,
        title="Histogram Distribusi Sale Dollars (dipotong di persentil ke-99)",
        labels={"x": "Sale Dollars (USD)"},
        color_discrete_sequence=["#e8735a"],
        template="plotly_dark",
    )
    med = df["sale_dollars"].median()
    mean = df["sale_dollars"].mean()
    fig_hist.add_vline(x=med, line_dash="dash", line_color="#f5c6c6",
                       annotation_text=f"Median: ${med:.0f}", annotation_position="top right")
    fig_hist.add_vline(x=mean, line_dash="dash", line_color="#e8b89a",
                       annotation_text=f"Mean: ${mean:.0f}", annotation_position="top left")
    fig_hist.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=360,
        xaxis_title="Sale Dollars (USD)", yaxis_title="Frekuensi",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_box:
    skew = df["sale_dollars"].skew()
    kurt = df["sale_dollars"].kurt()
    fig_box = px.box(
        df, y="sale_dollars",
        title="Boxplot Sale Dollars (Deteksi Outlier)",
        color_discrete_sequence=["#e8735a"],
        template="plotly_dark",
        points=False,
    )
    fig_box.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=360,
        yaxis_title="Sale Dollars (USD)",
    )
    fig_box.add_annotation(
        text=f"Skewness: {skew:.2f}<br>Kurtosis: {kurt:.2f}",
        x=0.02, y=0.97, xref="paper", yref="paper",
        showarrow=False, bgcolor="#2d1010", bordercolor="#5c2020",
        font=dict(color="#c9a0a0", size=12), align="left",
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Stats deskriptif
with st.expander("Statistik Deskriptif Sale Dollars"):
    desc = df["sale_dollars"].describe(percentiles=[.25, .50, .75, .90, .95, .99]).round(2)
    st.dataframe(desc.to_frame().T, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 3 – TREN WAKTU
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">2. Tren Penjualan dari Waktu ke Waktu</div>', unsafe_allow_html=True)

df_monthly = (
    df.groupby("month_period")
    .agg(
        total_sales=("sale_dollars", "sum"),
        avg_sales=("sale_dollars", "mean"),
        n_transaksi=("sale_dollars", "count"),
        total_bottles=("bottles_sold", "sum"),
    )
    .reset_index()
    .sort_values("month_period")
)
df_monthly["mom_growth"] = df_monthly["total_sales"].pct_change() * 100
df_monthly["cumulative_sales"] = df_monthly["total_sales"].cumsum()

col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    fig_trend = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.08,
                               subplot_titles=("Total Penjualan Bulanan (USD)", "Jumlah Transaksi per Bulan"))

    # Trend line (linear regression)
    x_idx = np.arange(len(df_monthly))
    if len(x_idx) >= 2:
        m_coef, b_coef, _, _, _ = stats.linregress(x_idx, df_monthly["total_sales"] / 1e6)
        trend_vals = m_coef * x_idx + b_coef
        fig_trend.add_trace(go.Scatter(
            x=df_monthly["month_period"], y=trend_vals,
            mode="lines", name=f"Trend (slope={m_coef:.2f}M/bln)",
            line=dict(dash="dash", color="#f5c6c6", width=1.5)
        ), row=1, col=1)

    fig_trend.add_trace(go.Scatter(
        x=df_monthly["month_period"], y=df_monthly["total_sales"] / 1e6,
        mode="lines+markers+text",
        text=[f"${v:.1f}M" for v in df_monthly["total_sales"] / 1e6],
        textposition="top center", textfont=dict(size=10, color="#e8735a"),
        name="Total Sales", line=dict(color="#e8735a", width=2.5),
        marker=dict(size=7),
        fill="tozeroy", fillcolor="rgba(232,115,90,0.08)",
    ), row=1, col=1)

    fig_trend.add_trace(go.Bar(
        x=df_monthly["month_period"], y=df_monthly["n_transaksi"],
        name="Jumlah Transaksi", marker_color="#c97b7b",
    ), row=2, col=1)

    fig_trend.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=420,
        legend=dict(orientation="h", y=-0.12),
        showlegend=True,
    )
    fig_trend.update_xaxes(tickangle=30)
    fig_trend.update_yaxes(title_text="Juta USD", row=1, col=1)
    fig_trend.update_yaxes(title_text="Transaksi", row=2, col=1)
    st.plotly_chart(fig_trend, use_container_width=True)

with col_trend2:
    # MoM Growth + Cumulative
    fig_mom = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.08,
                             subplot_titles=("Month-over-Month Growth Rate (%)", "Penjualan Kumulatif"))

    colors_bar = ["#e8735a" if v >= 0 else "#f5c6c6" for v in df_monthly["mom_growth"].fillna(0)]
    fig_mom.add_trace(go.Bar(
        x=df_monthly["month_period"], y=df_monthly["mom_growth"].fillna(0),
        name="MoM Growth (%)", marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in df_monthly["mom_growth"].fillna(0)],
        textposition="outside",
    ), row=1, col=1)
    fig_mom.add_hline(y=0, line_color="#ffffff", line_width=0.5, row=1, col=1)

    fig_mom.add_trace(go.Scatter(
        x=df_monthly["month_period"], y=df_monthly["cumulative_sales"] / 1e6,
        mode="lines+markers",
        name="Kumulatif",
        line=dict(color="#c0392b", width=2.5),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.1)",
    ), row=2, col=1)

    fig_mom.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=420,
        showlegend=False,
    )
    fig_mom.update_xaxes(tickangle=30)
    fig_mom.update_yaxes(title_text="%", row=1, col=1)
    fig_mom.update_yaxes(title_text="Juta USD", row=2, col=1)
    st.plotly_chart(fig_mom, use_container_width=True)

# Tabel insight
st.markdown("**Tabel Agregasi Bulanan:**")
df_monthly_display = df_monthly[["month_period", "total_sales", "avg_sales", "n_transaksi", "total_bottles", "mom_growth", "cumulative_sales"]].copy()
df_monthly_display.columns = ["Bulan", "Total Sales ($)", "Avg Sales ($)", "Transaksi", "Total Botol", "MoM Growth (%)", "Kumulatif ($)"]
df_monthly_display = df_monthly_display.round(2)
st.dataframe(df_monthly_display.style.format({
    "Total Sales ($)": "${:,.0f}", "Avg Sales ($)": "${:,.0f}",
    "Transaksi": "{:,}", "Total Botol": "{:,}",
    "MoM Growth (%)": "{:.2f}%", "Kumulatif ($)": "${:,.0f}",
}), use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4 – TOP KATEGORI
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">3. Analisis Kategori Liquor</div>', unsafe_allow_html=True)

df_cat = (
    df.groupby("category_name")
    .agg(
        total_sales=("sale_dollars", "sum"),
        avg_sales=("sale_dollars", "mean"),
        median_sales=("sale_dollars", "median"),
        total_bottles=("bottles_sold", "sum"),
        n_transaksi=("sale_dollars", "count"),
    )
    .reset_index()
    .sort_values("total_sales", ascending=False)
)
df_cat["share_pct"] = (df_cat["total_sales"] / df_cat["total_sales"].sum() * 100).round(2)

col_cat1, col_cat2 = st.columns(2)

with col_cat1:
    df_top_cat = df_cat.head(top_n).sort_values("total_sales")
    fig_cat_total = px.bar(
        df_top_cat, x="total_sales", y="category_name",
        orientation="h",
        title=f"Top {top_n} Kategori — Total Sales",
        text=df_top_cat["total_sales"].apply(lambda v: f"${v/1e3:.0f}K"),
        color="total_sales",
        color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_cat_total.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=400,
        xaxis_title="Total Sales (USD)", yaxis_title="",
        coloraxis_showscale=False, showlegend=False,
    )
    fig_cat_total.update_traces(textposition="outside")
    st.plotly_chart(fig_cat_total, use_container_width=True)

with col_cat2:
    df_top_avg = df_cat.sort_values("avg_sales", ascending=False).head(top_n).sort_values("avg_sales")
    fig_cat_avg = px.bar(
        df_top_avg, x="avg_sales", y="category_name",
        orientation="h",
        title=f"Top {top_n} Kategori — Rata-rata Sales per Transaksi",
        text=df_top_avg["avg_sales"].apply(lambda v: f"${v:,.0f}"),
        color="avg_sales",
        color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_cat_avg.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=400,
        xaxis_title="Avg Sales per Transaksi (USD)", yaxis_title="",
        coloraxis_showscale=False, showlegend=False,
    )
    fig_cat_avg.update_traces(textposition="outside")
    st.plotly_chart(fig_cat_avg, use_container_width=True)

# Pie chart share
col_pie, col_cat_stats = st.columns([1, 1])
with col_pie:
    top3_share = df_cat.head(3)["share_pct"].sum()
    df_pie = df_cat.head(8).copy()
    others = 100 - df_pie["share_pct"].sum()
    if others > 0:
        df_pie = pd.concat([df_pie, pd.DataFrame([{"category_name": "Lainnya", "share_pct": others}])], ignore_index=True)
    fig_pie = px.pie(
        df_pie, names="category_name", values="share_pct",
        title="Market Share per Kategori (%)",
        color_discrete_sequence=["#8b1a1a","#c0392b","#e74c3c","#e8735a","#c97b7b","#f1948a","#fadbd8","#7b241c","#922b21"],
        template="plotly_dark", hole=0.4,
    )
    fig_pie.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=360,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_cat_stats:
    st.markdown("**Insight Kategori Liquor**")
    top_total = df_cat.iloc[0]
    top_avg_cat = df_cat.sort_values("avg_sales", ascending=False).iloc[0]
    st.markdown(f"""
    | Insight | Nilai |
    |---|---|
    | Kategori terlaris (total) | **{top_total['category_name']}** |
    | Total revenue kategori #1 | **${top_total['total_sales']:,.0f}** |
    | Share revenue kategori #1 | **{top_total['share_pct']:.2f}%** |
    | Jumlah transaksi kategori #1 | **{top_total['n_transaksi']:,}** |
    | Kategori avg tertinggi | **{top_avg_cat['category_name']}** |
    | Avg sales / transaksi tertinggi | **${top_avg_cat['avg_sales']:,.2f}** |
    | Top 3 kategori kuasai | **{df_cat.head(3)['share_pct'].sum():.1f}% revenue** |
    | Total kategori unik | **{len(df_cat)}** |
    """)

    # Tabel kategori
    st.markdown("**Tabel Top Kategori:**")
    st.dataframe(
        df_cat.head(top_n)[["category_name", "total_sales", "avg_sales", "n_transaksi", "share_pct"]]
        .rename(columns={"category_name": "Kategori", "total_sales": "Total Sales ($)",
                         "avg_sales": "Avg Sales ($)", "n_transaksi": "Transaksi", "share_pct": "Share (%)"})
        .style.format({"Total Sales ($)": "${:,.0f}", "Avg Sales ($)": "${:,.0f}", "Share (%)": "{:.2f}%"}),
        use_container_width=True, height=280,
    )


# ─────────────────────────────────────────────
# SECTION 5 – TOP KOTA & TOKO
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">4. Kontribusi Penjualan: Kota & Toko</div>', unsafe_allow_html=True)

df_city_agg = (
    df.groupby("city")
    .agg(total_revenue=("sale_dollars", "sum"), avg_revenue=("sale_dollars", "mean"),
         n_transaksi=("sale_dollars", "count"), total_bottles=("bottles_sold", "sum"))
    .reset_index().sort_values("total_revenue", ascending=False)
)
df_city_agg["share_pct"] = (df_city_agg["total_revenue"] / df_city_agg["total_revenue"].sum() * 100).round(2)

df_store_agg = (
    df.groupby("store_name")
    .agg(total_revenue=("sale_dollars", "sum"), avg_revenue=("sale_dollars", "mean"),
         n_transaksi=("sale_dollars", "count"))
    .reset_index().sort_values("total_revenue", ascending=False)
)
df_store_agg["share_pct"] = (df_store_agg["total_revenue"] / df_store_agg["total_revenue"].sum() * 100).round(2)

col_city, col_store = st.columns(2)

with col_city:
    df_plot_city = df_city_agg.head(top_n).sort_values("total_revenue")
    fig_city = px.bar(
        df_plot_city, x="total_revenue", y="city",
        orientation="h",
        title=f"Top {top_n} Kota — Total Revenue",
        text=df_plot_city.apply(lambda r: f"${r['total_revenue']/1e3:.0f}K ({r['share_pct']:.1f}%)", axis=1),
        color="total_revenue", color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_city.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=420,
        xaxis_title="Total Revenue (USD)", yaxis_title="",
        coloraxis_showscale=False,
    )
    fig_city.update_traces(textposition="outside")
    st.plotly_chart(fig_city, use_container_width=True)

with col_store:
    df_plot_store = df_store_agg.head(top_n).sort_values("total_revenue")
    fig_store = px.bar(
        df_plot_store, x="total_revenue", y="store_name",
        orientation="h",
        title=f"Top {top_n} Toko — Total Revenue",
        text=df_plot_store.apply(lambda r: f"${r['total_revenue']/1e3:.0f}K ({r['share_pct']:.1f}%)", axis=1),
        color="total_revenue", color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_store.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=420,
        xaxis_title="Total Revenue (USD)", yaxis_title="",
        coloraxis_showscale=False,
    )
    fig_store.update_traces(textposition="outside")
    st.plotly_chart(fig_store, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 6 – KORELASI & REGRESI
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">5. Analisis Korelasi & Regresi Linear</div>', unsafe_allow_html=True)

corr_cols = ["sale_dollars", "bottles_sold", "state_bottle_retail", "volume_sold_liters"]
df_corr = df[corr_cols].corr().round(4)

col_heat, col_scatter = st.columns([1, 1])

with col_heat:
    fig_heat = px.imshow(
        df_corr,
        text_auto=".4f",
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        title="Correlation Heatmap — Variabel Numerik Utama",
        template="plotly_dark",
    )
    fig_heat.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=380,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_scatter:
    scatter_x = st.selectbox(
        "Pilih variabel X (scatter):",
        options=["bottles_sold", "state_bottle_retail", "volume_sold_liters", "bottle_volume_ml"],
        index=0,
    )
    scatter_y = "sale_dollars"

    sample_df = df.sample(min(2000, len(df)), random_state=42)
    df_clean = df[[scatter_x, scatter_y]].dropna()
    r, p_val = stats.pearsonr(df_clean[scatter_x], df_clean[scatter_y])
    X = df_clean[[scatter_x]].values
    y_arr = df_clean[scatter_y].values
    reg = LinearRegression().fit(X, y_arr)
    r2 = reg.score(X, y_arr)
    coef = reg.coef_[0]
    intercept = reg.intercept_

    x_range = np.linspace(X.min(), X.max(), 200)
    y_pred = reg.predict(x_range.reshape(-1, 1))

    if abs(r) >= 0.7:
        interp = "Korelasi Kuat"
    elif abs(r) >= 0.4:
        interp = "Korelasi Sedang"
    else:
        interp = "Korelasi Lemah"

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=sample_df[scatter_x], y=sample_df[scatter_y],
        mode="markers",
        marker=dict(size=4, color="#e8735a", opacity=0.3),
        name="Data",
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_range, y=y_pred,
        mode="lines",
        line=dict(color="#f5c6c6", width=2),
        name=f"Regresi Linear",
    ))
    fig_scatter.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=380,
        title=f"{scatter_x} vs {scatter_y}",
        xaxis_title=scatter_x, yaxis_title=scatter_y,
        legend=dict(orientation="h", y=-0.15),
    )
    fig_scatter.add_annotation(
        text=f"r = {r:.4f} ({interp})<br>R² = {r2:.4f}<br>y = {coef:.2f}x + {intercept:.2f}<br>p-value = {p_val:.2e}",
        x=0.03, y=0.97, xref="paper", yref="paper",
        showarrow=False,
        bgcolor="#2d1010", bordercolor="#5c2020",
        font=dict(color="#c9a0a0", size=11),
        align="left",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 7 – AGREGASI LANJUTAN
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">6. Agregasi Waktu: Sales & Bottles Sold</div>', unsafe_allow_html=True)

df_time = (
    df.groupby("month_period")
    .agg(
        total_sales=("sale_dollars", "sum"),
        total_bottles=("bottles_sold", "sum"),
    )
    .reset_index().sort_values("month_period")
)

col_t1, col_t2 = st.columns(2)

with col_t1:
    fig_sales_line = go.Figure()
    fig_sales_line.add_trace(go.Scatter(
        x=df_time["month_period"], y=df_time["total_sales"] / 1e6,
        mode="lines+markers+text",
        text=[f"${v:.1f}M" for v in df_time["total_sales"] / 1e6],
        textposition="top center",
        line=dict(color="#8b1a1a", width=2.5),
        marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(139,26,26,0.12)",
        name="Total Sales",
    ))
    fig_sales_line.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=300,
        title="Total Sales per Bulan (Juta USD)",
        xaxis_title="Bulan", yaxis_title="Juta USD",
        xaxis=dict(tickangle=30),
    )
    st.plotly_chart(fig_sales_line, use_container_width=True)

with col_t2:
    fig_bottles_line = go.Figure()
    fig_bottles_line.add_trace(go.Scatter(
        x=df_time["month_period"], y=df_time["total_bottles"] / 1e3,
        mode="lines+markers+text",
        text=[f"{v:.0f}K" for v in df_time["total_bottles"] / 1e3],
        textposition="top center",
        line=dict(color="#c0392b", width=2.5),
        marker=dict(size=8, symbol="square"),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.12)",
        name="Total Bottles",
    ))
    fig_bottles_line.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=300,
        title="Total Bottles Sold per Bulan (Ribu)",
        xaxis_title="Bulan", yaxis_title="Ribu Botol",
        xaxis=dict(tickangle=30),
    )
    st.plotly_chart(fig_bottles_line, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 8 – OUTLIER (IQR)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">7. Deteksi Outlier — Metode IQR</div>', unsafe_allow_html=True)

cols_to_check = ["sale_dollars", "bottles_sold", "volume_sold_liters", "state_bottle_cost", "state_bottle_retail"]
outlier_data = []
for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = round(n_outlier / len(df) * 100, 2)
    outlier_data.append({
        "Kolom": col, "Batas Bawah (IQR)": round(lower, 2),
        "Batas Atas (IQR)": round(upper, 2), "Jumlah Outlier": n_outlier, "Persentase (%)": pct
    })

df_outlier = pd.DataFrame(outlier_data)
col_out1, col_out2 = st.columns([1, 1])

with col_out1:
    st.dataframe(df_outlier.style.format({
        "Batas Bawah (IQR)": "{:,.2f}", "Batas Atas (IQR)": "{:,.2f}",
        "Jumlah Outlier": "{:,}", "Persentase (%)": "{:.2f}%",
    }), use_container_width=True, height=230)

with col_out2:
    fig_outlier_bar = px.bar(
        df_outlier, x="Kolom", y="Persentase (%)",
        title="Persentase Outlier per Kolom (%)",
        text=df_outlier["Persentase (%)"].apply(lambda v: f"{v:.2f}%"),
        color="Persentase (%)", color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig_outlier_bar.update_layout(
        plot_bgcolor="#1a0a0a", paper_bgcolor="#1a0a0a",
        font_color="#f5d0d0", height=260,
        coloraxis_showscale=False,
    )
    fig_outlier_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_outlier_bar, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 9 – RAW DATA
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">8. Data Mentah (Subset)</div>', unsafe_allow_html=True)

with st.expander("Lihat dataset (maks 500 baris pertama)"):
    cols_show = ["date", "store_name", "city", "county", "category_name", "item_description",
                 "bottles_sold", "sale_dollars", "volume_sold_liters", "margin_pct"]
    st.dataframe(df[cols_show].head(500), use_container_width=True, height=300)

    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data Terfilter (.csv)",
        data=csv_export,
        file_name="iowa_liquor_filtered.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#7a3535; font-size:0.8rem;'>"
    "Iowa Liquor Sales Dashboard 2026 · Data: BigQuery Public Dataset · "
    "Preprocessed: 757,888 baris → 30 kolom</p>",
    unsafe_allow_html=True
)
