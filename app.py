import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an

st.set_page_config(page_title="Marketing Campaign Dashboard", page_icon="🛒",
                   layout="wide")


@st.cache_data(show_spinner="Preparing data and training the response model...")
def get_data():
    raw = an.load_data("data/marketing_campaign.csv")
    df = an.prepare(raw)
    df, auc, importance = an.add_model_scores(df)
    return df, auc, importance


full, auc, importance = get_data()

# ----------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
st.sidebar.caption("Economics (used for profit and break-even)")
rev = st.sidebar.number_input("Revenue per accepted offer", min_value=0.1,
                              value=11.0, step=0.5)
cost = st.sidebar.number_input("Cost per customer contacted", min_value=0.1,
                               value=3.0, step=0.5)
breakeven = cost / rev

st.sidebar.divider()
st.sidebar.caption("Filter customers")
cltv_sel = st.sidebar.multiselect("CLTV segment", an.CLTV_ORDER,
                                  default=an.CLTV_ORDER)
fat_sel = st.sidebar.multiselect("Fatigue level", an.FATIGUE_ORDER,
                                 default=an.FATIGUE_ORDER)
df = full[full["CLTV_seg"].isin(cltv_sel) & full["Fatigue"].isin(fat_sel)]

st.title("🛒 Marketing Campaign Dashboard")
st.caption("Campaign performance, CLTV segments, customer fatigue and targeting "
           "strategy. Campaigns 1-6 are AcceptedCmp1-5 and Response, in that order.")

if df.empty:
    st.warning("No customers match the current filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["1. Campaign performance", "2. CLTV segments",
                                  "3. Customer fatigue", "4. Targeting strategy"])

# ================================================================ TAB 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in view", f"{len(df):,}")
    c2.metric("Campaign 6 response rate", f"{df['Response'].mean():.1%}")
    c3.metric("Never accepted any campaign", f"{(df['nAcc'] == 0).mean():.0%}")
    c4.metric("Break-even response rate", f"{breakeven:.1%}")

    perf = pd.DataFrame({
        "Campaign": an.CAMPAIGN_LABELS,
        "Accepted": [int(df[c].sum()) for c in an.CAMPAIGNS],
    })
    perf["Rate"] = perf["Accepted"] / len(df)
    perf["Revenue"] = perf["Accepted"] * rev
    perf["Cost"] = len(df) * cost
    perf["Net"] = perf["Revenue"] - perf["Cost"]

    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(perf, x="Campaign", y="Rate", text=perf["Rate"].map("{:.1%}".format))
        fig.add_hline(y=breakeven, line_dash="dash", line_color="red",
                      annotation_text=f"Break-even {breakeven:.1%}")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Acceptance rate",
                          height=380, margin=dict(t=30))
        st.plotly_chart(fig)
    with right:
        st.dataframe(perf.style.format({"Rate": "{:.1%}", "Revenue": "{:,.0f}",
                                        "Cost": "{:,.0f}", "Net": "{:,.0f}"}),
                     hide_index=True)
        st.caption("Net = revenue minus the cost of contacting everyone in view. "
                   "Bars below the red line lose money when everyone is contacted.")

    st.subheader("Who responds to Campaign 6?")
    a, b = st.columns(2)
    with a:
        by_prior = (df.assign(prior=df["nPrior"].clip(upper=4).astype(int).astype(str)
                              .replace({"4": "4+"}))
                    .groupby("prior")["Response"].agg(["mean", "size"]).reset_index())
        fig = px.bar(by_prior, x="prior", y="mean",
                     text=by_prior["mean"].map("{:.0%}".format),
                     labels={"prior": "Campaigns accepted before (1-5)",
                             "mean": "Campaign 6 response rate"})
        fig.update_layout(yaxis_tickformat=".0%", height=340, margin=dict(t=30))
        st.plotly_chart(fig)
    with b:
        dim = st.selectbox("Break response down by",
                           ["Income band", "Household", "Recency band",
                            "Education", "Marital"])
        g = df.groupby(dim)["Response"].agg(["mean", "size"]).reset_index()
        fig = px.bar(g, x=dim, y="mean", text=g["mean"].map("{:.0%}".format),
                     labels={"mean": "Campaign 6 response rate"})
        fig.update_layout(yaxis_tickformat=".0%", height=340, margin=dict(t=30))
        st.plotly_chart(fig)

    prof_cols = {"Income": "Income", "Recency": "Days since last purchase",
                 "Spend": "Total spend", "NumCatalogPurchases": "Catalog purchases",
                 "NumDealsPurchases": "Deal purchases", "Kids": "Children at home",
                 "NumWebVisitsMonth": "Web visits / month"}
    prof = (df.groupby("Response")[list(prof_cols)].mean().T
            .rename(columns={0: "Non-responders", 1: "Responders"})
            .rename(index=prof_cols).round(1))
    st.markdown("**Responders vs non-responders (averages)**")
    st.dataframe(prof)

# ================================================================ TAB 2
with tab2:
    st.markdown("Customers are scored 1-5 on **Recency, Frequency and Monetary** value, "
                "the three scores are added (3-15) and the customers are split into "
                "three equal groups. **CLTV** = monthly spend run-rate x 24 months x a "
                "recency-based retention factor (revenue, not margin).")
    seg = (full.groupby("CLTV_seg")
           .agg(Customers=("ID", "size"), Recency_days=("Recency", "mean"),
                Purchases=("F", "mean"), Avg_spend=("Spend", "mean"),
                Avg_CLTV=("CLTV", "mean"), Income=("Income", "mean"),
                Children=("Kids", "mean"), Catalog=("NumCatalogPurchases", "mean"),
                Campaign6_rate=("Response", "mean"), Total_CLTV=("CLTV", "sum"))
           .reindex(an.CLTV_ORDER))
    seg["Share_of_CLTV"] = seg["Total_CLTV"] / seg["Total_CLTV"].sum()
    st.caption("This tab always shows all customers so segments stay comparable.")
    st.dataframe(seg.drop(columns="Total_CLTV").style.format({
        "Recency_days": "{:.0f}", "Purchases": "{:.1f}", "Avg_spend": "{:,.0f}",
        "Avg_CLTV": "{:,.0f}", "Income": "{:,.0f}", "Children": "{:.2f}",
        "Catalog": "{:.1f}", "Campaign6_rate": "{:.1%}", "Share_of_CLTV": "{:.1%}"}))

    a, b = st.columns(2)
    with a:
        fig = px.bar(seg.reset_index(), x="CLTV_seg", y="Share_of_CLTV",
                     text=seg["Share_of_CLTV"].map("{:.0%}".format).values,
                     labels={"CLTV_seg": "CLTV segment", "Share_of_CLTV": "Share of total CLTV"},
                     category_orders={"CLTV_seg": an.CLTV_ORDER})
        fig.update_layout(yaxis_tickformat=".0%", height=340, margin=dict(t=30))
        st.plotly_chart(fig)
    with b:
        cat = full.groupby("CLTV_seg")[an.SPEND_COLS].sum().reindex(an.CLTV_ORDER)
        cat = cat.div(cat.sum(axis=1), axis=0).reset_index().melt(
            id_vars="CLTV_seg", var_name="Category", value_name="Share")
        cat["Category"] = cat["Category"].str.replace("Mnt", "").str.replace("Prods", "")
        fig = px.bar(cat, x="CLTV_seg", y="Share", color="Category",
                     category_orders={"CLTV_seg": an.CLTV_ORDER},
                     labels={"CLTV_seg": "CLTV segment", "Share": "Share of spend"})
        fig.update_layout(yaxis_tickformat=".0%", height=340, margin=dict(t=30))
        st.plotly_chart(fig)

    ch = full.groupby("CLTV_seg")[an.PURCHASE_COLS].sum().reindex(an.CLTV_ORDER)
    ch = ch.div(ch.sum(axis=1), axis=0).reset_index().melt(
        id_vars="CLTV_seg", var_name="Channel", value_name="Share")
    ch["Channel"] = ch["Channel"].str.replace("Num", "").str.replace("Purchases", "")
    fig = px.bar(ch, x="Share", y="CLTV_seg", color="Channel", orientation="h",
                 category_orders={"CLTV_seg": an.CLTV_ORDER},
                 labels={"CLTV_seg": "CLTV segment", "Share": "Share of purchases"})
    fig.update_layout(xaxis_tickformat=".0%", height=280, margin=dict(t=30))
    st.plotly_chart(fig)

# ================================================================ TAB 3
with tab3:
    st.markdown("Fatigue is defined only from **campaigns 1-5**, then checked against "
                "**Campaign 6**, so the check is a genuine test of the levels.")
    fat = (df.groupby("Fatigue")
           .agg(Customers=("ID", "size"), Campaign6_rate=("Response", "mean"),
                Responders=("Response", "sum"), Avg_CLTV=("CLTV", "mean"),
                Days_since_purchase=("Recency", "mean"), Income=("Income", "mean"),
                Children=("Kids", "mean"))
           .reindex([f for f in an.FATIGUE_ORDER if f in df["Fatigue"].unique()]))
    fat["Share_of_customers"] = fat["Customers"] / fat["Customers"].sum()
    st.dataframe(fat.style.format({
        "Campaign6_rate": "{:.1%}", "Avg_CLTV": "{:,.0f}", "Days_since_purchase": "{:.0f}",
        "Income": "{:,.0f}", "Children": "{:.2f}", "Share_of_customers": "{:.1%}"}))

    a, b = st.columns(2)
    with a:
        f2 = fat.reset_index()
        fig = px.bar(f2, x="Campaign6_rate", y="Fatigue", orientation="h",
                     text=f2["Campaign6_rate"].map("{:.0%}".format),
                     category_orders={"Fatigue": an.FATIGUE_ORDER},
                     labels={"Campaign6_rate": "Campaign 6 response rate"})
        fig.update_layout(xaxis_tickformat=".0%", height=360, margin=dict(t=30),
                          yaxis_title=None)
        st.plotly_chart(fig)
    with b:
        heat = (df.pivot_table(index="Fatigue", columns="CLTV_seg", values="Response",
                               aggfunc="mean")
                .reindex(index=[f for f in an.FATIGUE_ORDER if f in df["Fatigue"].unique()],
                         columns=[c for c in an.CLTV_ORDER if c in df["CLTV_seg"].unique()]))
        fig = px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                        labels={"color": "Campaign 6 rate"})
        fig.update_layout(height=360, margin=dict(t=30), xaxis_title="CLTV segment",
                          yaxis_title=None)
        st.plotly_chart(fig)
        st.caption("Blank cells mean no customers in that combination.")

    st.info("**Biggest opportunity: 'Ignoring buyers'.** They buy recently and spend "
            "above the median but have never accepted a campaign. Most are High CLTV. "
            "Test new channels and offers built on what they actually buy.")

# ================================================================ TAB 4
with tab4:
    st.markdown(f"A gradient-boosting model predicts each customer's chance of responding "
                f"(cross-validated AUC **{auc:.2f}**). Customers are tiered against the "
                f"break-even rate of **{breakeven:.1%}**.")

    d = df.copy()
    d["Tier"] = an.tier_from_p(d["p_response"], breakeven)

    tiers = (d.groupby("Tier")
             .agg(Customers=("ID", "size"), Expected_rate=("p_response", "mean"),
                  Actual_rate=("Response", "mean"), Responders=("Response", "sum"),
                  Avg_CLTV=("CLTV", "mean")))
    tiers["Net if contacted"] = tiers["Responders"] * rev - tiers["Customers"] * cost
    st.dataframe(tiers.style.format({
        "Expected_rate": "{:.1%}", "Actual_rate": "{:.1%}", "Avg_CLTV": "{:,.0f}",
        "Net if contacted": "{:,.0f}"}))
    st.caption("Actual_rate is what happened in Campaign 6 for customers in each tier.")

    st.subheader("How many customers should we contact?")
    pct = st.slider("Share of customers contacted (highest predicted response first)",
                    5, 100, 20, step=5)
    ranked = d.sort_values("p_response", ascending=False).reset_index(drop=True)
    ranked["contacts"] = np.arange(1, len(ranked) + 1)
    ranked["cum_resp"] = ranked["Response"].cumsum()
    ranked["net"] = ranked["cum_resp"] * rev - ranked["contacts"] * cost
    ranked["pct_contacted"] = ranked["contacts"] / len(ranked) * 100
    n_sel = max(1, int(round(len(ranked) * pct / 100)))
    sel = ranked.iloc[n_sel - 1]
    everyone = ranked.iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers contacted", f"{n_sel:,}")
    m2.metric("Responders captured", f"{int(sel['cum_resp']):,}",
              f"{sel['cum_resp'] / max(everyone['cum_resp'], 1):.0%} of all")
    m3.metric("Response rate", f"{sel['cum_resp'] / n_sel:.1%}")
    m4.metric("Net result", f"{sel['net']:,.0f}",
              f"{sel['net'] - everyone['net']:,.0f} vs contacting everyone")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ranked["pct_contacted"], y=ranked["net"], mode="lines",
                             name="Net result"))
    fig.add_vline(x=pct, line_dash="dot")
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(height=340, margin=dict(t=30), xaxis_title="% of customers contacted",
                      yaxis_title="Net result (revenue - cost)")
    st.plotly_chart(fig)
    best = ranked.loc[ranked["net"].idxmax()]
    st.caption(f"Profit peaks when contacting about {best['pct_contacted']:.0f}% of the "
               f"customers in view (net {best['net']:,.0f}).")

    st.subheader("What drives response?")
    imp = importance.head(10).sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
    fig.update_layout(height=340, margin=dict(t=30))
    st.plotly_chart(fig)

    st.subheader("Recommended treatment by fatigue level")
    st.table(pd.DataFrame({
        "Fatigue level": an.FATIGUE_ORDER,
        "Recommended action": [
            "Cap contact frequency. Reward loyalty (early access, points) instead of discounting.",
            "Repeat the offer type that worked last time. Keep contact frequent.",
            "Win-back offer based on the earlier accepted campaign. Send low-cost channels first.",
            "Test new channels (catalog first) and personalised offers built on their basket. Run A/B tests.",
            "One cheap win-back attempt (email/app), then suppress from paid channels.",
        ],
    }))
    st.markdown("**Other rules of thumb from the data:** lead with premium range rather than "
                "blanket discounts, trigger offers within about 30 days of a customer's last "
                "purchase, and keep a 10% hold-out group uncontacted to measure true lift.")

    export_cols = ["ID", "CLTV_seg", "Fatigue", "p_response", "Tier", "Recency", "Spend",
                   "CLTV", "nPrior"]
    st.download_button("Download target list (CSV)",
                       d.sort_values("p_response", ascending=False)[export_cols]
                       .to_csv(index=False).encode("utf-8"),
                       file_name="target_list.csv", mime="text/csv")
