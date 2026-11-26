import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churn Risk", layout="wide")

# ---------- 1. Load the saved pieces ----------
@st.cache_resource
def load_artifact():
    return joblib.load("churn_model.joblib")

@st.cache_data
def load_tables():
    scores = pd.read_csv("train_scores.csv")["churn_probability"]
    imp = pd.read_csv("family_importance.csv", index_col=0)
    return scores, imp

artifact     = load_artifact()
pipe         = artifact["pipeline"]
threshold    = artifact["threshold"]
feature_cols = artifact["feature_cols"]
train_scores, family_imp = load_tables()

# ---------- 2. Feature engineering ----------
def recompute_engineered(df):
    d = df.copy()
    epsilon = 1e-15
    d["Year"] = 2025
    d["Balance_to_Salary"] = d["Balance"] / (d["EstimatedSalary"] + epsilon)
    d["Product_Density"] = d["NumOfProducts"] / (d["Tenure"] + 1)
    d["Age_Tenure_Ratio"] = d["Tenure"] / d["Age"]
    d["Age_Tenure_Interaction"] = d["Age"] * d["Tenure"]
    d["Engagement_Product_Interaction"] = d["IsActiveMember"] * d["NumOfProducts"]
    d["Log_Balance_to_Salary"] = np.log1p(d["Balance_to_Salary"])
    d["Log_Product_Density"] = np.log(d["Product_Density"])
    return d

# ---------- 3. Scoring: raw inputs -> engineered features -> probability ----------
def score(raw: dict) -> float:
    df = recompute_engineered(pd.DataFrame([raw]))
    return float(pipe.predict_proba(df[feature_cols])[:, 1][0])

# ---------- 4. Sidebar: input customer features ----------
st.sidebar.header("Customer features")
raw = {
    "CreditScore":     st.sidebar.slider("Credit score", 350, 850, 650),
    "Geography":       st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"]),
    "Gender":          st.sidebar.selectbox("Gender", ["Female", "Male"]),
    "Age":             st.sidebar.slider("Age", 18, 92, 40),
    "Tenure":          st.sidebar.slider("Tenure (years)", 0, 10, 5),
    "Balance":         st.sidebar.number_input("Balance", 0.0, 260000.0, 60000.0, step=1000.0),
    "NumOfProducts":   st.sidebar.selectbox("Number of products", [1, 2, 3, 4], index=1),
    "HasCrCard":       int(st.sidebar.checkbox("Has credit card", value=True)),
    "IsActiveMember":  int(st.sidebar.checkbox("Active member", value=True)),
    "EstimatedSalary": st.sidebar.number_input("Estimated salary", 1.0, 200000.0, 100000.0, step=1000.0),
}
prob = score(raw)
flag = int(prob >= threshold)

tab1, tab2, tab3, tab4 = st.tabs(["Risk calculator", "Probability distribution",
                                  "Feature importance", "What-if simulator"])

# ---------- Tab 1: customer churn risk calculator ----------
with tab1:
    st.subheader("Churn risk for this customer")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{prob:.1%}")
    c2.metric("Decision threshold", f"{threshold:.1%}")
    c3.metric("Churn flag", "HIGH RISK (1)" if flag else "Low risk (0)")
    st.progress(min(prob, 1.0))

# ---------- Tab 2: probability distribution ----------
with tab2:
    st.subheader("Where this customer sits among all customers")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(train_scores, bins=40, color="lightsteelblue")
    ax.axvline(prob, color="crimson", lw=2, label=f"This customer ({prob:.1%})")
    ax.axvline(threshold, color="black", ls="--", label=f"Threshold ({threshold:.1%})")
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Customers")
    ax.legend()
    st.pyplot(fig)
    pct = (train_scores < prob).mean()
    st.caption(f"This customer's risk is higher than {pct:.0%} of customers in the training data.")

# ---------- Tab 3: feature importance dashboard ----------
with tab3:
    st.subheader("What drives churn: permutation importance by feature family")
    imp = family_imp.sort_values("mean_drop")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp.index, imp["mean_drop"], xerr=2 * imp["std"],
            color="steelblue", capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Drop in ROC-AUC when the family is shuffled")
    st.pyplot(fig)

# ---------- Tab 4: what-if scenario simulator ----------
with tab4:
    st.subheader("Adjust engagement and product values")
    w1, w2, w3 = st.columns(3)
    new_products = w1.selectbox("Number of products", [1, 2, 3, 4],
                                index=raw["NumOfProducts"] - 1, key="wi_products")
    new_active   = w2.checkbox("Active member", value=bool(raw["IsActiveMember"]), key="wi_active")
    new_card     = w3.checkbox("Has credit card", value=bool(raw["HasCrCard"]), key="wi_card")

    scenario = {**raw, "NumOfProducts": new_products,
                "IsActiveMember": int(new_active), "HasCrCard": int(new_card)}
    s_prob = score(scenario)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current probability", f"{prob:.1%}")
    m2.metric("Scenario probability", f"{s_prob:.1%}", delta=f"{(s_prob - prob):+.1%}",
              delta_color="inverse")      # a fall in risk shows as green
    m3.metric("Scenario flag", "HIGH RISK (1)" if s_prob >= threshold else "Low risk (0)")

    # Curve: probability as products change, for inactive vs active members
    curve = pd.DataFrame(
        {label: [score({**raw, "NumOfProducts": p, "IsActiveMember": a}) for p in [1, 2, 3, 4]]
         for label, a in [("Inactive", 0), ("Active", 1)]},
        index=[1, 2, 3, 4])
    curve.index.name = "Number of products"
    st.line_chart(curve)
    st.caption("Results show how the model's prediction responds to the inputs, not a causal effect.")