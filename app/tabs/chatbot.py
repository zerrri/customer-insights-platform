import streamlit as st
import pandas as pd

HELP_TEXT = """
Examples:
- "top risk customers"
- "top 10 risk customers with highest CLTV"
- "segment counts"
- "customers in Champions segment with risk > 0.7"
"""

def _top_risk(df: pd.DataFrame, k: int = 10, threshold: float | None = None):
    work = df.copy()
    if threshold is not None and "Churn_Probability" in work.columns:
        work = work[work["Churn_Probability"] >= threshold]
    cols = [c for c in ["CustomerID", "Churn_Probability", "CLTV", "Segment"] if c in work.columns]
    return work.sort_values(["Churn_Probability", "CLTV"], ascending=[False, False])[cols].head(k)

def render(df: pd.DataFrame):
    st.subheader("Chatbot (lightweight)")
    st.caption("Simple data Q&A over the current filtered dataset. No external API keys used.")
    st.caption(HELP_TEXT)

    q = st.text_input("Ask about the data")
    if not q:
        return

    q_lower = q.lower().strip()
    threshold = st.session_state.get("churn_threshold", 0.6)

    try:
        if "segment counts" in q_lower or ("segment" in q_lower and "count" in q_lower):
            if "Segment" in df.columns:
                st.write(df["Segment"].value_counts())
            else:
                st.info("No Segment column available.")
            return

        if "top" in q_lower and "risk" in q_lower:
            # detect top k if present
            k = 10
            for tok in q_lower.split():
                if tok.isdigit():
                    k = int(tok)
                    break
            # detect threshold override like > 0.7
            thr = threshold
            if ">" in q_lower:
                try:
                    thr = float(q_lower.split(">")[-1].strip().split()[0])
                except:
                    pass
            ans = _top_risk(df, k=k, threshold=thr)
            if ans.empty:
                st.info("No customers match that query.")
            else:
                st.write(ans)
            return

        # Simple segment + risk query
        if "segment" in q_lower and ("risk" in q_lower or "churn" in q_lower):
            # extract a possible segment keyword
            seg_candidates = ["champions", "loyal", "at risk", "hibernating"]
            target_seg = None
            for s in seg_candidates:
                if s in q_lower:
                    target_seg = s.title()
                    break
            thr = threshold
            if ">" in q_lower:
                try:
                    thr = float(q_lower.split(">")[-1].strip().split()[0])
                except:
                    pass

            work = df.copy()
            if target_seg and "Segment" in work.columns:
                work = work[work["Segment"].str.lower() == target_seg.lower()]
            if "Churn_Probability" in work.columns:
                work = work[work["Churn_Probability"] >= thr]

            cols = [c for c in ["CustomerID", "Segment", "CLTV", "Churn_Probability"] if c in work.columns]
            st.write(work.sort_values(["Churn_Probability", "CLTV"], ascending=[False, False])[cols].head(25))
            return

        st.write("Sorry, I didn't get that. Try the examples above.")
    except Exception as e:
        st.warning(f"Query failed: {e}")
