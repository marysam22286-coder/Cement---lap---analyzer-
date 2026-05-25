"""
Cement Lab Analyzer – Enhanced Version (v4‑Pro Style)
-------------------------------------------------------
Author: DeepSeek (emulating V4‑Pro)
Description: A robust Streamlit app for cement quality assessment
and strength prediction with advanced AI fallback chains.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Optional AI stack
# -------------------------------------------------------------------
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants & Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="محلل مختبر الأسمنت",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
    "CaO": (60.0, 67.0),
    "SiO2": (18.0, 24.0),
    "Al2O3": (3.0, 8.0),
    "Fe2O3": (1.5, 5.5),
    "MgO": (0.0, 5.0),
    "SO3": (2.0, 4.5),
    "Na2O": (0.0, 0.8),
    "K2O": (0.0, 1.2),
}

FEATURES = [
    "CaO",
    "SiO2",
    "Al2O3",
    "Fe2O3",
    "MgO",
    "SO3",
    "Na2O",
    "K2O",
    "Fineness",
    "Gypsum",
    "Clinker",
    "Mineral_Addition",
    "LSF_Input",
    "SM_Input",
    "AM_Input",
]
TARGETS = ["Strength_2d", "Strength_7d", "Strength_28d"]

HISTORY_FILE = Path("cement_history.csv")

# -------------------------------------------------------------------
# Pydantic-like validation (manual)
# -------------------------------------------------------------------
@dataclass
class OxideInput:
    cao: float
    sio2: float
    al2o3: float
    fe2o3: float
    mgo: float
    so3: float
    na2o: float
    k2o: float

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "OxideInput":
        return cls(**d)

    def to_dict(self) -> Dict[str, float]:
        return {
            "CaO": self.cao,
            "SiO2": self.sio2,
            "Al2O3": self.al2o3,
            "Fe2O3": self.fe2o3,
            "MgO": self.mgo,
            "SO3": self.so3,
            "Na2O": self.na2o,
            "K2O": self.k2o,
        }

# -------------------------------------------------------------------
# Chemistry helpers
# -------------------------------------------------------------------
def safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0, None) else float("nan")

def calc_indices(ox: Dict[str, float]) -> Dict[str, float]:
    """Calculate LSF, LSA, SM, AM and approximate Bogue phases."""
    lsf = 100.0 * safe_div(
        ox["CaO"],
        (2.8 * ox["SiO2"] + 1.18 * ox["Al2O3"] + 0.65 * ox["Fe2O3"]),
    )
    lsa = 100.0 * safe_div(
        ox["CaO"],
        (2.8 * ox["SiO2"] + 1.2 * ox["Al2O3"] + 0.65 * ox["Fe2O3"]),
    )
    sm = safe_div(ox["SiO2"], (ox["Al2O3"] + ox["Fe2O3"]))
    am = safe_div(ox["Al2O3"], ox["Fe2O3"])

    c3s = max(0.0, 4.071 * ox["CaO"] - 7.602 * ox["SiO2"] - 6.718 * ox["Al2O3"] - 1.429 * ox["Fe2O3"] - 2.852 * ox["SO3"])
    c2s = max(0.0, 2.867 * ox["SiO2"] - 0.7544 * c3s)
    c3a = max(0.0, 2.650 * ox["Al2O3"] - 1.692 * ox["Fe2O3"])
    c4af = max(0.0, 3.043 * ox["Fe2O3"])

    return {
        "LSF": lsf,
        "LSA": lsa,
        "SM": sm,
        "AM": am,
        "C3S_est": c3s,
        "C2S_est": c2s,
        "C3A_est": c3a,
        "C4AF_est": c4af,
    }

def classify_sample(
    ox: Dict[str, float],
    idx: Dict[str, float],
    limits: Dict[str, Tuple[float, float]],
) -> Tuple[str, List[str]]:
    notes: List[str] = []
    score = 0.0

    for k, (lo, hi) in limits.items():
        val = ox[k]
        if val < lo:
            notes.append(f"{k} أقل من الحد الأدنى ({lo})")
            score -= 0.5
        elif val > hi:
            notes.append(f"{k} أعلى من الحد الأعلى ({hi})")
            score -= 0.5
        else:
            notes.append(f"{k} ضمن النطاق")
            score += 1.0

    # LSF
    if idx["LSF"] < 88:
        notes.append("LSF منخفض: قد يقل الكلنكر الفعال والمقاومة المبكرة")
        score -= 1.0
    elif idx["LSF"] > 100:
        notes.append("LSF مرتفع: احتمال زيادة الجير الحر أو عدم الاتزان")
        score -= 0.5
    else:
        score += 1.0

    # SM
    if idx["SM"] < 2.1:
        notes.append("SM منخفض: قد يزيد الطور السائل")
        score -= 0.5
    elif idx["SM"] > 3.0:
        notes.append("SM مرتفع: لزوجة عالية")
        score -= 0.25
    else:
        score += 0.5

    # AM
    if idx["AM"] < 1.0:
        notes.append("AM منخفض (Fe₂O₃ مرتفع)")
    elif idx["AM"] > 3.0:
        notes.append("AM مرتفع (Al₂O₃ مرتفع)")

    # SO3
    if ox["SO3"] < 2.0:
        notes.append("SO₃ منخفض: قد يضعف المقاومة المبكرة")
        score -= 0.5
    elif ox["SO3"] > 4.0:
        notes.append("SO₃ مرتفع: خطر تمدد")
        score -= 0.75

    # MgO
    if ox["MgO"] > 5.0:
        notes.append("MgO مرتفع: خطر عدم الثبات الحجمي")
        score -= 1.0

    if score >= 8:
        label = "ممتاز"
    elif score >= 5:
        label = "جيد"
    elif score >= 2:
        label = "يحتاج تعديل"
    else:
        label = "غير مطابق"

    return label, notes

def heuristic_strengths(
    ox: Dict[str, float],
    idx: Dict[str, float],
    fineness: float,
    gypsum: float,
    clinker: float,
    mineral_addition: float,
) -> Dict[str, float]:
    """Fallback strength estimation based on empirical relations."""
    base_28 = 32.0
    base_28 += (idx["LSF"] - 92.0) * 0.18
    base_28 += (idx["SM"] - 2.7) * 1.4
    base_28 += (idx["AM"] - 1.4) * 0.6
    base_28 += (ox["SO3"] - 2.8) * 0.9
    base_28 += (ox["MgO"] - 2.0) * -0.4
    base_28 += (fineness - 320.0) * 0.02
    base_28 += (clinker - 95.0) * 0.08
    base_28 += (mineral_addition - 5.0) * -0.10
    base_28 += (gypsum - 4.5) * 0.06

    return {
        "Strength_2d": float(max(5.0, min(25.0, base_28 * 0.40))),
        "Strength_7d": float(max(15.0, min(42.0, base_28 * 0.78))),
        "Strength_28d": float(max(20.0, min(60.0, base_28))),
    }

# -------------------------------------------------------------------
# AI Model management
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="تدريب النموذج...")
def train_models(df: pd.DataFrame):
    """Train GradientBoosting (or XGBoost) pipeline on historical data."""
    if not SKLEARN_AVAILABLE:
        return None

    required_cols = set(FEATURES + TARGETS)
    if not required_cols.issubset(df.columns):
        return None

    X = df[FEATURES].copy()
    y = df[TARGETS].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    models = {}
    metrics = {}
    for target in TARGETS:
        # Prefer XGBoost if available
        if XGBOOST_AVAILABLE:
            reg = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbosity=0,
            )
        else:
            reg = GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
            )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg),
        ])
        pipeline.fit(X, y[target])
        preds = pipeline.predict(X)
        models[target] = pipeline
        metrics[target] = {
            "MAE": float(mean_absolute_error(y[target], preds)),
            "R2": float(r2_score(y[target], preds)),
            "CV_MAE": float(
                -cross_val_score(
                    pipeline, X, y[target], cv=3, scoring="neg_mean_absolute_error"
                ).mean()
            ),
        }
    return models, metrics

def predict_strengths_from_model(models, sample: pd.DataFrame) -> Dict[str, float]:
    return {t: float(m.predict(sample)[0]) for t, m in models.items()}

def format_float(v: float, digits: int = 2) -> str:
    try:
        if v is None or math.isnan(v) or math.isinf(v):
            return "—"
        return f"{v:.{digits}f}"
    except Exception:
        return "—"

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.title("🏭 محلل مختبر الأسمنت")
    st.caption("أداة أولية لتقييم عينات الإسمنت وتقدير المقاومة. الدقة تتحسن مع بياناتك التاريخية.")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("⚙️ الإعدادات")
        use_custom = st.checkbox("حدود قبول مخصصة")
        limits = {}
        for oxide, (lo, hi) in DEFAULT_LIMITS.items():
            if use_custom:
                c1, c2 = st.columns(2)
                with c1:
                    nlo = st.number_input(f"{oxide} min", value=float(lo), key=f"lo_{oxide}")
                with c2:
                    nhi = st.number_input(f"{oxide} max", value=float(hi), key=f"hi_{oxide}")
                limits[oxide] = (float(nlo), float(nhi))
            else:
                limits[oxide] = (lo, hi)

        st.divider()
        st.subheader("📂 بيانات التدريب")
        uploaded = st.file_uploader(
            "ارفع ملف CSV (اختياري)",
            type=["csv"],
            help="الأعمدة المطلوبة تشمل الأكاسيد، النعومة، الجبس، الكلنكر، الإضافة المعدنية، والمقاومة للأعمار 2، 7، 28 يوم.",
        )
        if uploaded is not None and uploaded.size > 10 * 1024 * 1024:
            st.error("حجم الملف كبير جداً. الحد الأقصى 10 ميغابايت.")
            uploaded = None

        if "trained_models" not in st.session_state:
            st.session_state.trained_models = None
            st.session_state.metrics = None

        if uploaded is not None:
            try:
                df_train = pd.read_csv(uploaded)
                st.success(f"تم تحميل {len(df_train)} صفاً.")
                trained = train_models(df_train)
                if trained is None:
                    st.warning("الملف لا يحتوي الأعمدة المطلوبة.")
                else:
                    st.session_state.trained_models, st.session_state.metrics = trained
            except Exception as e:
                st.error(f"خطأ في قراءة الملف: {e}")

        if st.button("مسح النموذج المدرب"):
            st.session_state.trained_models = None
            st.session_state.metrics = None
            st.cache_resource.clear()
            st.success("تم مسح النموذج.")

        st.divider()
        st.subheader("📋 سجل العينات")
        if "history" not in st.session_state:
            st.session_state.history = []
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("حذف السجل"):
                st.session_state.history = []
                if HISTORY_FILE.exists():
                    HISTORY_FILE.unlink()
                st.experimental_rerun()
        else:
            st.info("لا عينات محفوظة.")

    # ---------- Main form ----------
    with st.form("sample_form"):
        st.subheader("🧪 إدخال العينة")
        sample_name = st.text_input("اسم العينة", value="Sample-001")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cao = st.number_input("CaO (%)", min_value=0.0, value=63.0, step=0.1)
            sio2 = st.number_input("SiO₂ (%)", min_value=0.0, value=21.5, step=0.1)
        with col2:
            al2o3 = st.number_input("Al₂O₃ (%)", min_value=0.0, value=5.2, step=0.1)
            fe2o3 = st.number_input("Fe₂O₃ (%)", min_value=0.0, value=3.1, step=0.1)
        with col3:
            mgo = st.number_input("MgO (%)", min_value=0.0, value=2.1, step=0.1)
            so3 = st.number_input("SO₃ (%)", min_value=0.0, value=2.8, step=0.1)
        with col4:
            na2o = st.number_input("Na₂O (%)", min_value=0.0, value=0.3, step=0.01, format="%.2f")
            k2o = st.number_input("K₂O (%)", min_value=0.0, value=0.6, step=0.01, format="%.2f")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            fineness = st.number_input("النعومة (m²/kg)", min_value=0.0, value=320.0, step=5.0)
        with col6:
            gypsum = st.number_input("الجبس (%)", min_value=0.0, value=4.5, step=0.1)
        with col7:
            clinker = st.number_input("الكلنكر (%)", min_value=0.0, value=95.0, step=0.1)
        with col8:
            mineral_addition = st.number_input("الإضافة المعدنية (%)", min_value=0.0, value=5.0, step=0.1)

        submitted = st.form_submit_button("🔍 تحليل العينة", type="primary")

    if submitted:
        ox = OxideInput(
            cao=float(cao),
            sio2=float(sio2),
            al2o3=float(al2o3),
            fe2o3=float(fe2o3),
            mgo=float(mgo),
            so3=float(so3),
            na2o=float(na2o),
            k2o=float(k2o),
        ).to_dict()

        idx = calc_indices(ox)
        label, notes = classify_sample(ox, idx, limits)

        # Decide on strength source
        models = st.session_state.trained_models
        metrics = st.session_state.metrics
        if models is not None:
            sample_row = pd.DataFrame([{
                **ox,
                "Fineness": float(fineness),
                "Gypsum": float(gypsum),
                "Clinker": float(clinker),
                "Mineral_Addition": float(mineral_addition),
                "LSF_Input": float(idx["LSF"]),
                "SM_Input": float(idx["SM"]),
                "AM_Input": float(idx["AM"]),
            }])
            strengths = predict_strengths_from_model(models, sample_row)
            model_mode = True
        else:
            strengths = heuristic_strengths(
                ox, idx, float(fineness), float(gypsum),
                float(clinker), float(mineral_addition),
            )
            model_mode = False

        # Save to session history
        record = {
            "Sample": sample_name,
            **ox,
            "Fineness": float(fineness),
            "Gypsum": float(gypsum),
            "Clinker": float(clinker),
            "Mineral_Addition": float(mineral_addition),
            **idx,
            **strengths,
            "Assessment": label,
        }
        st.session_state.history.append(record)

        # Optional: persist to CSV
        try:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

        # ------- Display results -------
        st.divider()
        st.subheader("📊 النتائج")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("التقييم", label)
        m2.metric("LSF", format_float(idx["LSF"]))
        m3.metric("LSA", format_float(idx["LSA"]))
        m4.metric("SM", format_float(idx["SM"]))
        m5.metric("AM", format_float(idx["AM"]))

        left, right = st.columns(2)
        with left:
            st.markdown("#### 💪 المقاومة المتوقعة")
            src = "نموذج AI مدرّب" if model_mode else "تقدير تقريبي (heuristic)"
            st.caption(f"المنهجية: {src}")
            res_df = pd.DataFrame([
                {"العمر": "2 يوم", "المقاومة (MPa)": strengths["Strength_2d"]},
                {"العمر": "7 يوم", "المقاومة (MPa)": strengths["Strength_7d"]},
                {"العمر": "28 يوم", "المقاومة (MPa)": strengths["Strength_28d"]},
            ])
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            st.bar_chart(res_df.set_index("العمر"))

            # Download button
            st.download_button(
                "📥 تحميل النتائج CSV",
                data=pd.DataFrame([record]).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sample_name}_analysis.csv",
                mime="text/csv",
            )

        with right:
            st.markdown("#### 📝 الملاحظات")
            for note in notes:
                st.write(f"- {note}")

        st.markdown("#### 🧪 القيم الكيميائية والمشتقة")
        st.dataframe(pd.DataFrame([ox | idx]), use_container_width=True, hide_index=True)

        if model_mode and metrics:
            st.markdown("#### 📈 أداء النموذج")
            met_df = pd.DataFrame(metrics).T
            st.dataframe(met_df, use_container_width=True)

    # Footer
    st.divider()
    with st.expander("ℹ️ ملاحظات فنية"):
        st.markdown(
            """
            - تم بناء النموذج باستخدام `GradientBoostingRegressor` أو `XGBoost` (إن وُجد) مع معالجة أوتوماتيكية للقيم المفقودة والقياس المعياري.
            - الأداء يتحسن كثيراً عند توفير بيانات تاريخية حقيقية من المصنع.
            - حدود القبول يمكن تخصيصها من الشريط الجانبي.
            - للاستخدام الإنتاجي يُنصح بإضافة طبقة مصادقة وتشغيل التطبيق على الشبكة الداخلية.
            """
        )

if __name__ == "__main__":
    main()
