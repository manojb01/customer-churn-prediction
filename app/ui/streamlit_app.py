import os
import io
import joblib
import pandas as pd
import streamlit as st
from typing import List, Any, Dict
import requests

# Import from src package
from models.ml_pipeline import MLPipeline

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

RAW_COLUMNS = [
    "CustomerID",
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

CATEGORICAL_DEFAULTS = {
    "PreferredLoginDevice": ["Mobile Phone", "Computer", "Phone"],
    "PreferredPaymentMode": ["Credit Card", "Debit Card", "UPI", "Cash on Delivery"],
    "Gender": ["Male", "Female"],
    "PreferedOrderCat": ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
}

def load_from_mlflow(tracking_uri: str, experiment_name: str, model_name: str = "customer_churn_model"):
    """Load model and transformers from MLflow Model Registry"""
    if not MLFLOW_AVAILABLE:
        st.error("MLflow not available in this environment")
        return None, None, [], {}
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Load directly from Model Registry
        st.info(f"🔍 Loading '{model_name}' from MLflow Model Registry...")
        
        try:
            # Check if model is registered
            registered_models = client.search_registered_models(f"name='{model_name}'")
            if not registered_models:
                st.error(f"❌ Model '{model_name}' not found in Model Registry")
                st.info("Run training pipeline to register the model")
                return None, None, [], {}
            
            st.info(f"✓ Found registered model: {model_name}")
            
            # Get the latest version
            versions = client.search_model_versions(f"name='{model_name}'", order_by=["version_number DESC"], max_results=1)
            if not versions:
                st.error("❌ No versions found for model")
                return None, None, [], {}
            
            latest_version = versions[0]
            registry_run_id = latest_version.run_id
            st.info(f"   Loading version {latest_version.version} from run {registry_run_id}")
            
            # Load artifacts via HTTP API
            tracking_base = tracking_uri.rstrip('/')
            list_url = f"{tracking_base}/api/2.0/mlflow/artifacts/list"
            
            # Get artifacts list
            response = requests.get(list_url, params={"run_id": registry_run_id, "path": "best_model"})
            response.raise_for_status()
            artifacts_data = response.json()
            files = artifacts_data.get('files', [])
            
            # Find model and transformers
            model_pkl = next((f for f in files if f['path'].endswith('_model.pkl')), None)
            transformer_pkl = next((f for f in files if 'transformer' in f['path'].lower() and f['path'].endswith('.pkl')), None)
            
            if not model_pkl:
                st.error("❌ Model pickle file not found in artifacts")
                return None, None, [], {}
            
            if not transformer_pkl:
                st.error("❌ Transformer pickle file not found in artifacts")
                return None, None, [], {}
            
            # Download model
            st.info(f"   Downloading model ({model_pkl.get('file_size', 0) // 1024 // 1024}MB)...")
            model_url = f"{tracking_base}/get-artifact?path={model_pkl['path']}&run_uuid={registry_run_id}"
            model_response = requests.get(model_url, timeout=60)
            model_response.raise_for_status()
            
            model = joblib.load(io.BytesIO(model_response.content))
            st.success(f"✓ Model loaded: {type(model).__name__}")
            
            # Download transformers
            st.info("   Downloading transformers...")
            trans_url = f"{tracking_base}/get-artifact?path={transformer_pkl['path']}&run_uuid={registry_run_id}"
            trans_response = requests.get(trans_url, timeout=30)
            trans_response.raise_for_status()
            
            transformers = joblib.load(io.BytesIO(trans_response.content))
            pipeline = MLPipeline()
            pipeline.transformers = transformers
            selected_features = transformers.get("selected_features", [])
            st.success(f"✓ Transformers loaded: {len(selected_features)} features")
            
            # Return everything
            run_info = {
                'run_name': latest_version.name,
                'run_id': registry_run_id,
                'model_version': latest_version.version,
                'start_time': 'from_registry'
            }
            return pipeline, model, selected_features, run_info
            
        except Exception as e:
            st.error(f"❌ Failed to load from MLflow: {str(e)[:200]}")
            import traceback
            with st.expander("🐛 Debug traceback"):
                st.code(traceback.format_exc())
            return None, None, [], {}
        

        
    except Exception as e:
        st.error(f"MLflow client error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, [], {}

def prepare_features(pipeline: MLPipeline, model: Any, selected_features: List[str], df: pd.DataFrame) -> pd.DataFrame:
    feature_set = pipeline.engineer_features(df, fit=False, target_col=None)
    feats = feature_set.features
    used_cols = [c for c in selected_features if c in feats.columns] or feats.columns.tolist()
    return feats[used_cols]

st.set_page_config(page_title="Churn Prediction UI", layout="wide")
st.title("Customer Churn Prediction")

with st.sidebar:
    st.header("🤖 MLflow Model Loader")
    st.caption("Loads model from MLflow Model Registry")
    tracking_uri = st.text_input("MLflow Tracking URI", value=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    model_name = st.text_input("Model Name", value=os.getenv("MLFLOW_MODEL_NAME", "customer_churn_model"))
    
    st.divider()
    col1, col2 = st.columns(2)
    load_button = col1.button("🔄 Load Latest Model", type="primary", use_container_width=True)
    clear_cache = col2.button("🗑️ Clear Cache", use_container_width=True)
    if clear_cache:
        st.cache_resource.clear()
        st.session_state.clear()
        st.success("Cache cleared")
        st.rerun()
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []

if load_button:
    with st.spinner("🔄 Loading latest model from MLflow..."):
        # Clear cache to ensure we get the latest model
        st.cache_resource.clear()
        result = load_from_mlflow(tracking_uri, None, model_name)
        if result and len(result) == 4:
            p, m, sf, run_info = result
            st.session_state.pipeline = p
            st.session_state.model = m
            st.session_state.selected_features = sf
            st.session_state.run_name = run_info.get('run_name', 'Unknown')
            st.session_state.run_id = run_info.get('run_id', 'Unknown')
        else:
            p, m, sf = result if result else (None, None, [])
            st.session_state.pipeline = p
            st.session_state.model = m
            st.session_state.selected_features = sf
            
        if m:
            st.success("✅ Latest model loaded successfully from MLflow!")
            st.balloons()
        else:
            st.error("❌ Failed to load model from MLflow. Check the debug panel for details.")

pipeline = st.session_state.pipeline
model = st.session_state.model
selected_features = st.session_state.selected_features

if not model or pipeline is None:
    st.warning("⚠️ Model not loaded. Click '🔄 Load Latest Model' in the sidebar to get started.")
else:
    run_name = st.session_state.get('run_name', 'Unknown')
    run_id = st.session_state.get('run_id', 'N/A')
    
    # Show current model info in an info box
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.success(f"✅ **Model:** {run_name}")
        col2.info(f"📊 **Features:** {len(selected_features)}")
        if col3.button("🔄 Reload"):
            st.cache_resource.clear()
            st.rerun()
    
    st.divider()
    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab_single:
        st.subheader("Single Customer Input")
        cols1 = st.columns(3)
        cols2 = st.columns(3)
        cols3 = st.columns(3)

        # Numeric inputs
        Tenure = cols1[0].number_input("Tenure", min_value=0, value=12)
        CityTier = cols1[1].number_input("CityTier", min_value=1, max_value=3, value=2)
        WarehouseToHome = cols1[2].number_input("WarehouseToHome", min_value=0.0, value=10.0)

        HourSpendOnApp = cols2[0].number_input("HourSpendOnApp", min_value=0.0, value=2.0)
        NumberOfDeviceRegistered = cols2[1].number_input("NumberOfDeviceRegistered", min_value=0, value=2)
        SatisfactionScore = cols2[2].number_input("SatisfactionScore", min_value=1.0, max_value=5.0, value=3.0)

        NumberOfAddress = cols3[0].number_input("NumberOfAddress", min_value=0, value=1)
        Complain = cols3[1].selectbox("Complain", options=[0,1], index=0)
        OrderAmountHikeFromlastYear = cols3[2].number_input("OrderAmountHikeFromlastYear", value=0.0)

        cols4 = st.columns(3)
        CouponUsed = cols4[0].number_input("CouponUsed", min_value=0, value=0)
        OrderCount = cols4[1].number_input("OrderCount", min_value=0, value=1)
        DaySinceLastOrder = cols4[2].number_input("DaySinceLastOrder", min_value=0, value=7)

        cols5 = st.columns(3)
        CashbackAmount = cols5[0].number_input("CashbackAmount", min_value=0.0, value=0.0)
        PreferredLoginDevice = cols5[1].selectbox("PreferredLoginDevice", CATEGORICAL_DEFAULTS["PreferredLoginDevice"]) 
        PreferredPaymentMode = cols5[2].selectbox("PreferredPaymentMode", CATEGORICAL_DEFAULTS["PreferredPaymentMode"]) 

        cols6 = st.columns(3)
        Gender = cols6[0].selectbox("Gender", CATEGORICAL_DEFAULTS["Gender"])
        PreferedOrderCat = cols6[1].selectbox("PreferedOrderCat", CATEGORICAL_DEFAULTS["PreferedOrderCat"]) 
        MaritalStatus = cols6[2].selectbox("MaritalStatus", CATEGORICAL_DEFAULTS["MaritalStatus"]) 

        if st.button("Predict", type="primary"):
            try:
                inp = {
                    "Tenure": Tenure,
                    "CityTier": CityTier,
                    "WarehouseToHome": WarehouseToHome,
                    "HourSpendOnApp": HourSpendOnApp,
                    "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
                    "SatisfactionScore": SatisfactionScore,
                    "NumberOfAddress": NumberOfAddress,
                    "Complain": int(Complain),
                    "OrderAmountHikeFromlastYear": OrderAmountHikeFromlastYear,
                    "CouponUsed": CouponUsed,
                    "OrderCount": OrderCount,
                    "DaySinceLastOrder": DaySinceLastOrder,
                    "CashbackAmount": CashbackAmount,
                    "PreferredLoginDevice": PreferredLoginDevice,
                    "PreferredPaymentMode": PreferredPaymentMode,
                    "Gender": Gender,
                    "PreferedOrderCat": PreferedOrderCat,
                    "MaritalStatus": MaritalStatus,
                }
                df = pd.DataFrame([inp])
                X = prepare_features(pipeline, model, selected_features, df)
                res = pipeline.predict(model, X)
                prob = float(res["churn_probability"][0])
                seg = str(res["risk_segment"][0])
                
                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Churn Probability", f"{prob:.1%}", delta=None)
                
                seg_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                col2.metric("Risk Segment", f"{seg_color.get(seg, '⚪')} {seg.upper()}")
                
                # Recommendation
                if seg == "high":
                    st.error("⚠️ **High Risk Customer** - Immediate retention action recommended")
                elif seg == "medium":
                    st.warning("📊 **Medium Risk Customer** - Monitor and engage proactively")
                else:
                    st.success("✅ **Low Risk Customer** - Continue regular engagement")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    with tab_batch:
        st.subheader("Batch Prediction (CSV or Excel)")
        st.caption("Upload a file with customer data. Required columns: " + ", ".join([c for c in RAW_COLUMNS if c != "CustomerID"][:5]) + "...")
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"]) 
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                st.info(f"Loaded {len(df)} rows from {uploaded.name}")
                
                missing = [c for c in RAW_COLUMNS if c not in df.columns and c != "CustomerID"]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                else:
                    with st.spinner("Making predictions..."):
                        X = prepare_features(pipeline, model, selected_features, df)
                        res = pipeline.predict(model, X)
                        out = pd.DataFrame({
                            "CustomerID": df.get("CustomerID", pd.Series(range(len(df)))).values,
                            "churn_probability": res["churn_probability"],
                            "prediction": res["predictions"],
                            "risk_segment": res["risk_segment"],
                        })
                    
                    st.success(f"✅ Predictions completed for {len(out)} customers")
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("High Risk", f"{res['high_risk_count']} ({res['high_risk_count']/len(out):.1%})")
                    col2.metric("Medium Risk", f"{res['medium_risk_count']} ({res['medium_risk_count']/len(out):.1%})")
                    col3.metric("Low Risk", f"{res['low_risk_count']} ({res['low_risk_count']/len(out):.1%})")
                    
                    st.dataframe(out, use_container_width=True)

                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download predictions CSV", data=csv, file_name="churn_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
