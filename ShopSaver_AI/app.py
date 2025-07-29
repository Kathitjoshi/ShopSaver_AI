import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import io
import scipy.stats

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap

# Set page config
st.set_page_config(
    page_title="ShopSaver AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and readability
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem; /* Increased padding */
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-card {
        background-color: #ffe6e6;
        padding: 1.5rem; /* Increased padding */
        border-radius: 10px;
        border-left: 5px solid #ff4444;
    }
    .safe-card {
        background-color: #e6ffe6;
        padding: 1.5rem; /* Increased padding */
        border-radius: 10px;
        border-left: 5px solid #44ff44;
    }
    /* Specific styling for text within cards */
    .metric-card h3, .warning-card h3, .safe-card h3 {
        font-size: 1.8rem; /* Larger heading */
        margin-bottom: 0.5rem;
    }
    .metric-card p, .warning-card p, .safe-card p {
        font-size: 1.1rem; /* Larger paragraph text */
        line-height: 1.5;
    }
    .metric-card p strong, .warning-card p strong, .safe-card p strong {
        font-size: 1.3rem; /* Even larger for key values */
        color: #000000; /* Ensure strong contrast */
    }
</style>
""", unsafe_allow_html=True)

class ShopSaverAI:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = None
        self.regressor = None
        self.feature_columns = None
    
    def load_and_preprocess_data(self, uploaded_file_content, column_mapping):
        """Loads and preprocesses the uploaded e-commerce transaction data based on column mapping."""
        st.info("Loading and preprocessing data based on your column mapping...")
        
        df = pd.read_csv(uploaded_file_content)
        df.columns = df.columns.str.strip() # Strip any whitespace from column names

        # Validate required columns before renaming
        # Explicitly define the 5 essential columns for core model functionality
        required_model_cols = {
            'user_id': column_mapping.get('user_id_col'),
            'event_time': column_mapping.get('event_time_col'),
            'product_id': column_mapping.get('product_id_col'),
            'price': column_mapping.get('unit_price_col'),
            'Quantity': column_mapping.get('quantity_col')
        }
        
        for model_col, csv_col in required_model_cols.items():
            if not csv_col or csv_col not in df.columns:
                st.error(f"Error: Required column for '{model_col}' (mapped from '{csv_col}') not found in your CSV or mapping is empty. Please check your column mapping.")
                return None # Indicate failure
        
        # Prepare for renaming
        rename_dict = {
            column_mapping['user_id_col']: 'user_id',
            column_mapping['event_time_col']: 'event_time',
            column_mapping['product_id_col']: 'product_id',
            column_mapping['unit_price_col']: 'price',
            column_mapping['quantity_col']: 'Quantity'
        }

        # Add optional columns to rename_dict if provided and exist
        if column_mapping.get('invoice_session_id_col') and column_mapping['invoice_session_id_col'] in df.columns:
            rename_dict[column_mapping['invoice_session_id_col']] = 'invoice_no_raw'
        if column_mapping.get('description_category_col') and column_mapping['description_category_col'] in df.columns:
            rename_dict[column_mapping['description_category_col']] = 'category_code'
        if column_mapping.get('country_col') and column_mapping['country_col'] in df.columns:
            rename_dict[column_mapping['country_col']] = 'country'
        
        df.rename(columns=rename_dict, inplace=True)

        # Initial Cleaning and Type Conversion (more generic)
        df.dropna(subset=['user_id', 'price', 'Quantity', 'event_time'], inplace=True)
        df = df[df['Quantity'] > 0] # Remove items with zero or negative quantity (common for returns/cancellations)
        
        df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
        df.dropna(subset=['event_time'], inplace=True) # Drop rows where date conversion failed
        
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(subset=['Quantity', 'price'], inplace=True) # Drop rows where quantity/price conversion failed
        
        df['user_id'] = df['user_id'].astype(str) # Keep user_id as string for generality
        
        # Add 'event_type' and 'user_session'
        df['event_type'] = 'purchase' # All rows are purchases in transactional datasets
        
        # Generate user_session using mapped invoice_no or a fallback
        if 'invoice_no_raw' in df.columns:
            df['user_session'] = df['invoice_no_raw'].astype(str) + '_' + df['user_id'].astype(str)
        else:
            df['user_session'] = df['user_id'].astype(str) + '_' + df['event_time'].dt.strftime('%Y%m%d%H%M%S') # Fallback session

        # Generate category_code using mapped description or a fallback
        if 'category_code' not in df.columns: # If description_category_col was not mapped
             df['category_code'] = 'unknown_category'
        
        # Determine 'is_discounted' based on mapping or heuristic
        discount_col_name = column_mapping.get('discount_col')
        if discount_col_name and discount_col_name in df.columns:
            st.info(f"Using '{discount_col_name}' from your CSV for explicit discount information.")
            # Convert discount column to boolean, handling various input types (0/1, T/F, 'yes'/'no')
            df['is_discounted'] = df[discount_col_name].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            st.warning("No explicit 'Discount Applied Column' provided or found. Using a heuristic for 'is_discounted'. For better accuracy, please provide a direct discount column if available in your data.")
            # Fallback heuristic: mark as 'discounted' if price is significantly below median for that product
            temp_df = df.copy()
            product_median_price = temp_df.groupby('product_id')['price'].transform('median').replace(0, np.nan)
            temp_df['is_discounted'] = ((temp_df['price'] < product_median_price * 0.8) & (temp_df['price'] > 0)).fillna(False)
            df['is_discounted'] = temp_df['is_discounted']
        
        df['brand'] = 'unknown' # Placeholder for brand. Real brand extraction would be complex from generic CSV.
        
        return df
    
    def engineer_features(self, df):
        """Create features for ML models"""
        
        # Sort by user and time
        df = df.sort_values(['user_id', 'event_time'])
        
        user_features = []
        
        # Get the overall max event time for recency calculation (end of data period)
        overall_max_event_time = df['event_time'].max()
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].copy()
            
            # Basic statistics
            total_events = len(user_data)
            unique_sessions = user_data['user_session'].nunique()
            unique_products = user_data['product_id'].nunique()
            unique_categories = user_data['category_code'].nunique()
            
            # Event type distribution
            event_counts = user_data['event_type'].value_counts()
            views = event_counts.get('view', 0)
            carts = event_counts.get('cart', 0)
            purchases = event_counts.get('purchase', 0)
            
            # Time-based features
            if not user_data['event_time'].empty:
                time_span = (user_data['event_time'].max() - user_data['event_time'].min()).days + 1
                recency_days = (overall_max_event_time - user_data['event_time'].max()).days
                customer_lifetime_days = (user_data['event_time'].max() - user_data['event_time'].min()).days
            else:
                time_span = 0
                recency_days = -1 # Indicates no activity or very old
                customer_lifetime_days = 0

            events_per_day = total_events / time_span if time_span > 0 else 0
            
            # Session behavior
            avg_events_per_session = total_events / unique_sessions if unique_sessions > 0 else 0
            
            # Conversion rates (will be skewed as only purchases are present in this type of data)
            view_to_cart_rate = carts / views if views > 0 else 0
            cart_to_purchase_rate = purchases / carts if carts > 0 else 0
            overall_conversion_rate = purchases / total_events if total_events > 0 else 0
            
            # Price and discount analysis
            purchase_data = user_data[user_data['event_type'] == 'purchase']
            
            if len(purchase_data) > 0:
                avg_order_value = purchase_data['price'].mean()
                total_quantity_purchased = purchase_data['Quantity'].sum()
                total_spend = (purchase_data['price'] * purchase_data['Quantity']).sum()

                discount_purchases = purchase_data['is_discounted'].sum()
                discount_dependency = discount_purchases / len(purchase_data) if len(purchase_data) > 0 else 0
                
                # Price behavior
                price_std = purchase_data['price'].std() if len(purchase_data['price']) > 1 else 0
                min_price = purchase_data['price'].min()
                max_price = purchase_data['price'].max()

                avg_quantity_per_purchase = total_quantity_purchased / purchases if purchases > 0 else 0
                avg_price_per_item = total_spend / total_quantity_purchased if total_quantity_purchased > 0 else 0
                
                # Time between purchases
                if len(purchase_data) > 1:
                    purchase_times = purchase_data['event_time'].sort_values()
                    time_diffs_seconds = purchase_times.diff().dt.total_seconds().dropna()
                    if not time_diffs_seconds.empty:
                        avg_time_between_purchases = time_diffs_seconds.mean() / 3600
                    else:
                        avg_time_between_purchases = 0
                else:
                    avg_time_between_purchases = 0
            else:
                avg_order_value = 0
                total_quantity_purchased = 0
                total_spend = 0
                discount_dependency = 0
                price_std = 0
                min_price = 0
                max_price = 0
                avg_quantity_per_purchase = 0
                avg_price_per_item = 0
                avg_time_between_purchases = 0
            
            # Category diversity
            category_diversity = unique_categories / total_events if total_events > 0 else 0

            num_unique_brands = user_data['brand'].nunique()
            num_unique_countries = user_data['country'].nunique() if 'country' in user_data.columns else 1

            # NEW FEATURES
            # Most Common Purchase Hour
            if not user_data['event_time'].empty:
                purchase_hours = user_data['event_time'].dt.hour
                most_common_purchase_hour = purchase_hours.mode()[0] if not purchase_hours.empty else -1
            else:
                most_common_purchase_hour = -1

            # Most Common Purchase Day of Week (Monday=0, Sunday=6)
            if not user_data['event_time'].empty:
                purchase_days_of_week = user_data['event_time'].dt.dayofweek
                most_common_purchase_day_of_week = purchase_days_of_week.mode()[0] if not purchase_days_of_week.empty else -1
            else:
                most_common_purchase_day_of_week = -1

            # Category Entropy
            if not user_data['category_code'].empty and total_events > 0:
                category_counts = user_data['category_code'].value_counts(normalize=True)
                category_entropy = scipy.stats.entropy(category_counts, base=2) # Base 2 for bits
            else:
                category_entropy = 0

            # --- RELAXED DEFINITION FOR is_discount_abuser to get more diverse predictions ---
            # This makes the definition less strict, potentially identifying more 'abusers'
            # and leading to more varied prediction probabilities.
            is_discount_abuser = (
                discount_dependency >= 0.5 and # Lowered from 0.7
                purchases >= 3 and          # Lowered from 5
                avg_time_between_purchases <= 72 # Increased from 24 (allowing less frequent, but still engaged)
            )
            
            monthly_spend = total_spend # Target for regression model, represents total spend in observed period
            
            user_features.append({
                'user_id': user_id,
                'total_events': total_events,
                'unique_sessions': unique_sessions,
                'unique_products': unique_products,
                'unique_categories': unique_categories,
                'views': views,
                'carts': carts,
                'purchases': purchases,
                'events_per_day': events_per_day,
                'avg_events_per_session': avg_events_per_session,
                'view_to_cart_rate': view_to_cart_rate,
                'cart_to_purchase_rate': cart_to_purchase_rate,
                'overall_conversion_rate': overall_conversion_rate,
                'avg_order_value': avg_order_value,
                'total_spend': total_spend,
                'discount_dependency': discount_dependency,
                'price_std': price_std,
                'min_price': min_price,
                'max_price': max_price,
                'avg_time_between_purchases': avg_time_between_purchases,
                'category_diversity': category_diversity,
                'recency_days': recency_days,
                'customer_lifetime_days': customer_lifetime_days,
                'avg_quantity_per_purchase': avg_quantity_per_purchase,
                'avg_price_per_item': avg_price_per_item,
                'total_quantity_purchased': total_quantity_purchased,
                'num_unique_brands': num_unique_brands,
                'num_unique_countries': num_unique_countries,
                'most_common_purchase_hour': most_common_purchase_hour,
                'most_common_purchase_day_of_week': most_common_purchase_day_of_week,
                'category_entropy': category_entropy,
                'is_discount_abuser': is_discount_abuser,
                'monthly_spend': monthly_spend
            })
        
        return pd.DataFrame(user_features)
    
    def train_models(self, features_df):
        """Train classification and regression models"""
        
        feature_cols = [col for col in features_df.columns
                        if col not in ['user_id', 'is_discount_abuser', 'monthly_spend']]
        
        X = features_df[feature_cols]
        y_class = features_df['is_discount_abuser']
        y_reg = features_df['monthly_spend']
        
        # Convert all feature columns to numeric, coercing errors to NaN
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0) # Then fill remaining NaNs with 0

        self.feature_columns = feature_cols
        
        if y_class.nunique() > 1:
            X_train, X_test, y_class_train, y_class_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
        else:
            st.warning("Only one class found for 'is_discount_abuser'. Cannot stratify split for classification.")
            X_train, X_test, y_class_train, y_class_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42
            )
        
        _, _, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.classifier = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.08,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.classifier.fit(X_train_scaled, y_class_train)
        
        self.regressor = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.08,
            random_state=42
        )
        self.regressor.fit(X_train_scaled, y_reg_train)
        
        class_pred = self.classifier.predict(X_test_scaled)
        reg_pred = self.regressor.predict(X_test_scaled)
        
        class_accuracy = (class_pred == y_class_test).mean()
        
        reg_mse = mean_squared_error(y_reg_test, reg_pred)
        reg_r2 = r2_score(y_reg_test, reg_pred)
        
        return {
            'classification_accuracy': class_accuracy,
            'regression_mse': reg_mse,
            'regression_r2': reg_r2,
            'feature_importance_class': dict(zip(feature_cols, self.classifier.feature_importances_)),
            'feature_importance_reg': dict(zip(feature_cols, self.regressor.feature_importances_))
        }
    
    def predict_user(self, user_features):
        """Predict for a single user"""
        if self.classifier is None or self.regressor is None:
            st.warning("Models not trained. Please train models first.")
            return 0.5, 0.0, []
        
        if len(user_features) != len(self.feature_columns):
            st.error(f"Prediction input mismatch: Expected {len(self.feature_columns)} features, got {len(user_features)}. Please ensure all input fields are correctly filled after training.")
            return 0.5, 0.0, []

        X = np.array(user_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        abuse_prob = self.classifier.predict_proba(X_scaled)[0][1]
        spend_pred = self.regressor.predict(X_scaled)[0]
        
        # SHAP explanation
        explainer = shap.TreeExplainer(self.classifier)
        shap_output = explainer.shap_values(X_scaled)

        shap_values_raw = None
        # Handle SHAP output based on its structure for binary classification
        if isinstance(shap_output, list) and len(shap_output) > 1:
            shap_values_raw = shap_output[1] # For binary classification, index 1 is usually the positive class
        elif isinstance(shap_output, np.ndarray):
            if shap_output.ndim == 2 and shap_output.shape[0] == 1:
                shap_values_raw = shap_output[0] # For a single prediction, might be a 2D array of 1 row
            elif shap_output.ndim == 1:
                shap_values_raw = shap_output # For a single prediction, might be a 1D array directly
        
        # Validate that shap_values_raw is a valid numpy array before proceeding
        if shap_values_raw is None or not isinstance(shap_values_raw, np.ndarray) or not np.issubdtype(shap_values_raw.dtype, np.number):
            st.warning("Failed to get valid SHAP values for display. This can happen with very simple models or uniform predictions.")
            return abuse_prob, spend_pred, [] # Return empty for contributions

        # Ensure shap_values_raw is flattened to 1D if it's not already
        if shap_values_raw.ndim > 1:
            shap_values_raw = shap_values_raw.flatten()

        shap_values_abs = np.abs(shap_values_raw)
        
        # Get top contributing features
        if self.feature_columns and len(shap_values_raw) == len(self.feature_columns):
            feature_contributions = sorted(zip(self.feature_columns, shap_values_raw, shap_values_abs), key=lambda x: x[2], reverse=True)[:5]
        else:
            st.warning("Could not generate meaningful SHAP feature contributions due to feature list/SHAP value mismatch. Check your data and model training.")
            feature_contributions = [] # Return empty list if SHAP values are problematic

        return abuse_prob, spend_pred, feature_contributions

# Initialize the app
@st.cache_resource
def load_model():
    return ShopSaverAI()

def main():
    st.markdown('<h1 class="main-header">🛍️ ShopSaver AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Customer Discount Abuse Detector & Spend Predictor</p>', unsafe_allow_html=True)
    
    model = load_model()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "📊 Data Analysis",
        "🤖 Model Training",
        "🔍 User Prediction",
        "📈 Business Dashboard"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis(model)
    elif page == "🤖 Model Training":
        show_model_training(model)
    elif page == "🔍 User Prediction":
        show_prediction_page(model)
    elif page == "📈 Business Dashboard":
        show_business_dashboard(model)

def show_home_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## Welcome to ShopSaver AI! 🎯
        
        This application helps e-commerce businesses identify customers who abuse discount codes
        and predict future customer spending patterns.
        
        ### 🔧 Features:
        - **Discount Abuse Detection**: Identify customers who only purchase with discounts
        - **Spend Prediction**: Forecast monthly customer spending
        - **SHAP Explainability**: Understand why customers are flagged
        - **Real-time Analysis**: Upload data and get instant insights
        
        ### 📝 How it Works:
        1. **Data Processing**: Analyze customer transaction data.
        2. **Feature Engineering**: Create detailed behavioral indicators including RFM and temporal patterns.
        3. **ML Prediction**: Use enhanced XGBoost and Gradient Boosting models with optimized hyperparameters.
        4. **Business Insights**: Get actionable recommendations.
        
        ### 🚀 Get Started:
        Navigate to different sections using the sidebar to explore the functionality!
        """)

def show_data_analysis(model):
    st.header("📊 Data Analysis & Feature Engineering")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        current_file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"

        if 'raw_data_identifier' not in st.session_state or st.session_state.get('raw_data_identifier') != current_file_identifier:
            st.session_state['uploaded_file_object'] = uploaded_file # Store the file object temporarily
            st.session_state['raw_data_identifier'] = current_file_identifier
            # Reset dependent states when a new file is uploaded
            st.session_state['raw_data_loaded'] = False
            if 'raw_data' in st.session_state: del st.session_state['raw_data']
            if 'features_df' in st.session_state: del st.session_state['features_df']
            if 'model_trained' in st.session_state: del st.session_state['model_trained']
            if 'model_results' in st.session_state: del st.session_state['model_results']
            
            st.info(f"File '{uploaded_file.name}' uploaded. Please configure column mapping below.")
            try:
                # Use a BytesIO buffer to read the first few lines without consuming the stream
                uploaded_file.seek(0)
                temp_df_head = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')), nrows=5).columns.tolist()
                st.write("**Columns detected in your CSV (first 5 rows):**")
                st.code(temp_df_head)
                uploaded_file.seek(0) # Reset file pointer for full read later
            except Exception as e:
                st.error(f"Error reading uploaded CSV for preview: {e}. Please ensure it's a valid CSV.")
                st.stop()
        else:
            # If the same file is re-selected, assume it's already processed or ready for mapping
            st.info(f"File '{uploaded_file.name}' already uploaded. Review column mapping below.")
            uploaded_file = st.session_state['uploaded_file_object'] # Retrieve the stored file object
    else:
        # If no file is uploaded, clear all related session states
        if 'raw_data_identifier' in st.session_state: del st.session_state['raw_data_identifier']
        if 'uploaded_file_object' in st.session_state: del st.session_state['uploaded_file_object']
        if 'raw_data_loaded' in st.session_state: del st.session_state['raw_data_loaded']
        if 'raw_data' in st.session_state: del st.session_state['raw_data']
        if 'features_df' in st.session_state: del st.session_state['features_df']
        if 'model_trained' in st.session_state: del st.session_state['model_trained']
        if 'model_results' in st.session_state: del st.session_state['model_results']
        st.info("Please upload your CSV file to proceed.")
        return # Exit function if no file to process

    st.subheader("Configure Column Mapping")
    st.markdown("Please map your CSV's column names to the expected model fields. **All 5 core fields are required for the model to function.**")
    st.markdown("**(Note: For optional fields like 'Discount Applied' or 'Country', leave them blank if not present in your data; default assumptions will be used.)**")


    # Initialize column_mapping in session state if not present
    if 'column_mapping' not in st.session_state:
        # Default mapping for Online Retail dataset for convenience
        st.session_state['column_mapping'] = {
            'user_id_col': 'CustomerID',
            'event_time_col': 'InvoiceDate',
            'product_id_col': 'StockCode',
            'unit_price_col': 'UnitPrice',
            'quantity_col': 'Quantity', # Now explicitly required for robust spend prediction
            'invoice_session_id_col': 'InvoiceNo',
            'description_category_col': 'Description',
            'discount_col': '', # No direct discount column in Online Retail
            'country_col': 'Country' # New optional column
        }

    col_map_cols = st.columns(2)
    with col_map_cols[0]:
        st.session_state['column_mapping']['user_id_col'] = st.text_input("User ID Column (Required)", value=st.session_state['column_mapping']['user_id_col'], help="e.g., CustomerID")
        st.session_state['column_mapping']['event_time_col'] = st.text_input("Event Time Column (Required)", value=st.session_state['column_mapping']['event_time_col'], help="e.g., InvoiceDate (must be parseable by pandas as datetime)")
        st.session_state['column_mapping']['product_id_col'] = st.text_input("Product ID Column (Required)", value=st.session_state['column_mapping']['product_id_col'], help="e.g., StockCode")
        st.session_state['column_mapping']['unit_price_col'] = st.text_input("Unit Price Column (Required)", value=st.session_state['column_mapping']['unit_price_col'], help="e.g., UnitPrice (price per single item)")
    with col_map_cols[1]:
        st.session_state['column_mapping']['quantity_col'] = st.text_input("Quantity Column (Required)", value=st.session_state['column_mapping']['quantity_col'], help="e.g., Quantity (number of items in transaction). Critical for accurate spend calculation.", key="quantity_input")
        st.session_state['column_mapping']['invoice_session_id_col'] = st.text_input("Invoice/Session ID Column (Optional)", value=st.session_state['column_mapping']['invoice_session_id_col'], help="e.g., InvoiceNo. Used to group events into sessions. If empty, a fallback session ID is created.")
        st.session_state['column_mapping']['description_category_col'] = st.text_input("Description/Category Column (Optional)", value=st.session_state['column_mapping']['description_category_col'], help="e.g., Description. Used for category diversity. If empty, 'unknown_category' is used.")
        st.session_state['column_mapping']['discount_col'] = st.text_input("Discount Applied Column (Optional)", value=st.session_state['column_mapping']['discount_col'], help="e.g., 'IsDiscounted', 'DiscountValue'. If your CSV has a column explicitly stating if a discount was applied (e.g., True/False, 1/0, discount amount). Leave blank to use a heuristic.", key="discount_input")
        st.session_state['column_mapping']['country_col'] = st.text_input("Country Column (Optional)", value=st.session_state['column_mapping']['country_col'], help="e.g., Country. Used for geographical features. If empty, 'num_unique_countries' defaults to 1.", key="country_input")

    if st.button("Apply Column Mapping & Load Data", type="primary"):
        if st.session_state.get('uploaded_file_object') is None:
            st.warning("Please upload a CSV file first.")
            return

        with st.spinner("Applying mapping and loading data..."):
            st.session_state['uploaded_file_object'].seek(0) # Reset file pointer
            df = model.load_and_preprocess_data(st.session_state['uploaded_file_object'], st.session_state['column_mapping'])
            
            if df is not None:
                st.session_state['raw_data'] = df
                st.session_state['raw_data_loaded'] = True
                st.session_state['last_applied_mapping'] = st.session_state['column_mapping'].copy() # Store applied mapping
                st.success("Data loaded and preprocessed successfully with applied mapping!")
                st.rerun()
            else:
                st.session_state['raw_data_loaded'] = False
                return
    
    if st.session_state.get('raw_data_loaded', False) and 'raw_data' in st.session_state:
        df = st.session_state['raw_data']
        
        st.subheader("Preprocessed Data Overview")
        st.write(f"Dataset Shape: {df.shape}")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events (Purchases)", f"{len(df):,}")
        with col2:
            st.metric("Unique Users", f"{df['user_id'].nunique():,}")
        with col3:
            st.metric("Unique Products", f"{df['product_id'].nunique():,}")
        with col4:
            if 'event_time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['event_time']):
                st.metric("Date Range", f"{df['event_time'].dt.date.nunique()} days")
            else:
                st.metric("Date Range", "N/A (check event_time format)")
        
        st.subheader("Event Distribution")
        event_counts = df['event_type'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=event_counts.index, y=event_counts.values,
                         title="Event Type Distribution")
            fig.update_layout(xaxis_title="Event Type", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(values=event_counts.values, names=event_counts.index,
                         title="Event Type Proportion")
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Generate Features"):
            with st.spinner("Engineering features..."):
                features_df = model.engineer_features(df)
                st.session_state['features_df'] = features_df
            
            st.success("Features generated successfully!")
        
        if 'features_df' in st.session_state:
            features_df = st.session_state['features_df']
            
            st.subheader("Feature Overview")
            st.dataframe(features_df.head(), use_container_width=True)
            
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['user_id']]
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_features:
                cols = st.columns(2)
                for i, feature in enumerate(selected_features):
                    with cols[i % 2]:
                        fig = px.histogram(features_df, x=feature,
                                           title=f"Distribution of {feature}")
                        st.plotly_chart(fig, use_container_width=True)


def show_model_training(model):
    st.header("🤖 Model Training & Evaluation")
    
    if 'features_df' not in st.session_state:
        st.warning("Please upload data and generate features in the Data Analysis section first!")
        return
    
    features_df = st.session_state['features_df']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Target Variable Analysis")
        
        abuse_counts = features_df['is_discount_abuser'].value_counts()
        labels = ['Non-Abuser', 'Abuser']
        values = [abuse_counts.get(False, 0), abuse_counts.get(True, 0)]

        fig = px.bar(x=labels, y=values,
                     title="Discount Abuse Distribution",
                     color=labels,
                     color_discrete_map={'Non-Abuser': 'lightblue', 'Abuser': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        
        total_users = len(features_df)
        abuser_count = abuse_counts.get(True, 0)
        abuse_rate = abuser_count / total_users * 100 if total_users > 0 else 0
        st.write(f"**Abuse Rate**: {abuse_rate:.1f}%")
    
    with col2:
        st.subheader("Training Controls")
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training ML models..."):
                results = model.train_models(features_df)
                st.session_state['model_results'] = results
                st.session_state['model_trained'] = True
            
            st.success("Models trained successfully!")
    
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        
        st.subheader("Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classification Accuracy", f"{results['classification_accuracy']:.3f}")
        with col2:
            st.metric("Regression R²", f"{results['regression_r2']:.3f}")
        with col3:
            st.metric("Regression MSE", f"{results['regression_mse']:.0f}")
        
        st.subheader("Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Classification Model**")
            class_importance = sorted(results['feature_importance_class'].items(),
                                      key=lambda x: x[1], reverse=True)[:10]
            
            fig = px.bar(x=[x[1] for x in class_importance],
                         y=[x[0] for x in class_importance],
                         orientation='h',
                         title="Top 10 Features - Classification")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Regression Model**")
            reg_importance = sorted(results['feature_importance_reg'].items(),
                                    key=lambda x: x[1], reverse=True)[:10]
            
            fig = px.bar(x=[x[1] for x in reg_importance],
                         y=[x[0] for x in reg_importance],
                         orientation='h',
                         title="Top 10 Features - Regression")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model):
    st.header("🔍 User Prediction & Analysis")
    
    if 'model_trained' not in st.session_state:
        st.warning("Please train the models first!")
        return
    
    st.subheader("Input User Behavior")
    
    feature_names = model.feature_columns

    user_input_features = {}
    
    cols = st.columns(3)
    
    if 'features_df' in st.session_state and not st.session_state['features_df'].empty:
        df_stats = st.session_state['features_df'].mean(numeric_only=True)
        df_stds = st.session_state['features_df'].std(numeric_only=True)
    else:
        df_stats = {}
        df_stds = {}
        
    general_defaults = {
        'total_events': 50, 'unique_sessions': 5, 'unique_products': 8, 'unique_categories': 3,
        'views': 0.0, 'carts': 0.0, 'purchases': 5.0,
        'events_per_day': 2.5, 'avg_events_per_session': 10.0,
        'view_to_cart_rate': 0.0, 'cart_to_purchase_rate': 0.0, 'overall_conversion_rate': 0.1,
        'avg_order_value': 75.0, 'total_spend': 375.0, 'discount_dependency': 0.4,
        'price_std': 25.0, 'min_price': 50.0, 'max_price': 100.0,
        'avg_time_between_purchases': 48.0, 'category_diversity': 0.1,
        'recency_days': 30.0, 'customer_lifetime_days': 100.0,
        'avg_quantity_per_purchase': 10.0, 'avg_price_per_item': 5.0,
        'total_quantity_purchased': 500.0, 'num_unique_brands': 1.0, 'num_unique_countries': 1.0,
        'most_common_purchase_hour': 12.0, 'most_common_purchase_day_of_week': 3.0, 'category_entropy': 2.0
    }

    for i, feature_name in enumerate(feature_names):
        with cols[i % 3]:
            default_value = df_stats.get(feature_name, general_defaults.get(feature_name, 0.0))
            
            if feature_name in ['discount_dependency', 'view_to_cart_rate', 'cart_to_purchase_rate', 'overall_conversion_rate', 'category_diversity']:
                user_input_features[feature_name] = st.slider(feature_name.replace('_', ' ').title(), 0.0, 1.0, float(default_value), key=f"input_{feature_name}")
            elif feature_name in ['most_common_purchase_hour', 'most_common_purchase_day_of_week']:
                min_val = 0
                max_val = 23 if feature_name == 'most_common_purchase_hour' else 6
                user_input_features[feature_name] = st.number_input(feature_name.replace('_', ' ').title(), min_value=min_val, max_value=max_val, value=int(default_value), key=f"input_{feature_name}")
            else:
                user_input_features[feature_name] = st.number_input(feature_name.replace('_', ' ').title(), min_value=0.0, value=float(default_value), key=f"input_{feature_name}")

    user_features_for_prediction = [user_input_features[f] for f in feature_names]

    if st.button("Analyze User", type="primary"):
        with st.spinner("Analyzing user behavior..."):
            abuse_prob, spend_pred, feature_contributions = model.predict_user(user_features_for_prediction)
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_message = ""
            if abuse_prob > 0.7:
                risk_level_html = "<div class='warning-card'>"
                risk_level_html += "<h3>⚠️ HIGH RISK USER</h3>"
                risk_message = "This user shows strong patterns of discount abuse behavior. **Immediate action is recommended.**"
            elif abuse_prob > 0.4:
                risk_level_html = "<div class='metric-card'>"
                risk_level_html += "<h3>⚡ MODERATE RISK USER</h3>"
                risk_message = "This user exhibits some patterns of discount abuse. **Monitoring is recommended.**"
            else:
                risk_level_html = "<div class='safe-card'>"
                risk_level_html += "<h3>✅ LOW RISK USER</h3>"
                risk_message = "This user shows normal purchasing behavior. **Engagement opportunities are high.**"
            
            risk_level_html += f"<p><strong>Abuse Probability: {abuse_prob:.1%}</strong></p>"
            risk_level_html += f"<p>{risk_message}</p>"
            risk_level_html += "</div>"
            st.markdown(risk_level_html, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 SPEND PREDICTION</h3>
                <p><strong>Predicted Monthly Spend: ${spend_pred:,.2f}</strong></p>
                <p>This forecast is based on the provided behavioral patterns.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Risk Assessment Gauge")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = abuse_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Abuse Risk (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Factors for Abuse Risk Prediction")
        if feature_contributions:
            shap_df = pd.DataFrame(feature_contributions, columns=['Feature', 'SHAP Value (Raw)', 'SHAP Value (Absolute)'])
            
            fig_shap = px.bar(shap_df.sort_values(by='SHAP Value (Absolute)', ascending=False), 
                              x='SHAP Value (Raw)', 
                              y='Feature', 
                              orientation='h',
                              color='SHAP Value (Raw)',
                              color_continuous_scale=px.colors.sequential.RdBu,
                              title="Top 5 Influencing Factors (Positive values increase risk, Negative decrease)")
            fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)
            st.info("Positive SHAP values indicate the feature's contribution to a higher abuse risk probability. Negative values indicate a contribution to lower risk.")
        else:
            st.info("No feature contributions available for display. This might occur if models are not yet trained, or if SHAP encounters issues with the specific input data (e.g., highly uniform predictions).")
            
        st.subheader("Business Recommendations")
        
        if abuse_prob > 0.7:
            st.error("""
            **Immediate Actions Required:**
            - Limit discount code access for this user
            - Review recent purchase history for patterns (e.g., small orders, multiple accounts)
            - Consider implementing purchase limits or temporary discount restrictions
            - Monitor for multiple account creation or suspicious activity
            """)
        elif abuse_prob > 0.4:
            st.warning("""
            **Monitoring Recommended:**
            - Track discount usage frequency and types of discounts used
            - Set alerts for unusual purchase patterns or sudden changes in behavior
            - Consider targeted retention campaigns that don't solely rely on deep discounts
            """)
        else:
            st.success("""
            **Engagement Opportunities:**
            - Offer loyalty program enrollment and exclusive non-discount benefits
            - Send personalized product recommendations based on past purchases
            - Consider premium service offerings or early access to new products
            - Focus on building long-term customer value
            """)

def show_business_dashboard(model):
    st.header("📈 Business Dashboard")

    if 'features_df' not in st.session_state or 'model_trained' not in st.session_state:
        st.warning("Please upload data and generate features, then train models in previous sections to see the dashboard.")
        return

    features_df = st.session_state['features_df']
    
    st.subheader("Overall Customer Segments")
    
    X_all = features_df[model.feature_columns].fillna(0)
    X_all_scaled = model.scaler.transform(X_all)
    
    features_df['predicted_abuse_prob'] = model.classifier.predict_proba(X_all_scaled)[:, 1]
    features_df['predicted_spend'] = model.regressor.predict(X_all_scaled)
    
    features_df['abuse_risk_level'] = pd.cut(features_df['predicted_abuse_prob'],
                                               bins=[0, 0.4, 0.7, 1.0], # Keep these bins for consistency
                                               labels=['Low Risk', 'Moderate Risk', 'High Risk'],
                                               right=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers Analyzed", f"{features_df['user_id'].nunique():,}")
    with col2:
        avg_predicted_spend = features_df['predicted_spend'].mean()
        st.metric("Avg Predicted Monthly Spend", f"${avg_predicted_spend:,.2f}")
    with col3:
        high_risk_count = (features_df['abuse_risk_level'] == 'High Risk').sum()
        st.metric("High-Risk Abusers", f"{high_risk_count:,} ({high_risk_count / len(features_df) * 100:.1f}%)")

    st.subheader("Discount Abuse Risk Distribution")
    risk_counts = features_df['abuse_risk_level'].value_counts().sort_index()
    fig_risk = px.bar(x=risk_counts.index, y=risk_counts.values,
                      color=risk_counts.index,
                      color_discrete_map={'Low Risk': 'lightgreen', 'Moderate Risk': 'yellow', 'High Risk': 'red'},
                      title="Customer Distribution by Abuse Risk Level")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("Predicted Monthly Spend Distribution")
    fig_spend = px.histogram(features_df, x='predicted_spend', nbins=50,
                             title="Distribution of Predicted Monthly Spend",
                             labels={'predicted_spend': 'Predicted Spend ($)'})
    st.plotly_chart(fig_spend, use_container_width=True)

    # NEW PLOT FOR DASHBOARD: Recency vs. Predicted Spend
    st.subheader("Customer Recency vs. Predicted Spend by Risk")
    fig_recency_spend = px.scatter(features_df, x='recency_days', y='predicted_spend',
                                   color='abuse_risk_level',
                                   color_discrete_map={'Low Risk': 'lightgreen', 'Moderate Risk': 'yellow', 'High Risk': 'red'},
                                   hover_name='user_id',
                                   title="Customer Recency vs. Predicted Monthly Spend",
                                   labels={'recency_days': 'Days Since Last Purchase (Lower is more recent)', 'predicted_spend': 'Predicted Spend ($)'})
    st.plotly_chart(fig_recency_spend, use_container_width=True)

    # NEW PLOT FOR DASHBOARD: Discount Dependency Distribution by Risk Level
    st.subheader("Discount Dependency Distribution by Abuse Risk Level")
    fig_discount_dist = px.histogram(features_df, x='discount_dependency', color='abuse_risk_level',
                                     marginal="box", # Show box plot for distribution
                                     color_discrete_map={'Low Risk': 'lightgreen', 'Moderate Risk': 'yellow', 'High Risk': 'red'},
                                     title="Distribution of Discount Dependency Across Risk Levels",
                                     labels={'discount_dependency': 'Discount Dependency (0-1)'})
    st.plotly_chart(fig_discount_dist, use_container_width=True)

    # NEW PLOT FOR DASHBOARD: Total Spend vs Discount Dependency
    st.subheader("Total Spend vs. Discount Dependency by Risk Level")
    fig_spend_discount_scatter = px.scatter(features_df, x='total_spend', y='discount_dependency',
                                            color='abuse_risk_level',
                                            color_discrete_map={'Low Risk': 'lightgreen', 'Moderate Risk': 'yellow', 'High Risk': 'red'},
                                            hover_name='user_id',
                                            title="Total Spend vs. Discount Dependency by Risk Level",
                                            labels={'total_spend': 'Total Historical Spend ($)', 'discount_dependency': 'Discount Dependency (0-1)'})
    st.plotly_chart(fig_spend_discount_scatter, use_container_width=True)


    st.subheader("Deep Dive into High-Risk Users")
    high_risk_df = features_df[features_df['abuse_risk_level'] == 'High Risk'].sort_values('predicted_abuse_prob', ascending=False)
    if not high_risk_df.empty:
        st.dataframe(high_risk_df[[
            'user_id', 'predicted_abuse_prob', 'discount_dependency', 'purchases', 
            'total_spend', 'avg_time_between_purchases', 'recency_days', 
            'customer_lifetime_days', 'avg_quantity_per_purchase', 'avg_price_per_item',
            'most_common_purchase_hour', 'most_common_purchase_day_of_week', 'category_entropy'
        ]].head(15), use_container_width=True)
    else:
        st.info("No high-risk users detected in the current dataset based on the model's prediction and thresholds.")

    st.markdown("""
    ---
    ### Business Insights & Actions:
    - **Identify potential fraud**: Focus on "High Risk" users to investigate suspicious activities.
    - **Optimize discount strategy**: Understand which customers are highly dependent on discounts and adjust offers.
    - **Personalize marketing**: Target high-spending, low-risk customers with premium offers.
    - **Customer Retention**: Identify customers with declining spend and proactively engage them.
    """)


if __name__ == '__main__':
    main()