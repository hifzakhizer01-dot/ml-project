import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Project - Classification & Regression",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    
    # Clean TotalCharges column - convert to numeric, handling empty strings/spaces
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill NaN values with 0 or median
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median() if df['TotalCharges'].notna().any() else 0)
    
    # Convert any other object columns that should be numeric
    for col in df.select_dtypes(include=['object']).columns:
        # Try to convert to numeric if possible
        if col not in ['customerID', 'Churn']:  # Skip ID and target columns
            try:
                # Check if column can be converted to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.5:  # If more than 50% are numeric
                    df[col] = numeric_series
            except:
                pass
    
    return df

@st.cache_data
def preprocess_data(df, target_col, task_type='classification'):
    """Preprocess data for ML models"""
    df_processed = df.copy()
    
    # Handle missing values
    if df_processed[target_col].dtype == 'object':
        df_processed[target_col] = df_processed[target_col].fillna(df_processed[target_col].mode()[0])
    else:
        df_processed[target_col] = df_processed[target_col].fillna(df_processed[target_col].median())
    
    # Drop customerID if exists (not useful for ML)
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop('customerID', axis=1)
    
    # Encode categorical variables
    le_dict = {}
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    # Encode target variable for classification
    if task_type == 'classification':
        le_target = LabelEncoder()
        df_processed[target_col] = le_target.fit_transform(df_processed[target_col])
        return df_processed, le_dict, le_target
    else:
        return df_processed, le_dict, None

def generate_profile_report(df):
    """Generate ydata-profiling report"""
    # Create a clean copy for profiling
    df_clean = df.copy()
    
    # Clean object columns - convert to string explicitly to avoid PyArrow issues
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Replace NaN/None with empty string, then convert to string
        df_clean[col] = df_clean[col].fillna('').astype(str)
        # Replace empty strings with 'Missing' for better profiling
        df_clean[col] = df_clean[col].replace('', 'Missing')
    
    # Generate profile - use minimal=False for full report, but handle errors gracefully
    try:
        # Try with full report first
        profile = ProfileReport(
            df_clean, 
            explorative=True, 
            minimal=False,
            title="Dataset Profile Report",
            dataset={"description": "ML Project Dataset"},
            # Disable problematic features that might cause PyArrow issues
            correlations=None  # Can set to None or specific correlation types
        )
    except Exception as e:
        # Fallback: try with minimal=True
        try:
            st.warning(f"Full profile mode encountered an issue. Using minimal mode.")
            profile = ProfileReport(
                df_clean, 
                explorative=True, 
                minimal=True,
                title="Dataset Profile Report",
                dataset={"description": "ML Project Dataset"}
            )
        except Exception as e2:
            # Last resort: very minimal report
            st.error(f"Error generating profile: {str(e2)}")
            raise e2
    
    return profile

def train_classification_model(X_train, X_test, y_train, y_test, model_type='Random Forest'):
    """Train classification model"""
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, cm, y_pred

def train_regression_model(X_train, X_test, y_train, y_test, model_type='Random Forest'):
    """Train regression model"""
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, rmse, mae, r2, y_pred

def main():
    st.markdown('<h1 class="main-header">📊 ML Project - Classification & Regression</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Data Overview", "Exploratory Data Analysis", "Classification", "Regression"]
    )
    
    # Load data
    try:
        df = load_data('Telco_Cusomer_Churn.csv')
        st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()
    
    # Data Overview Page
    if page == "Data Overview":
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Column Names:**")
            st.write(df.columns.tolist())
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Missing Values")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df)) * 100
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("No missing values found!")
    
    # EDA Page
    elif page == "Exploratory Data Analysis":
        st.header("🔍 Exploratory Data Analysis")
        
        st.info("📊 Generating comprehensive data profile using ydata-profiling...")
        
        if st.button("Generate Full Profile Report", type="primary"):
            try:
                with st.spinner("Generating profile report (this may take a minute)..."):
                    # Generate report without caching to avoid PyArrow issues
                    df_clean = df.copy()
                    
                    # Clean TotalCharges if it exists
                    if 'TotalCharges' in df_clean.columns:
                        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
                        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median() if df_clean['TotalCharges'].notna().any() else 0)
                    
                    # Clean object columns
                    for col in df_clean.select_dtypes(include=['object']).columns:
                        df_clean[col] = df_clean[col].fillna('').astype(str)
                        df_clean[col] = df_clean[col].replace('', 'Missing')
                    
                    # Generate profile report
                    profile = ProfileReport(
                        df_clean, 
                        explorative=True, 
                        minimal=False,
                        title="Dataset Profile Report"
                    )
                    
                    # Save report to HTML file
                    report_path = "profile_report.html"
                    profile.to_file(report_path)
                    
                    st.success("✅ Profile report generated successfully!")
                    
                    # Provide download link and display option
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Report (HTML)",
                                data=f.read(),
                                file_name="profile_report.html",
                                mime="text/html"
                            )
                    with col2:
                        if st.button("📊 View Report in Browser"):
                            # Try to display, but handle errors gracefully
                            try:
                                with open(report_path, "r", encoding="utf-8") as f:
                                    html_report = f.read()
                                components.html(html_report, height=800, scrolling=True)
                            except Exception as display_error:
                                st.warning(f"Could not display report inline: {str(display_error)}")
                                st.info("Please download the report and open it in your browser instead.")
                    
                    st.info(f"📁 Report saved to: {report_path}")
                    
            except Exception as e:
                st.error(f"Error generating profile report: {str(e)}")
                st.info("💡 Tip: Try using the Quick Visualizations below for basic EDA, or check the dataset for data type issues.")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
        
        st.subheader("Quick Visualizations")
        
        # Select column for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.write("**Numerical Column Distribution**")
            selected_num_col = st.selectbox("Select a numerical column", numeric_cols)
            if selected_num_col:
                fig = px.histogram(df, x=selected_num_col, nbins=30, title=f"Distribution of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        if len(categorical_cols) > 0:
            st.write("**Categorical Column Analysis**")
            selected_cat_col = st.selectbox("Select a categorical column", categorical_cols)
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Value Counts for {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix for numerical columns
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                          title="Correlation Matrix of Numerical Features")
            st.plotly_chart(fig, use_container_width=True)
    
    # Classification Page
    elif page == "Classification":
        st.header("🎯 Classification Models")
        
        # Select target column
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'customerID' in categorical_cols:
            categorical_cols.remove('customerID')
        
        target_col = st.selectbox("Select target column for classification", categorical_cols, 
                                 index=len(categorical_cols)-1 if 'Churn' in categorical_cols else 0)
        
        if target_col:
            st.info(f"Target variable: **{target_col}**")
            st.write(f"Class distribution:\n{df[target_col].value_counts()}")
            
            # Preprocess data
            df_processed, le_dict, le_target = preprocess_data(df, target_col, task_type='classification')
            
            # Select features
            feature_cols = [col for col in df_processed.columns if col != target_col]
            selected_features = st.multiselect("Select features", feature_cols, default=feature_cols)
            
            if len(selected_features) > 0:
                X = df_processed[selected_features]
                y = df_processed[target_col]
                
                # Train-test split
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Model selection
                model_type = st.selectbox("Select model", ["Random Forest", "Logistic Regression"])
                
                if st.button("Train Classification Model", type="primary"):
                    with st.spinner("Training model..."):
                        model, accuracy, report, cm, y_pred = train_classification_model(
                            X_train_scaled, X_test_scaled, y_train, y_test, model_type
                        )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Accuracy (%)", f"{accuracy*100:.2f}%")
                    
                    with col2:
                        st.write("**Classification Report**")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    fig = px.imshow(cm, text_auto=True, aspect="auto", 
                                   labels=dict(x="Predicted", y="Actual"),
                                   title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance (for Random Forest)
                    if model_type == "Random Forest":
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                                   orientation='h', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
    
    # Regression Page
    elif page == "Regression":
        st.header("📈 Regression Models")
        
        # Select target column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            st.warning("No numerical columns found for regression!")
            st.stop()
        
        target_col = st.selectbox("Select target column for regression", numeric_cols)
        
        if target_col:
            st.info(f"Target variable: **{target_col}**")
            
            # Show target distribution
            fig = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Statistics:**\n{df[target_col].describe()}")
            
            # Preprocess data
            df_processed, le_dict, _ = preprocess_data(df, target_col, task_type='regression')
            
            # Select features
            feature_cols = [col for col in df_processed.columns if col != target_col]
            selected_features = st.multiselect("Select features", feature_cols, default=feature_cols)
            
            if len(selected_features) > 0:
                X = df_processed[selected_features]
                y = df_processed[target_col]
                
                # Handle any remaining NaN values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                # Train-test split
                test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Model selection
                model_type = st.selectbox("Select model", ["Random Forest", "Linear Regression"])
                
                if st.button("Train Regression Model", type="primary"):
                    with st.spinner("Training model..."):
                        model, mse, rmse, mae, r2, y_pred = train_regression_model(
                            X_train_scaled, X_test_scaled, y_train, y_test, model_type
                        )
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col3:
                        st.metric("MAE", f"{mae:.4f}")
                    with col4:
                        st.metric("MSE", f"{mse:.4f}")
                    
                    # Prediction vs Actual plot
                    st.subheader("Predictions vs Actual")
                    results_df = pd.DataFrame({
                        'Actual': y_test.values,
                        'Predicted': y_pred
                    })
                    
                    fig = px.scatter(results_df, x='Actual', y='Predicted', 
                                   title="Actual vs Predicted Values",
                                   labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})
                    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                           y=[y_test.min(), y_test.max()],
                                           mode='lines', name='Perfect Prediction',
                                           line=dict(dash='dash', color='red')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residuals plot
                    st.subheader("Residuals Plot")
                    residuals = y_test.values - y_pred
                    fig = px.scatter(x=y_pred, y=residuals, 
                                   title="Residuals Plot",
                                   labels={'x': 'Predicted Values', 'y': 'Residuals'})
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance (for Random Forest)
                    if model_type == "Random Forest":
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_importance, x='Importance', y='Feature', 
                                   orientation='h', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

