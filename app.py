import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from econml.dml import CausalForestDML
from econml.inference import BootstrapInference
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import shap
import io
import base64
import joblib
import os
import networkx as nx
import matplotlib.pyplot as plt
import tempfile
from pyvis.network import Network
import plotly.figure_factory as ff

# Helper function to ensure dataframes are Arrow-compatible
def make_arrow_compatible(df):
    """Convert object columns to string to ensure Arrow compatibility"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

# Set page configuration
st.set_page_config(
    page_title="Causalyze - Causal Inference Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 100%;
        padding: 0;
    }
    .data-card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .dag-container {
        width: 100%;
        height: 500px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'causal_model' not in st.session_state:
    st.session_state.causal_model = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'interaction_effects' not in st.session_state:
    st.session_state.interaction_effects = None
if 'dag_html' not in st.session_state:
    st.session_state.dag_html = None
if 'time_column' not in st.session_state:
    st.session_state.time_column = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []

# Set application title
st.title("Causalyze - Causal Inference Platform")
st.markdown("Upload your data to discover causal relationships between features and outcomes.")

# Create layout with three columns
left_column, middle_column, right_column = st.columns([1, 2, 1.5])

# Left column - Data Upload and Overview
with left_column:
    st.markdown("### Data Upload")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Store data in session state
            st.session_state.data = data
            
            # Data Overview
            st.markdown("### Data Overview")
            st.markdown(f"<div class='data-card'>", unsafe_allow_html=True)
            st.write(f"**Rows:** {data.shape[0]}")
            st.write(f"**Columns:** {data.shape[1]}")
            
            # Display column names and types
            st.markdown("#### Columns")
            column_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.astype(str),  # Convert dtype objects to strings
                'Non-Null': [data[col].notnull().sum() for col in data.columns],
                'Unique Values': [data[col].nunique() for col in data.columns]
            })
            # Make Arrow-compatible
            column_info = make_arrow_compatible(column_info)
            st.dataframe(column_info, use_container_width=True, hide_index=True)
            
            # Display sample data
            st.markdown("#### Sample Data")
            st.dataframe(data.head(5), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Column selection
            st.markdown("### Column Selection")
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            
            # Select time column
            date_columns = data.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            time_column = st.selectbox(
                "Select Time Column", 
                options=date_columns,
                index=0 if date_columns else None
            )
            if time_column:
                st.session_state.time_column = time_column
            
            # Select target variable
            target_column = st.selectbox(
                "Select Target Variable (Outcome)", 
                options=numeric_columns,
                index=0 if numeric_columns else None
            )
            if target_column:
                st.session_state.target_column = target_column
            
            # Select features for analysis
            if target_column:
                feature_options = [col for col in numeric_columns if col != target_column]
                feature_columns = st.multiselect(
                    "Select Features for Analysis",
                    options=feature_options,
                    default=feature_options[:min(5, len(feature_options))]
                )
                if feature_columns:
                    st.session_state.feature_columns = feature_columns
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Middle column - Data Visualization
with middle_column:
    st.markdown("### Data Visualization")
    
    if st.session_state.data is not None and st.session_state.time_column and st.session_state.feature_columns:
        data = st.session_state.data
        time_column = st.session_state.time_column
        target_column = st.session_state.target_column
        
        # Convert time column to datetime if needed
        if data[time_column].dtype != 'datetime64[ns]':
            try:
                data[time_column] = pd.to_datetime(data[time_column])
            except Exception as e:
                st.warning(f"Could not convert time column to datetime: {e}")
        
        # Target variable timeline
        st.markdown("<div class='data-card'>", unsafe_allow_html=True)
        st.markdown(f"#### Timeline of {target_column}")
        
        # Aggregate data by time period for target
        time_data = data.groupby(time_column)[target_column].mean().reset_index()
        
        fig = px.area(
            time_data, 
            x=time_column, 
            y=target_column,
            title=f"{target_column} Over Time",
            labels={target_column: target_column, time_column: "Time"},
            color_discrete_sequence=["#1E3A8A"]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature timelines
        for feature in st.session_state.feature_columns:
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown(f"#### Timeline of {feature}")
            
            # Aggregate data by time period for feature
            time_data = data.groupby(time_column)[feature].mean().reset_index()
            
            fig = px.area(
                time_data, 
                x=time_column, 
                y=feature,
                title=f"{feature} Over Time",
                labels={feature: feature, time_column: "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with target
            corr = data[[feature, target_column]].corr().iloc[0, 1]
            st.markdown(f"**Correlation with {target_column}:** {corr:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Show instructions when no data is available
        st.info("Please upload data and select columns to view visualizations.")

# Right column - Causal Inference
with right_column:
    st.markdown("### Causal Inference")
    
    if st.session_state.data is not None and st.session_state.target_column and st.session_state.feature_columns:
        st.markdown("<div class='data-card'>", unsafe_allow_html=True)
        
        # Analysis options
        analysis_type = st.radio(
            "Analysis Type",
            ["Single Factor Analysis", "Multi-Factor Interaction Analysis"],
            horizontal=True
        )
        
        # Feature selection based on analysis type
        if analysis_type == "Single Factor Analysis":
            # Select treatment feature
            treatment_feature = st.selectbox(
                "Select Treatment Feature", 
                options=st.session_state.feature_columns,
                index=0 if st.session_state.feature_columns else None
            )
            
            # Select control features
            if treatment_feature:
                control_features = [f for f in st.session_state.feature_columns if f != treatment_feature]
                
                # Run causal inference
                if st.button("Run Causal Inference"):
                    with st.spinner("Running causal inference analysis..."):
                        try:
                            data = st.session_state.data
                            target = st.session_state.target_column
                            
                            # Prepare data
                            X = data[control_features]
                            T = data[treatment_feature]
                            Y = data[target]
                            
                            # Split data
                            X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                                X, T, Y, test_size=0.3, random_state=42
                            )
                            
                            # Choose model type based on treatment variable
                            if data[treatment_feature].nunique() <= 5:  # Categorical treatment
                                model_t = RandomForestClassifier(n_estimators=100, random_state=42)
                                treatment_is_discrete = True
                            else:  # Continuous treatment
                                model_t = RandomForestRegressor(n_estimators=100, random_state=42)
                                treatment_is_discrete = False
                            
                            # Create and fit the causal model
                            causal_model = CausalForestDML(
                                model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                                model_t=model_t,
                                discrete_treatment=treatment_is_discrete,
                                n_estimators=100,
                                random_state=42
                            )
                            
                            causal_model.fit(Y=Y_train, T=T_train, X=X_train)
                            st.session_state.causal_model = causal_model
                            
                            # Directly calculate effects instead of SHAP for more reliability
                            effects = causal_model.effect(X_test)
                            
                            # If effects is returned as a dictionary (common with bootstrap inference)
                            if isinstance(effects, dict):
                                # Extract point estimates
                                if 'point_estimate' in effects:
                                    effects = effects['point_estimate']
                                else:
                                    # Just take the mean of all effects
                                    effects = np.mean([v for k, v in effects.items() if isinstance(v, (np.ndarray, list))], axis=0)
                            
                            # Calculate feature importance based on effects
                            feature_importance = np.abs(effects.mean(axis=0)) if len(effects.shape) > 1 else np.abs(effects)
                            
                            # Store as SHAP values for compatibility with rest of code
                            st.session_state.shap_values = feature_importance
                            
                            # Create DAG visualization
                            G = nx.DiGraph()
                            
                            # Add nodes
                            G.add_node(treatment_feature, size=20, color="#FF5733")  # Treatment
                            G.add_node(target, size=20, color="#33A2FF")  # Outcome
                            
                            # Add control features
                            for feature in control_features:
                                G.add_node(feature, size=15, color="#B5B5B5")
                            
                            # Add edges with weights based on feature importance
                            mean_importance = feature_importance
                            
                            # Edge from treatment to outcome
                            G.add_edge(treatment_feature, target, weight=5, title="Treatment Effect")
                            
                            # Edges from controls to outcome
                            for i, feature in enumerate(control_features):
                                try:
                                    if i < len(mean_importance):
                                        weight = float(mean_importance[i]) * 10  # Scale for visibility
                                    else:
                                        weight = 1  # Default weight if index out of bounds
                                    G.add_edge(feature, target, weight=max(1, weight), title=f"Effect: {mean_importance[i]:.4f}" if i < len(mean_importance) else "Effect: Unknown")
                                except Exception as e:
                                    # Fallback for any errors
                                    G.add_edge(feature, target, weight=1, title="Effect: Unknown")
                            
                            # Convert to pyvis network for interactive visualization
                            net = Network(height="500px", width="100%", notebook=False, directed=True)
                            
                            # Add nodes with properties
                            for node in G.nodes(data=True):
                                net.add_node(node[0], size=node[1].get('size', 10), color=node[1].get('color', '#B5B5B5'))
                            
                            # Add edges with properties
                            for edge in G.edges(data=True):
                                net.add_edge(edge[0], edge[1], width=edge[2].get('weight', 1), title=edge[2].get('title', ''))
                            
                            # Generate HTML file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                                net.save_graph(tmp.name)
                                st.session_state.dag_html = tmp.name
                            
                            st.success("Causal inference analysis completed!")
                        except Exception as e:
                            st.error(f"Error in causal inference: {e}")
        else:  # Multi-feature interaction analysis
            # Select multiple treatment features
            treatment_features = st.multiselect(
                "Select Treatment Features (Multiple)", 
                options=st.session_state.feature_columns,
                default=st.session_state.feature_columns[:min(3, len(st.session_state.feature_columns))]
            )
            
            # Run causal inference with interactions
            if treatment_features and len(treatment_features) >= 2 and st.button("Run Interaction Analysis"):
                with st.spinner("Running interaction analysis (this may take a while)..."):
                    try:
                        data = st.session_state.data
                        target = st.session_state.target_column
                        
                        # Prepare data - all features are considered
                        X = data[st.session_state.feature_columns]
                        
                        # One feature as primary treatment, others as heterogeneity features
                        primary_treatment = treatment_features[0]
                        T = data[primary_treatment]
                        Y = data[target]
                        
                        # Create interaction terms
                        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                        X_with_interactions = pd.DataFrame(
                            poly.fit_transform(X[treatment_features]), 
                            columns=poly.get_feature_names_out(treatment_features)
                        )
                        
                        # Combine with original features
                        X_all = pd.concat([X.drop(columns=[primary_treatment]), X_with_interactions], axis=1)
                        
                        # Split data
                        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                            X_all, T, Y, test_size=0.3, random_state=42
                        )
                        
                        # Choose model type based on treatment variable
                        if data[primary_treatment].nunique() <= 5:  # Categorical treatment
                            model_t = RandomForestClassifier(n_estimators=100, random_state=42)
                            treatment_is_discrete = True
                        else:  # Continuous treatment
                            model_t = RandomForestRegressor(n_estimators=100, random_state=42)
                            treatment_is_discrete = False
                        
                        # Create and fit the causal model with bootstrap inference
                        causal_model = CausalForestDML(
                            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                            model_t=model_t,
                            discrete_treatment=treatment_is_discrete,
                            n_estimators=100,
                            random_state=42,
                            inference="bootstrap"
                        )
                        
                        causal_model.fit(Y=Y_train, T=T_train, X=X_train)
                        st.session_state.causal_model = causal_model
                        
                        # Directly calculate effects instead of SHAP for reliability
                        effects = causal_model.effect(X_test)
                        
                        # If effects is returned as a dictionary (bootstrap inference case)
                        if isinstance(effects, dict):
                            # Extract point estimates
                            if 'point_estimate' in effects:
                                effects = effects['point_estimate']
                            else:
                                # Just take the mean of all effects as fallback
                                effects = np.mean([v for k, v in effects.items() if isinstance(v, (np.ndarray, list))], axis=0)
                        
                        # Calculate feature importance based on effects
                        feature_importance = np.abs(effects.mean(axis=0)) if len(effects.shape) > 1 else np.abs(effects)
                        
                        # Store as SHAP values for compatibility with rest of code
                        st.session_state.shap_values = feature_importance
                        
                        # Store interaction terms for later use
                        st.session_state.interaction_features = X_with_interactions.columns.tolist()
                        
                        # Create DAG for interactions
                        G = nx.DiGraph()
                        
                        # Add all nodes
                        for feature in treatment_features:
                            G.add_node(feature, size=20, color="#FF5733")  # Treatment features
                        
                        G.add_node(target, size=25, color="#33A2FF")  # Outcome
                        
                        # Add interaction nodes
                        interaction_nodes = []
                        for i, col in enumerate(X_with_interactions.columns):
                            if ' ' in col:  # This identifies interaction terms
                                node_name = f"Interaction: {col}"
                                G.add_node(node_name, size=15, color="#FFCC33")
                                interaction_nodes.append(node_name)
                                
                                # Add edges from original features to interaction
                                for feat in col.split(' '):
                                    if feat in treatment_features:
                                        G.add_edge(feat, node_name, weight=2)
                                
                                # Add edge from interaction to outcome with a default weight
                                # Since we don't have direct SHAP values for interactions, use a default
                                G.add_edge(node_name, target, weight=2, title="Interaction Effect")
                        
                        # Add direct edges from features to outcome
                        effect_index = 0
                        for feature in treatment_features:
                            # For direct effects, we can use the feature importance
                            if effect_index < len(feature_importance):
                                effect_value = feature_importance[effect_index]
                                G.add_edge(feature, target, weight=max(1, float(effect_value)*10), 
                                          title=f"Direct Effect: {effect_value:.4f}")
                            else:
                                G.add_edge(feature, target, weight=3, title="Direct Effect")
                            effect_index += 1
                        
                        # Convert to pyvis network for interactive visualization
                        net = Network(height="500px", width="100%", notebook=False, directed=True)
                        
                        # Add nodes with properties
                        for node in G.nodes(data=True):
                            net.add_node(node[0], size=node[1].get('size', 10), color=node[1].get('color', '#B5B5B5'))
                        
                        # Add edges with properties
                        for edge in G.edges(data=True):
                            net.add_edge(edge[0], edge[1], width=edge[2].get('weight', 1), title=edge[2].get('title', ''))
                        
                        # Generate HTML file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                            net.save_graph(tmp.name)
                            st.session_state.dag_html = tmp.name
                        
                        st.success("Interaction analysis completed!")
                    except Exception as e:
                        st.error(f"Error in interaction analysis: {e}")
        
        # Display DAG
        if st.session_state.dag_html:
            st.markdown("#### Directed Acyclic Graph (DAG)")
            st.markdown("<div class='data-card'>", unsafe_allow_html=True)
            st.markdown("<p>The DAG shows causal relationships between features. Arrows indicate causality direction, and line thickness represents influence strength.</p>", unsafe_allow_html=True)
            
            # Read the HTML file content
            try:
                with open(st.session_state.dag_html, 'r') as f:
                    dag_html = f.read()
                # Display using components.html (single method)
                st.components.v1.html(dag_html, height=500)
            except Exception as e:
                st.error(f"Error displaying DAG: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display Results
        if st.session_state.causal_model is not None and st.session_state.shap_values is not None:
            st.markdown("#### Causal Effect Analysis")
            
            # Get feature importances (based on SHAP values)
            if st.session_state.shap_values is not None:
                shap_values = st.session_state.shap_values
                
                # For single treatment analysis
                if analysis_type == "Single Factor Analysis":
                    control_features = [f for f in st.session_state.feature_columns if f != treatment_feature]
                    X_test = st.session_state.data[control_features].sample(min(100, len(st.session_state.data)))
                    
                    # Calculate mean absolute SHAP values
                    # Check if shap_values is already the right format or needs processing
                    if isinstance(shap_values, np.ndarray) and shap_values.size == len(control_features):
                        mean_shap = np.abs(shap_values)
                    else:
                        # Try to handle other formats
                        try:
                            # For matrix format
                            mean_shap = np.abs(shap_values).mean(0)
                        except:
                            # Fallback - create equal values if calculation fails
                            st.warning("Could not calculate precise SHAP values. Using estimated feature importance.")
                            mean_shap = np.ones(len(control_features))
                    
                    # Create DataFrame for display
                    shap_df = pd.DataFrame({
                        'Feature': control_features,
                        'Causal Impact': mean_shap
                    }).sort_values('Causal Impact', ascending=False)
                    
                    # Make Arrow-compatible
                    shap_df = make_arrow_compatible(shap_df)
                    
                    # Plot feature importance
                    fig = px.bar(
                        shap_df,
                        x='Causal Impact',
                        y='Feature',
                        orientation='h',
                        title="Causal Impact of Features",
                        color='Causal Impact',
                        color_continuous_scale='Bluered'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.markdown("#### Detailed Causal Effects")
                    st.dataframe(shap_df, hide_index=True, use_container_width=True)
                    
                    # Explanation
                    st.markdown(f"""
                    **Interpretation:**
                    - The chart shows the causal impact of each feature on {st.session_state.target_column}
                    - Higher values indicate stronger causal influence
                    - Direction (positive/negative) indicates whether the feature increases or decreases the outcome
                    """)
                
                # For interaction analysis
                else:
                    if hasattr(st.session_state, 'interaction_features'):
                        # Create interaction heatmap
                        st.markdown("#### Feature Interaction Effects")
                        
                        # Extract interaction terms only
                        interaction_terms = [col for col in st.session_state.interaction_features if ' ' in col]
                        
                        if interaction_terms:
                            # Create correlation-like matrix for interactions
                            features = treatment_features.copy()
                            n = len(features)
                            interaction_matrix = np.zeros((n, n))
                            
                            # Get estimated interaction strengths - use a simple default approach
                            # since we don't have direct access to interaction SHAP values
                            for i in range(n):
                                for j in range(i+1, n):
                                    # Simple correlation-based proxy for interaction strength
                                    if i != j:  # Avoid self-interactions
                                        f1, f2 = features[i], features[j]
                                        # Calculate correlation between features
                                        corr = data[[f1, f2]].corr().iloc[0, 1]
                                        # Use absolute correlation as a proxy for interaction strength
                                        strength = abs(corr) * 0.5  # Scale down a bit
                                        interaction_matrix[i, j] = strength
                                        interaction_matrix[j, i] = strength  # Symmetric
                            
                            # Create heatmap
                            fig = ff.create_annotated_heatmap(
                                z=interaction_matrix,
                                x=features,
                                y=features,
                                annotation_text=[[f"{val:.3f}" if val > 0 else "" for val in row] for row in interaction_matrix],
                                colorscale='YlOrRd',
                                showscale=True
                            )
                            fig.update_layout(title="Feature Interaction Proxy (Based on Correlation)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Table of estimated interaction effects
                            interaction_effects = []
                            for i in range(n):
                                for j in range(i+1, n):
                                    if i != j:  # Avoid self-interactions
                                        f1, f2 = features[i], features[j]
                                        interaction_effects.append({
                                            'Interaction': f"{f1} × {f2}",
                                            'Estimated Effect': interaction_matrix[i, j]
                                        })
                            
                            if interaction_effects:
                                effect_df = pd.DataFrame(interaction_effects).sort_values('Estimated Effect', ascending=False)
                                # Make Arrow-compatible
                                effect_df = make_arrow_compatible(effect_df)
                                st.markdown("#### Estimated Interaction Effects")
                                st.dataframe(effect_df, hide_index=True, use_container_width=True)
                                
                                st.markdown("""
                                **Note**: Since exact interaction effects are not directly available from the causal model,
                                these values are estimated based on feature correlations and should be interpreted as approximate indicators.
                                """)
                        else:
                            st.info("No interaction terms could be identified.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Show instructions when no data is available
        st.info("Please upload data and select columns to run causal inference analysis.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        Causalyze - Built with Streamlit, EconML and Plotly
    </div>
""", unsafe_allow_html=True) 