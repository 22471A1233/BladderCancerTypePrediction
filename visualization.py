import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_prediction_probability(probabilities, classes):
    """
    Plot prediction probability bar chart using Plotly
    """
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Cancer/Condition Type',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix using Matplotlib/Seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return fig

def plot_feature_importance(importances, features):
    """
    Plot feature importance bar chart using Plotly
    """
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top Important Features',
        labels={'Importance': 'Relative Importance', 'Feature': 'Clinical Parameter'},
        template='plotly_white',
        height=500
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
