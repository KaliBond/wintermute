"""
Simple test for Node State Matrix functionality
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.title("🧠 Node State Matrix Test")

# Create test data
test_data = np.array([
    [5.2, 6.1, 3.4, 7.8],  # Executive
    [6.3, 7.2, 4.1, 5.9],  # Army  
    [7.1, 5.8, 2.3, 8.5],  # StateMemory
    [5.9, 6.4, 3.7, 8.9],  # Priesthood
    [4.8, 6.7, 4.2, 6.3],  # Stewards
    [7.4, 8.1, 3.1, 5.2],  # Craft
    [6.2, 6.5, 5.3, 4.7],  # Flow
    [4.3, 5.1, 6.8, 3.4]   # Hands
])

node_names = ['Executive', 'Army', 'StateMemory', 'Priesthood', 'Stewards', 'Craft', 'Flow', 'Hands']
dimensions = ['Coherence', 'Capacity', 'Stress', 'Abstraction']

st.success("✅ Test data created successfully!")

# Create the heatmap
st.markdown("### 📊 Node State Matrix [C, K, S, A]")

fig = go.Figure(data=go.Heatmap(
    z=test_data,
    x=dimensions,
    y=node_names,
    colorscale='RdBu_r',
    colorbar=dict(title="State Value"),
    text=np.round(test_data, 2),
    texttemplate="%{text}",
    textfont={"size": 12},
    hovertemplate="Node: %{y}<br>Dimension: %{x}<br>Value: %{z:.2f}<extra></extra>"
))

fig.update_layout(
    title="Node State Matrix Test",
    height=500,
    xaxis=dict(title="System Dimensions"),
    yaxis=dict(title="Institutional Nodes")
)

st.plotly_chart(fig, use_container_width=True)

# Summary table
st.markdown("### 📋 Summary Statistics")

summary_data = []
for i, node in enumerate(node_names):
    C, K, S, A = test_data[i]
    summary_data.append({
        'Node': node,
        'Coherence': f"{C:.2f}",
        'Capacity': f"{K:.2f}", 
        'Stress': f"{S:.2f}",
        'Abstraction': f"{A:.2f}",
        'Mean': f"{np.mean(test_data[i]):.2f}",
        'Max': f"{np.max(test_data[i]):.2f}"
    })

df = pd.DataFrame(summary_data)
st.dataframe(df, use_container_width=True)

st.success("🎉 Node State Matrix test completed successfully!")

st.markdown("---")
st.info("If you can see the heatmap and table above, the Node State Matrix is working correctly!")