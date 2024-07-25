import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Title and description
st.title("Diseases and Vaccine Recommendation App")

st.markdown("""
This is a web app for Diseases and Vaccine recommendation where all the data is taken from WHO.
""")

# Data editor example
df_commands = pd.DataFrame(
    [
        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        {"command": "st.balloons", "rating": 5, "is_widget": False},
        {"command": "st.time_input", "rating": 3, "is_widget": True},
    ]
)
edited_df = st.data_editor(df_commands)

favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

# Spiral plot example
def generate_spiral_data(num_points, num_turns):
    """Generate data for a spiral with the given number of points and turns."""
    indices = np.linspace(0, 1, num_points)
    theta = 2 * np.pi * num_turns * indices
    radius = indices

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    rand = np.random.randn(num_points)

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "idx": indices,
        "rand": rand,
    })
    
    return df

def create_spiral_chart(df):
    """Create an Altair chart for the spiral data."""
    chart = alt.Chart(df, height=700, width=700).mark_point(filled=True).encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    )
    return chart

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

spiral_data = generate_spiral_data(num_points, num_turns)
spiral_chart = create_spiral_chart(spiral_data)

st.altair_chart(spiral_chart)

