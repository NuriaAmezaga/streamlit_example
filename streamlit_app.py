import pandas as pd
import streamlit as st

# Title and description
st.title("Diseases and Vaccine Recommendation App")

st.markdown("""
This is a web app for Diseases and Vaccine recommendation where all the data is taken from WHO.
""")

# Load and display the CSV file
try:
    df = pd.read_csv("sp.csv")
    st.markdown("### Diseases and Descriptions")
    st.dataframe(df[['disease', 'description']])
except FileNotFoundError:
    st.error("The file 'sp.csv' was not found. Please ensure it is in the correct directory.")
except KeyError:
    st.error("The file 'sp.csv' must contain 'disease' and 'description' columns.")

# Data editor for images
data_df = pd.DataFrame(
    {
        "apps": [
            "https://drive.google.com/uc?export=view&id=1rcmdKL8iGM5UoF7-qOcvkiiSVxvbrEoy",
            "https://drive.google.com/uc?export=view&id=1E0iFNctXsa5GK1R9hd2XZNfYRVWFprYX",
        ],
    }
)

# Check Streamlit version for column_config
if hasattr(st, 'column_config') and hasattr(st.column_config, 'ImageColumn'):
    st.data_editor(
        data_df,
        column_config={
            "apps": st.column_config.ImageColumn(
                "Preview Image", help="Disease and vaccine images"
            )
        },
        hide_index=True,
    )
else:
    st.error("The current Streamlit version does not support `st.column_config.ImageColumn`. Please upgrade to the latest version.")


