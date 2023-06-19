import streamlit as st

st.set_page_config(page_title="Introduction",
                   layout="wide")

st.write("# Semi-Conductor Manufacturing Process Dashboard")
st.write("### By: Carlos Monsivais")

st.sidebar.success("Introduction")

st.markdown(
    """
    Welcome to my semi-conductor manufacturing process data science presentation!

    The dataset that will be analzyed comes from the following repository: [UCI SECOM Data](https://archive.ics.uci.edu/dataset/179/secom) 

    The repository with the code to recreate this analysis can be found here: [Carlos Monsivais GitHub Repository](https://github.com/carlosmonsivais123/semiconductor_ml_dashbaord/tree/master)
"""
)

st.markdown('##### Summary')


st.markdown("""
    During semi-conductor manufacturing there are thousands of signals being collected to ensure the quality of the chip. 
    As a result, not all the signals collected may be helpful for determining the quality of a chip whether it will pass or fail inspection.
    With this in mind, there should be a more proactive approach to determining not only when a chip will fail inspection but the biggest question being 
    why it failed inspection and what sensors were the most important when determining this. This is the problem that will be solved using data science 
    tools such Python, sklearn, mlflow and plotly to give an overall visualization to try and solve this problem.
"""
)


col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("Images/semi-conductor.jpeg")

with col3:
    st.write(' ')