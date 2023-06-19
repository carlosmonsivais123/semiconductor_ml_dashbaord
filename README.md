# Semi-Conductor Manufacturing Process Dashboard
### By: Carlos Monsivais

Welcome to my semi-conductor manufacturing process data science presentation!

#### Data
The dataset that will be analzyed comes from the following repository: [UCI SECOM Data](https://archive.ics.uci.edu/dataset/179/secom) 

#### Summary
During semi-conductor manufacturing there are thousands of signals being collected to ensure the quality of the chip. 
As a result, not all the signals collected may be helpful for determining the quality of a chip whether it will pass or fail inspection.
With this in mind, there should be a more proactive approach to determining not only when a chip will fail inspection but the biggest question being 
why it failed inspection and what sensors were the most important when determining this. This is the problem that will be solved using data science 
tools such Python, sklearn, mlflow and plotly to give an overall visualization to try and solve this problem.

#### How to Run
1. Create a virtual environment and install the requirements.txt
2. Go to the config.yaml file located in the Configuration_Variables/config.yaml
    - Set your configuration variables you want to use, or leave them as is in default.
3. Run the main.py file by executing:
    - python3 main.py
4. After a successful execution, you can execute the streamlit application by running the following command:
    - streamlit run Introduction.py

#### Live Application
[Semi-Conductor Manufacturing Process Dashboard](https://carlosmonsivais123-semiconduct-streamlit-appintroduction-q51yyf.streamlit.app/)