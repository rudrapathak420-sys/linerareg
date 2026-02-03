# import streamlit as st
# import joblib
# import numpy as np

# # load the trained regression model

# model = joblib.load('regression_model.joblib')

# # APP UI

# st.title("Job Package Prediction Based on CGPA")
# st.write("Enter your CGPA to predict the expected job package:")

# # CGPA Input

# cgpa = st.number_input(
#     "CGPA",
#     min_value=0.0,
#     max_value=10.0,
#     step=0.1
# )

# # Predict button

# if st.button("Predict Package"): 

#     # Prepare input for model 
#     input_data =  np.array([[cgpa]])  

#     # Make Prediction
#     prediction = np.array(input_data)  

#     # Convert Numpy Output to Python Float Safety
#     predicted_value = prediction.item()

#     # Optional : prevent negative Output
#     predicted_value = max(predicted_value, 0)

#     # Display Result
#     st.success(f"Predicted Package: ₹ {predicted_value:,.2f} LPA")




import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load('regression_model.joblib')

# App UI
st.title("🎓 Job Package Prediction Based on CGPA")
st.write("Enter your CGPA to predict the expected job package (LPA):")

# CGPA Input
cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)

if st.button("Predict Package"):

    # Prediction for user input
    input_data = np.array([[cgpa]])
    predicted_value = max(float(model.predict(input_data)[0]), 0)

    st.success(f"💼 Predicted Package: ₹ {predicted_value:,.2f} LPA")

    # ---- GRAPH ----
    cgpa_range = np.linspace(0, 10, 50).reshape(-1, 1)
    predicted_packages = model.predict(cgpa_range)

    plt.figure()
    plt.plot(cgpa_range, predicted_packages)
    plt.scatter(cgpa, predicted_value)
    plt.xlabel("CGPA")
    plt.ylabel("Package (LPA)")
    plt.title("CGPA vs Predicted Job Package")

    st.pyplot(plt)

   
