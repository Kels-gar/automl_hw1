import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="AutoML Classifier", layout="wide")

page = st.sidebar.radio("Navigation", ["Home", "Information"])

if page == "Information":
    st.title("Information")
    st.markdown("""
### What is AutoML?
AutoML, or Automated Machine Learning, is the process of using automation to select, train, and evaluate machine learning models. It helps simplify the process of building predictive models, especially for users without a background in data science. AutoML takes care of common tasks like cleaning data, choosing algorithms, and tuning parameters, making it accessible for more people to analyze data effectively.

### What You'll Need
- A CSV file with **a target column** (what you want to predict)
- All other columns should be **features**
- The app will handle missing values, encoding, and scaling for you.

### Models Included and What They're For

- **Logistic Regression**: A statistical model used for binary classification problems (like predicting yes/no or true/false). It works well when the relationship between features and the outcome is roughly linear.

- **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their results. It's good for handling both categorical and numerical data, dealing with missing values, and reducing overfitting.

- **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies new points by looking at the most common class among its nearest neighbors. It's simple, non-parametric, and works well when the data is small and clearly clustered.

- **Naive Bayes**: A fast, probabilistic model based on Bayes' Theorem. It assumes feature independence and works particularly well for high-dimensional data like text (e.g., spam detection, sentiment analysis).

- **Gradient Boosting**: A boosting algorithm that builds models sequentially, each correcting errors from the previous one. It is powerful and typically achieves high accuracy, although it may take longer to train and is more sensitive to noise.

### Performance Metrics
- **Accuracy**: The percentage of total predictions the model got right. Good for measuring overall correctness.
- **Precision**: The percentage of predictions labeled positive that were actually positive. Useful when false positives are costly (e.g., fraud detection).
- **Confusion Matrix**: A table that summarizes prediction results by showing true vs. predicted values.

_Models with over 80% accuracy are typically considered strong performers for general classification tasks._
""")

elif page == "Home":
    st.title("AutoML App – Tabular Classification")
    st.markdown("""
Welcome to the AutoML Classifier App

This tool is designed to help you build and compare multiple machine learning models on your own data — no coding required.

If you’re not sure what AutoML is or how these models work, check the **Information** tab on the left-hand side for definitions and descriptions.
""")

    st.markdown("""
### What you’ll do:

1. **Upload your CSV dataset**  
   Your file should include one column you want to predict (the **target**) and several other columns with helpful information (**features**).

2. **Review the data**  
   Get a quick summary so you can understand what’s in your file.

3. **Choose models**  
   Select 3 to 5 machine learning algorithms you’d like to run.

4. **Pick your target column**  
   Tell the app which column contains the thing you're trying to predict.

5. **Get results**  
   The app will automatically clean, train, and evaluate your data using the models you selected.

6. **View metrics**  
   You’ll see easy-to-read results like **accuracy**, **precision**, and model comparisons.
""")

    if "step" not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        st.markdown("### Step 1: Upload your dataset")
        file = st.file_uploader("Upload your CSV file", type=["csv"])
        if file:
            try:
                df = pd.read_csv(file)
                st.session_state.df = df
                st.write("### Data Preview")
                st.dataframe(df.head())

                st.write("### Summary Statistics")
                st.write(df.describe(include='all'))

                num_rows, num_cols = df.shape
                categorical_count = len(df.select_dtypes(include='object').columns)
                recommendation = []

                if categorical_count >= num_cols / 2:
                    recommendation.append("Naive Bayes")
                if num_rows > 5000:
                    recommendation.append("Random Forest")
                if num_cols > 20:
                    recommendation.append("Gradient Boosting")
                if num_rows < 1000:
                    recommendation.append("K-Nearest Neighbors")
                recommendation.append("Logistic Regression")

                st.info(f"Recommended Models: {', '.join(set(recommendation[:3]))}")

                if st.button("Next: Select Models"):
                    st.session_state.step = 2
            except Exception as e:
                st.error(f"Upload failed: {e}")

    elif st.session_state.step == 2:
        st.markdown("### Step 2: Select Classification Models (3–5 required)")
        model_options = [
            "Logistic Regression",
            "Random Forest",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Gradient Boosting"
        ]
        selected_models = st.multiselect("Choose models to run:", model_options)

        if len(selected_models) < 3:
            st.warning("Please select at least 3 models.")
        elif len(selected_models) > 5:
            st.warning("Please select no more than five models.")
        else:
            st.session_state.models_selected = selected_models
            if st.button("Next: Select Target Column"):
                st.session_state.step = 3

    elif st.session_state.step == 3:
        st.markdown("### Step 3: Select your target column")
        df = st.session_state.df
        target_col = st.selectbox("Choose the column to predict:", df.columns)

        if target_col:
            df = df[df[target_col].notna()]
            if df.empty:
                st.error("Target column contains only missing values.")
            else:
                st.session_state.df_clean = df
                st.session_state.target_col = target_col
                if st.button("Run Models"):
                    st.session_state.step = 4

    elif st.session_state.step == 4:
        st.markdown("### Step 4: Training and Evaluation")
        df = st.session_state.df_clean
        target_col = st.session_state.target_col
        selected_models = st.session_state.models_selected

        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = {}

        for name in selected_models:
            model = model_map[name]
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                results[name] = {"accuracy": acc * 100, "precision": prec * 100}
            except Exception as e:
                st.warning(f"{name} failed: {e}")

        if results:
            st.subheader("Individual Model Results")
            st.markdown("*Models with over 80% accuracy are typically considered strong performers for general classification tasks.*")
            st.markdown("*Accuracy tells you how often the model was right overall. Precision tells you how trustworthy the positive predictions are.*")

            for name, res in results.items():
                st.markdown(f"**{name}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Accuracy", value=f"{res['accuracy']:.2f}%")
                with col2:
                    st.metric(label="Precision", value=f"{res['precision']:.2f}%")

            summary_df = pd.DataFrame([
                {"Model": name, "Accuracy": res["accuracy"], "Precision": res["precision"]}
                for name, res in results.items()
            ])

            st.subheader("Chart 1: Accuracy and Precision Bar Chart")
            st.markdown("This bar chart shows both accuracy and precision for each selected model. It’s helpful for quickly identifying which models performed best and where they might be weaker (e.g., high accuracy but low precision).")
            fig1, ax1 = plt.subplots()
            summary_df.set_index("Model").plot(kind="bar", ax=ax1, rot=45)
            ax1.set_ylabel("Score (%)")
            ax1.set_title("Accuracy and Precision by Model")
            st.pyplot(fig1)

            st.subheader("Chart 2: Accuracy vs. Precision Scatter Plot")
            st.markdown("This scatter plot places each model on a 2D grid, with accuracy on one axis and precision on the other. Models in the top-right quadrant are generally best, showing both accurate and reliable predictions.")
            fig2, ax2 = plt.subplots()
            ax2.scatter(summary_df["Accuracy"], summary_df["Precision"])
            for _, row in summary_df.iterrows():
                ax2.annotate(row["Model"], (row["Accuracy"] + 0.3, row["Precision"] + 0.3))
            ax2.set_xlabel("Accuracy (%)")
            ax2.set_ylabel("Precision (%)")
            ax2.set_title("Model Accuracy vs. Precision")
            st.pyplot(fig2)

            st.subheader("Chart 3: Simulated Accuracy Spread Box Plot")
            st.markdown("This box plot gives a visual sense of variability. Each model is shown with a range of synthetic accuracy scores to simulate how it might perform across different datasets or runs.")
            simulated = {m: np.random.normal(loc=v["accuracy"], scale=2, size=10) for m, v in results.items()}
            fig3, ax3 = plt.subplots()
            ax3.boxplot(simulated.values())
            ax3.set_xticklabels(simulated.keys(), rotation=25)
            ax3.set_ylabel("Accuracy (%)")
            ax3.set_title("Simulated Accuracy Spread")
            st.pyplot(fig3)

            st.download_button(
                label="Download Results as CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="automl_results.csv",
                mime="text/csv"
            )

            st.markdown("### Start Over")
            st.markdown("*Click the button below to reset everything. If nothing happens, click it again.*")
            if st.button("Start Over"):
                st.session_state.clear()
                st.rerun()
        else:
            st.error("No models were successfully trained.")
