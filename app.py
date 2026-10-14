import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Status Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Status Prediction")
st.write(
    "A machine-learning demonstration that learns from historical loan applications "
    "and predicts an estimated loan status. This is an educational project, not a bank decision."
)

DATA_FILE = "train.csv"

@st.cache_data
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    required = [
        "Loan_ID", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Status"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "The CSV does not look like the expected loan prediction dataset. "
            "Missing columns: " + ", ".join(missing_cols)
        )

    df = df.dropna().copy()
    df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    df["Dependents"] = df["Dependents"].replace("3+", "4")

    category_maps = {
        "Married": {"No": 0, "Yes": 1},
        "Gender": {"Female": 0, "Male": 1},
        "Self_Employed": {"No": 0, "Yes": 1},
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
        "Education": {"Not Graduate": 0, "Graduate": 1},
    }
    for col, mapping in category_maps.items():
        df[col] = df[col].map(mapping)

    feature_cols = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area"
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols + ["Loan_Status"])

    if len(df) < 10:
        raise ValueError("Not enough complete rows remain after cleaning to train the model.")

    X = df[feature_cols]
    y = df["Loan_Status"].astype(int)
    return df, X, y, feature_cols

try:
    data, X, y, feature_cols = load_and_prepare_data(DATA_FILE)
except FileNotFoundError:
    st.error("I can't find train.csv. Put train.csv in the same folder as app.py, then run the app again.")
    st.stop()
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=2
)
model = SVC(kernel="linear")
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

with st.expander("📊 Model information", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows used", f"{len(data)}")
    c2.metric("Test accuracy", f"{test_accuracy:.1%}")
    c3.metric("Test samples", f"{len(X_test)}")
    st.caption(
        "Accuracy is measured on a small held-out sample from this dataset. "
        "It does not guarantee performance on new real-world applications."
    )

st.subheader("Enter applicant details")
st.write("Use the form below to generate a prediction.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender_label = st.selectbox("Gender", ["Male", "Female"])
        married_label = st.selectbox("Married?", ["Yes", "No"])
        dependents_label = st.selectbox("Number of dependents", ["0", "1", "2", "3+"])
        education_label = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed_label = st.selectbox("Self-employed?", ["No", "Yes"])
        applicant_income = st.number_input("Applicant income (monthly, dataset units)", min_value=0, value=5000, step=500)
    with col2:
        coapplicant_income = st.number_input("Co-applicant income (monthly, dataset units)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan amount (dataset units)", min_value=1, value=150, step=10)
        loan_term = st.selectbox("Loan term (months)", [360, 300, 240, 180, 120, 60, 36, 12])
        credit_history_label = st.selectbox("Credit history", ["Good (1)", "Not good / unavailable (0)"])
        property_label = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("🔍 Predict Loan Status", use_container_width=True)

if submitted:
    input_row = pd.DataFrame([{
        "Gender": 1 if gender_label == "Male" else 0,
        "Married": 1 if married_label == "Yes" else 0,
        "Dependents": 4 if dependents_label == "3+" else int(dependents_label),
        "Education": 1 if education_label == "Graduate" else 0,
        "Self_Employed": 1 if self_employed_label == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history_label.startswith("Good") else 0,
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_label],
    }], columns=feature_cols)

    prediction = int(model.predict(input_row)[0])
    st.divider()
    if prediction == 1:
        st.success("Prediction: LOAN APPROVED ✅")
        st.write("The model predicts the application belongs to the approved category.")
    else:
        st.warning("Prediction: LOAN NOT APPROVED ❌")
        st.write("The model predicts the application belongs to the not-approved category.")
    st.info(
        "This is only a prediction from a classroom dataset. It is not financial advice "
        "and must not be used as an actual lending decision."
    )

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
