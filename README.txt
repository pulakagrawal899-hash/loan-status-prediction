LOAN STATUS PREDICTION — WEB APP

WHAT IS INCLUDED
- app.py: Streamlit web app
- requirements.txt: Python libraries required
- Put your existing train.csv in this same folder.

FOLDER SHOULD LOOK LIKE THIS:
Loan_Status_Prediction_WebApp/
  app.py
  requirements.txt
  train.csv

HOW TO RUN ON WINDOWS
1. Extract this ZIP file to a folder.
2. Copy your train.csv into that folder.
3. Open the folder in File Explorer.
4. Click the address bar, type cmd, and press Enter.
5. Run:
   py -3.14 -m pip install -r requirements.txt
6. Then run:
   py -3.14 -m streamlit run app.py
7. A browser tab should open with the app. If it does not, copy the Local URL shown in the terminal (usually http://localhost:8501) into your browser.

TROUBLESHOOTING
- If the command says streamlit is not recognized, use:
  py -3.14 -m streamlit run app.py
- If the app says train.csv is missing, ensure train.csv is beside app.py.
- Keep the terminal window open while presenting the app. Stop it with Ctrl+C after your demo.

DEMO TIP
Enter example details, click Predict Loan Status, and explain that the app uses a Support Vector Machine (SVM) trained on historical data.
The result is educational only and does not guarantee an actual bank's decision.
