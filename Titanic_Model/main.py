import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# Load trained model
try:
    model = joblib.load('titanic_model.pkl')
except FileNotFoundError:
    model = None

# ---------------- Main Window ----------------
root = tk.Tk()
root.title("Titanic Survival Predictor")
root.geometry("480x600")
root.resizable(False, False)
root.configure(bg="#f0f2f5")

# ---------------- Styles ----------------
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f0f2f5", font=("Segoe UI", 11))
style.configure("Header.TLabel", background="#f0f2f5", font=("Segoe UI", 18, "bold"))
style.configure("Sub.TLabel", background="#f0f2f5", font=("Segoe UI", 10), foreground="#555")
style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=8)
style.configure("TCombobox", font=("Segoe UI", 11))
style.configure("TEntry", font=("Segoe UI", 11))

# ---------------- Header ----------------
header_frame = tk.Frame(root, bg="#f0f2f5")
header_frame.pack(fill="x", pady=(20, 10))

ttk.Label(header_frame, text="🚢 Titanic Survival Predictor", style="Header.TLabel").pack()
ttk.Label(header_frame, text="Enter passenger details to predict survival", style="Sub.TLabel").pack(pady=(2, 0))

# ---------------- Form Frame ----------------
form_frame = tk.Frame(root, bg="#ffffff", padx=25, pady=25,
                       highlightbackground="#d9d9d9", highlightthickness=1)
form_frame.pack(padx=20, pady=10, fill="both", expand=False)

# Configure two columns: labels and inputs
form_frame.columnconfigure(0, weight=1)
form_frame.columnconfigure(1, weight=2)

# --- Pclass ---
ttk.Label(form_frame, text="Passenger Class").grid(row=0, column=0, sticky="w", pady=8)
pclass_var = tk.StringVar(value="1")
pclass_combo = ttk.Combobox(form_frame, textvariable=pclass_var, values=["1", "2", "3"], state="readonly")
pclass_combo.grid(row=0, column=1, sticky="ew", pady=8, padx=(10, 0))

# --- Sex ---
ttk.Label(form_frame, text="Sex").grid(row=1, column=0, sticky="w", pady=8)
sex_var = tk.StringVar(value="Male")
sex_combo = ttk.Combobox(form_frame, textvariable=sex_var, values=["Male", "Female"], state="readonly")
sex_combo.grid(row=1, column=1, sticky="ew", pady=8, padx=(10, 0))

# --- Age ---
ttk.Label(form_frame, text="Age").grid(row=2, column=0, sticky="w", pady=8)
age_entry = ttk.Entry(form_frame)
age_entry.insert(0, "30")
age_entry.grid(row=2, column=1, sticky="ew", pady=8, padx=(10, 0))

# --- Siblings/Spouses ---
ttk.Label(form_frame, text="Siblings/Spouses Aboard").grid(row=3, column=0, sticky="w", pady=8)
sibsp_entry = ttk.Entry(form_frame)
sibsp_entry.insert(0, "0")
sibsp_entry.grid(row=3, column=1, sticky="ew", pady=8, padx=(10, 0))

# --- Parents/Children ---
ttk.Label(form_frame, text="Parents/Children Aboard").grid(row=4, column=0, sticky="w", pady=8)
parch_entry = ttk.Entry(form_frame)
parch_entry.insert(0, "0")
parch_entry.grid(row=4, column=1, sticky="ew", pady=8, padx=(10, 0))

# --- Fare ---
ttk.Label(form_frame, text="Fare Paid ($)").grid(row=5, column=0, sticky="w", pady=8)
fare_entry = ttk.Entry(form_frame)
fare_entry.insert(0, "32.0")
fare_entry.grid(row=5, column=1, sticky="ew", pady=8, padx=(10, 0))

# --- Embarked ---
ttk.Label(form_frame, text="Port of Embarkation").grid(row=6, column=0, sticky="w", pady=8)
embarked_var = tk.StringVar(value="Southampton")
embarked_combo = ttk.Combobox(
    form_frame, textvariable=embarked_var,
    values=["Southampton", "Cherbourg", "Queenstown"], state="readonly"
)
embarked_combo.grid(row=6, column=1, sticky="ew", pady=8, padx=(10, 0))

# ---------------- Result Display ----------------
result_frame = tk.Frame(root, bg="#f0f2f5")
result_frame.pack(pady=(10, 5), fill="x")

result_label = tk.Label(result_frame, text="", font=("Segoe UI", 14, "bold"),
                         bg="#f0f2f5", wraplength=420, justify="center")
result_label.pack()

# ---------------- Mapping helpers ----------------
SEX_MAP = {"Male": 0, "Female": 1}
EMBARKED_MAP = {"Southampton": 2, "Cherbourg": 0, "Queenstown": 1}

# ---------------- Predict Function ----------------
def predict():
    if model is None:
        messagebox.showerror("Error", "Model file 'titanic_model.pkl' not found.")
        return

    try:
        pclass = int(pclass_var.get())
        sex = SEX_MAP[sex_var.get()]
        age = float(age_entry.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        fare = float(fare_entry.get())
        embarked = EMBARKED_MAP[embarked_var.get()]
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for Age, Fare, etc.")
        return

    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                               columns=['Pclass', 'Sex', 'Age', 'sibsp', 'Parch', 'Fare', 'Embarked'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        result_label.config(text=f"✅ Likely to SURVIVE\n(Confidence: {probability:.1f}%)", fg="#1e7e34")
    else:
        result_label.config(text=f"❌ Likely NOT to Survive\n(Confidence: {100 - probability:.1f}%)", fg="#c0392b")

# ---------------- Predict Button ----------------
predict_btn = ttk.Button(root, text="Predict Survival", command=predict)
predict_btn.pack(pady=15, ipadx=20)

# ---------------- Footer ----------------
footer = ttk.Label(root, text="Model: Logistic Regression | Titanic Dataset", style="Sub.TLabel")
footer.pack(pady=(5, 15))

root.mainloop()