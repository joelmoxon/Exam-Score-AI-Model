import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Global variables to store dataset and trained model
df = None
model = None

# Function to replace non-numerirc values in numeric columns with mean
def preprocess_data(dataframe):
    df_processed = dataframe.copy()
    numeric_columns = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'exam_score']
    for column in numeric_columns:
        df_processed[column] = df_processed[column].astype(str)
        numeric_mask = df_processed[column].str.replace('.', '', 1).str.isdigit() 
        if numeric_mask.any():
            numeric_values = pd.to_numeric(df_processed.loc[numeric_mask, column])
            median = numeric_values.median()
            df_processed.loc[~numeric_mask, column] = median
            df_processed[column] = pd.to_numeric(df_processed[column])
    return df_processed

# Function to load dataset and call pre-process function 
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            df = preprocess_data(df)
            messagebox.showinfo("Success", "Dataset loaded successfully *but did you check the script!")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

# Function to save the processed data as a CSV file
def save_processed_data():
    if df is None:
        messagebox.showerror("Error", "No dataset loaded")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        df.to_csv(file_path, index=False)
        messagebox.showinfo("", "Your data has been saved")


# Function to train model
def train_model(df, features, target):
    global model
    df_copy = df.copy()
    try:

        # Check if features are categorical and convert to numerical 
        for feature in features:
            if df_copy[feature].dtype == "object":
                df_copy[feature] = LabelEncoder().fit_transform(df_copy[feature].astype(str))

        # Check if target is categorical and convert to numerical 
        if df_copy[target].dtype == "object":
                df_copy[target] = LabelEncoder().fit_transform(df_copy[target].astype(str))

        X = df_copy[features]
        y = df_copy[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {mse:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

# Function that uses trained model to predict outcomes 
def make_predictions(model, df, features):
    df_copy = df.copy()
    try:

        # Check if features are categorical and convert to numerical 
        for feature in features:
            if df_copy[feature].dtype == "object":
                df_copy[feature] = LabelEncoder().fit_transform(df_copy[feature].astype(str))
        X_new = df_copy[features]
        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Please add funtion comment
root = tk.Tk()
root.title("Student Predictive Grades")

# Please add funtion comment
load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset())
load_button.pack(pady=10)

# Button to save the preprocessed data
save_button = tk.Button(root, text="Save Pre-Processed Data", command=save_processed_data)
save_button.pack(pady=10)

#Please add funtion comment
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Please add funtion comment
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Please add funtion comment
train_button = tk.Button(root, text="Train Model", command=lambda: train_model(df, features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Please add funtion comment
predict_button = tk.Button(root, text="Make Predictions", command=lambda: make_predictions(model, df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Please add funtion comment
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Please add funtion comment
root.mainloop()

