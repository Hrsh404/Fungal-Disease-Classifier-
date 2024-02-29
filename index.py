
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font as tkfont
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class FungalDiseaseClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Fungal Disease Classifier")
        master.geometry("500x550")  # initial size of the window (Sir/Ma'am set window size according to your computer screen.)

       
        self.style = ttk.Style(master)
        self.style.theme_use('clam')  # Try 'alt', 'default', 'clam', 'classic' to see what fits your taste

  
        self.title_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        self.button_font = tkfont.Font(family="Helvetica", size=10)


        self.title_label = ttk.Label(master, text="Fungal Disease Classifier ONPRICE INFOTECH", font=self.title_font)
        self.title_label.pack(pady=10)

   
        self.load_button = ttk.Button(master, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(pady=5)

        self.train_button = ttk.Button(master, text="Train Model", command=self.train_model, state='disabled')
        self.train_button.pack(pady=5)

        self.evaluate_button = ttk.Button(master, text="Evaluate Model", command=self.evaluate_model, state='disabled')
        self.evaluate_button.pack(pady=5)

        self.clf = None  
        self.df = None  # Placeholder for the dataframe

    def load_dataset(self):
        file_path = "C:\\Users\\DELL\\Downloads\\ONPRICE InfotechProject\\fungal_disease_oilseeds_dataset_varied.csv"
        self.df = pd.read_csv(file_path)
        self.preprocess_dataset()
        self.train_button['state'] = '!disabled'
        messagebox.showinfo("Dataset Loaded", "The dataset has been loaded successfully!")

    def preprocess_dataset(self):
        le = LabelEncoder()
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = le.fit_transform(self.df[col])

    def train_model(self):
        X = self.df.drop('Disease Name', axis=1)
        y = self.df['Disease Name']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.clf = RandomForestClassifier(n_estimators=10, random_state=42)
        self.clf.fit(X_train, y_train)

        self.X_test, self.y_test = X_test, y_test  # Store for evaluation
        self.evaluate_button['state'] = '!disabled'
        messagebox.showinfo("Model Trained", "The model has been trained successfully!")

    def evaluate_model(self):
        if self.clf:
            y_pred = self.clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)

            messagebox.showinfo("Evaluation Metrics", f"Accuracy: {accuracy*100:.2f}%\n\nClassification Report:\n{report}")
            self.visualize_feature_importance()
            self.visualize_confusion_matrix(y_pred)

    def visualize_feature_importance(self):
        feature_importance_values = self.clf.feature_importances_
        features = self.X_test.columns
        sorted_indices = feature_importance_values.argsort()

        plt.figure(figsize=(10, 7))
        plt.title("Feature Importances")
        sns.barplot(x=feature_importance_values[sorted_indices], y=features[sorted_indices])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

    def visualize_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(self.y_test), yticklabels=np.unique(y_pred))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = FungalDiseaseClassifierApp(root)
    root.mainloop()
