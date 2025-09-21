import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from supabase import create_client, Client
import os
import threading
from tkinterdnd2 import DND_FILES, TkinterDnD
from dotenv import load_dotenv

# --- Machine Learning Logic (integrated) ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save_model(attendance_df, scores_df, progress_callback):
    """Trains a model on the provided dataframes and saves it."""
    progress_callback("-> Training Machine Learning model...")
    
    attendance_summary = attendance_df.groupby('student_id')['status'].agg(
        total='count', present=lambda x: (x == 'Present').sum()
    ).reset_index()
    attendance_summary['attendance_percentage'] = (attendance_summary['present'] / attendance_summary['total']) * 100
    
    training_data = pd.merge(scores_df, attendance_summary[['student_id', 'attendance_percentage']], on='student_id')

    def create_target_label(row):
        if row['attendance_percentage'] < 70 and row['average_score'] < 50: return 'High'
        if row['attendance_percentage'] < 75 or row['average_score'] < 60 or row['exam_attempts'] > 3: return 'Medium'
        return 'Low'
    
    training_data['risk_level'] = training_data.apply(create_target_label, axis=1)

    le = LabelEncoder()
    training_data['fee_status_encoded'] = le.fit_transform(training_data['fee_status'])

    features = ['attendance_percentage', 'average_score', 'exam_attempts', 'fee_status_encoded']
    target = 'risk_level'
    X = training_data[features]
    y = training_data[target]
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)
    
    joblib.dump(model, 'risk_model.joblib')
    joblib.dump(le, 'fee_status_encoder.joblib')
    progress_callback("✅ Model trained and saved as 'risk_model.joblib'.")
    return model, le

# --- Main Application Class ---
class DataUploaderApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("TeamVidya - Initial Data Setup")
        self.geometry("750x700")
        self.resizable(False, False)
        self.configure(bg="#2b2b2b")

        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2b2b2b')
        self.style.configure('TLabel', background='#2b2b2b', foreground='#dcdcdc', font=('Helvetica', 10))
        self.style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 10, 'italic'))
        self.style.configure('Accent.TButton', background='#007acc', foreground='white', font=('Helvetica', 12, 'bold'), borderwidth=0)
        self.style.map('Accent.TButton', background=[('active', '#005f9e')])
        
        self.file_paths = {"students": None, "scores": None, "attendance": None}
        self.supabase_creds = {"url": None, "key": None}

        self._create_widgets()
        self.load_env_credentials()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="25")
        main_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(main_frame, text="TeamVidya Initial Setup", style='Header.TLabel').pack(pady=(0, 5))
        ttk.Label(main_frame, text="Drag your data files below to begin the one-time setup process.", foreground="#888888").pack(pady=(0, 20))

        cred_frame = ttk.Frame(main_frame, padding=10, relief="solid", borderwidth=1)
        cred_frame.pack(fill=tk.X, pady=10)
        self.cred_status_label = ttk.Label(cred_frame, text="🔍 Searching for .env file...", style='Status.TLabel', foreground="#d29922")
        self.cred_status_label.pack()

        drop_area = ttk.Frame(main_frame)
        drop_area.pack(fill=tk.X, pady=15, expand=True)
        drop_area.grid_columnconfigure((0,1,2), weight=1)

        self.drop_zones = {}
        for i, (key, text) in enumerate([("students", "Student Info"), ("scores", "Scores Data"), ("attendance", "Attendance History")]):
            frame = ttk.Frame(drop_area, padding=10)
            frame.grid(row=0, column=i, sticky="nsew", padx=10)
            
            canvas = tk.Canvas(frame, bg="#3c3c3c", bd=2, relief="groove", highlightthickness=0)
            canvas.pack(expand=True, fill=tk.BOTH)
            
            label = ttk.Label(canvas, text=f"📂\n\n{text}", wraplength=150, justify=tk.CENTER, background="#3c3c3c", foreground="#a0a0a0", font=('Helvetica', 11))
            label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            
            canvas.drop_target_register(DND_FILES)
            canvas.dnd_bind('<<Drop>>', lambda e, k=key: self.on_drop(e, k))
            self.drop_zones[key] = {'canvas': canvas, 'label': label}

        self.upload_button = ttk.Button(main_frame, text="Start Setup", style='Accent.TButton', command=self.start_upload_thread, state=tk.DISABLED)
        self.upload_button.pack(pady=20, ipady=10, fill=tk.X)

        progress_frame = ttk.Frame(main_frame, relief='solid', borderwidth=1)
        progress_frame.pack(expand=True, fill=tk.BOTH, pady=10)
        self.progress_text = tk.Text(progress_frame, height=10, bg="#1e1e1e", fg="#dcdcdc", bd=0, highlightthickness=0, relief='flat', font=('Consolas', 10))
        self.progress_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

    def load_env_credentials(self):
        load_dotenv()
        self.supabase_creds["url"] = os.getenv("SUPABASE_URL")
        self.supabase_creds["key"] = os.getenv("SUPABASE_KEY")

        if self.supabase_creds["url"] and self.supabase_creds["key"]:
            self.cred_status_label.config(text="✅ Supabase credentials loaded successfully from .env file.", foreground="#238636")
        else:
            self.cred_status_label.config(text="❌ .env file not found or is missing Supabase credentials.", foreground="#f85149")

    def check_all_files_dropped(self):
        if all(self.file_paths.values()):
            self.upload_button.config(state=tk.NORMAL)

    def on_drop(self, event, key):
        filepath = event.data.strip('{}')
        if filepath.lower().endswith(('.xlsx', '.xls', '.csv')):
            self.file_paths[key] = filepath
            filename = os.path.basename(filepath)
            
            zone = self.drop_zones[key]
            zone['canvas'].config(bg="#2d332d", relief="solid", bd=2)
            zone['label'].config(text=f"✔️\n\n{filename}", background="#2d332d", foreground="#238636")
            self.check_all_files_dropped()
        else:
            messagebox.showerror("Invalid File", "Please drop a valid Excel (.xlsx, .xls) or CSV (.csv) file.")
    
    def log_progress(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.update_idletasks()

    def start_upload_thread(self):
        if not self.supabase_creds["url"] or not self.supabase_creds["key"]:
            messagebox.showerror("Credentials Missing", "Cannot proceed without Supabase credentials. Please check your .env file.")
            return

        self.upload_button.config(state=tk.DISABLED, text="Processing...")
        self.progress_text.delete('1.0', tk.END)
        
        thread = threading.Thread(target=self.run_setup_process)
        thread.start()

    def run_setup_process(self):
        try:
            self.log_progress("Connecting to Supabase...")
            supabase: Client = create_client(self.supabase_creds["url"], self.supabase_creds["key"])
            self.log_progress("✅ Connection successful.")

            self.log_progress("\nReading local data files...")
            base_info_df = pd.read_csv(self.file_paths["students"]) if self.file_paths["students"].endswith('.csv') else pd.read_excel(self.file_paths["students"])
            scores_df = pd.read_csv(self.file_paths["scores"]) if self.file_paths["scores"].endswith('.csv') else pd.read_excel(self.file_paths["scores"])
            historical_df = pd.read_csv(self.file_paths["attendance"]) if self.file_paths["attendance"].endswith('.csv') else pd.read_excel(self.file_paths["attendance"])
            self.log_progress("✅ All files read successfully.")

            model, encoder = train_and_save_model(historical_df, scores_df, self.log_progress)

            self.log_progress("\nUploading historical attendance data...")
            supabase.from_('daily_attendance').delete().neq('student_id', 0).execute()
            for i in range(0, len(historical_df), 500):
                chunk = historical_df[i:i + 500]
                supabase.from_('daily_attendance').insert(chunk.to_dict(orient='records')).execute()
            self.log_progress("✅ Historical attendance upload complete.")

            self.log_progress("\nCreating final student profiles with ML predictions...")
            students_df = pd.merge(base_info_df, scores_df, on='student_id', how='left')
            attendance_summary = historical_df.groupby('student_id')['status'].agg(total='count', present=lambda x: (x == 'Present').sum()).reset_index()
            attendance_summary['attendance_percentage'] = (attendance_summary['present'] / attendance_summary['total']) * 100
            students_df = pd.merge(students_df.drop(columns=['attendance_percentage']), attendance_summary[['student_id', 'attendance_percentage']], on='student_id', how='left')
            students_df.fillna({'attendance_percentage': 0, 'average_score': 50, 'exam_attempts': 1, 'fee_status': 'Paid'}, inplace=True)
            
            students_df['attendance_percentage'] = students_df['attendance_percentage'].astype(int)
            students_df['average_score'] = students_df['average_score'].astype(int)
            students_df['exam_attempts'] = students_df['exam_attempts'].astype(int)
            
            valid_students = students_df.dropna(subset=['attendance_percentage', 'average_score', 'exam_attempts', 'fee_status'])
            if not valid_students.empty:
                valid_students_copy = valid_students.copy()
                valid_students_copy['fee_status_encoded'] = encoder.transform(valid_students_copy['fee_status'])
                features = ['attendance_percentage', 'average_score', 'exam_attempts', 'fee_status_encoded']
                predictions = model.predict(valid_students_copy[features])
                students_df.loc[valid_students.index, 'risk_level'] = predictions
            
            supabase.from_('students').upsert(students_df.to_dict(orient='records'), on_conflict='student_id').execute()
            self.log_progress(f"✅ Successfully created/updated profiles for {len(students_df)} students.")
            self.log_progress("\n--- 🎉 ALL DONE! Your database is ready. ---")
            messagebox.showinfo("Success", "The initial data setup is complete! You can now close this tool and run your main web application.")

        except Exception as e:
            self.log_progress(f"\n--- ❌ AN ERROR OCCURRED ---")
            self.log_progress(str(e))
            messagebox.showerror("Error", f"An error occurred during the setup process. Please check the log for details.\n\n{e}")
        finally:
            self.upload_button.config(state=tk.NORMAL, text="Start Setup")

if __name__ == "__main__":
    app = DataUploaderApp()
    app.mainloop()

