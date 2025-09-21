import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_and_save_model():
    """
    This function trains a Decision Tree model on the historical data
    and saves it to a file for later use.
    """
    print("   -> Training new machine learning model...")
    
    try:
        attendance_df = pd.read_csv('data/historical_attendance.csv')
        scores_df = pd.read_csv('data/scores.csv')
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: Could not find data file for training. {e}")
        return None

    attendance_summary = attendance_df.groupby('student_id')['status'].agg(
        total='count', 
        present=lambda x: (x == 'Present').sum()
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    model_filename = 'risk_model.joblib'
    joblib.dump(model, model_filename)
    joblib.dump(le, 'fee_status_encoder.joblib') 
    
    print(f"   ✅ Model trained and saved as '{model_filename}'")
    return model_filename

def load_model_and_predict(student_data_df):
    """
    Loads the saved model and predicts risk levels for a new set of students.
    """
    try:
        model = joblib.load('risk_model.joblib')
        encoder = joblib.load('fee_status_encoder.joblib')
    except FileNotFoundError:
        print("   ⚠️ Model file not found. Retraining model...")
        train_and_save_model()
        model = joblib.load('risk_model.joblib')
        encoder = joblib.load('fee_status_encoder.joblib')

    df_copy = student_data_df.copy()
    df_copy['fee_status_encoded'] = encoder.transform(df_copy['fee_status'])
    features = ['attendance_percentage', 'average_score', 'exam_attempts', 'fee_status_encoded']
    
    predictions = model.predict(df_copy[features])
    return predictions

