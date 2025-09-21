import os
import pandas as pd
import smtplib
from flask import Flask, render_template, request, jsonify
from supabase import create_client, Client
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import date
from apscheduler.schedulers.background import BackgroundScheduler
from ml_model import load_model_and_predict

# --- Basic Setup & Config ---
load_dotenv()
app = Flask(__name__, template_folder='templates')
scheduler = BackgroundScheduler(daemon=True)

# Supabase and Email Config...
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = os.getenv("EMAIL_PORT")


# --- REAL-TIME DATA PROCESSING & ML PREDICTION ---
def update_student_profiles(use_ml_model=True):
    """Recalculates and updates all student profiles in the database."""
    try:
        print("Recalculating all student profiles...")
        attendance_summary_df = calculate_overall_attendance()
        
        # We need the local files to get the full student list and scores
        base_info_df = pd.read_csv('data/attendance.csv')
        scores_df = pd.read_csv('data/scores.csv')
        students_df = pd.merge(base_info_df, scores_df, on='student_id', how='left')

        if not attendance_summary_df.empty:
            students_df = pd.merge(students_df.drop(columns=['attendance_percentage'], errors='ignore'), attendance_summary_df, on='student_id', how='left')
        
        students_df.fillna({'attendance_percentage': 0, 'average_score': 50, 'exam_attempts': 1, 'fee_status': 'Paid'}, inplace=True)
        
        students_df['attendance_percentage'] = students_df['attendance_percentage'].astype(int)
        students_df['average_score'] = students_df['average_score'].astype(int)
        students_df['exam_attempts'] = students_df['exam_attempts'].astype(int)
        students_df['class'] = students_df['class'].astype(int)

        if use_ml_model:
            print("   -> Using ML model for risk prediction.")
            students_df['risk_level'] = load_model_and_predict(students_df)
        
        supabase.from_('students').upsert(students_df.to_dict(orient='records'), on_conflict='student_id').execute()
        print("✅ Successfully updated profiles.")
        return True
    except Exception as e:
        print(f"❌ An error occurred while updating student profiles: {e}")
        return False

def calculate_overall_attendance():
    response = supabase.from_('daily_attendance').select('student_id, status').execute()
    if not response.data: return pd.DataFrame(columns=['student_id', 'attendance_percentage'])
    df = pd.DataFrame(response.data)
    summary = df.groupby('student_id')['status'].agg(total='count', present=lambda x: (x == 'Present').sum()).reset_index()
    summary['attendance_percentage'] = round((summary['present'] / summary['total']) * 100)
    return summary[['student_id', 'attendance_percentage']]

def scheduled_alert_job():
    print("\n--- ⏰ Running Scheduled Weekly Alert Job ---")
    with app.app_context():
        try:
            response = supabase.from_('students').select('*').in_('risk_level', ['High', 'Medium']).execute()
            if not response.data:
                print("No at-risk students found. No alerts sent.")
                return
            print(f"Found {len(response.data)} at-risk students. Sending alerts...")
            for student in response.data: send_email_alert(student)
            print("--- ✅ Scheduled Job Complete ---")
        except Exception as e:
            print(f"--- ❌ Error in scheduled job: {e} ---")

# --- FLASK ROUTES ---
@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html', date=date)

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        response = supabase.from_('students').select('*').order('student_id').execute()
        resp = jsonify(response.data)
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/kpi-stats', methods=['GET'])
def get_kpi_stats():
    try:
        response = supabase.from_('students').select('risk_level, attendance_percentage, fee_status').execute()
        if not response.data: return jsonify({"error": "No data found"}), 404
        df = pd.DataFrame(response.data)
        return jsonify({
            "total_students": len(df),
            "high_risk_count": len(df[df['risk_level'] == 'High']),
            "average_attendance": int(df['attendance_percentage'].mean()),
            "overdue_fees_count": len(df[df['fee_status'] == 'Overdue'])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    try:
        response = supabase.from_('students').select('risk_level, attendance_percentage, average_score').execute()
        if not response.data: return jsonify({"error": "No data found"}), 404
        df = pd.DataFrame(response.data)
        risk_counts = df['risk_level'].value_counts().to_dict()
        scatter_data = df[['attendance_percentage', 'average_score']].to_dict(orient='records')
        return jsonify({"risk_distribution": risk_counts, "attendance_vs_scores": scatter_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mentor-suggestion/<int:student_id>', methods=['GET'])
def get_mentor_suggestion(student_id):
    try:
        response = supabase.from_('students').select('risk_level, attendance_percentage, average_score, exam_attempts').eq('student_id', student_id).single().execute()
        if not response.data: return jsonify({"error": "Student not found"}), 404
        student = response.data
        suggestion = "Student is performing well. Continue to provide encouragement and monitor progress."
        if student['risk_level'] == 'High':
            suggestion = "High Priority: The model predicts a high risk. This student's low attendance and scores require immediate intervention. Recommend a parent-teacher meeting to discuss a personalized support plan."
        elif student['risk_level'] == 'Medium':
            if student['attendance_percentage'] < 75:
                suggestion = "The model predicts a medium risk, primarily due to low attendance. A follow-up conversation is needed to understand the reasons for absence and reinforce the importance of regular classes."
            elif student['average_score'] < 60:
                suggestion = "The model predicts a medium risk because academic scores are dropping. Suggest scheduling extra tutorial sessions and focusing on weaker subjects."
            else:
                suggestion = "The model predicts a medium risk. While individual metrics aren't critical, the overall pattern is concerning. Recommend a check-in to discuss any challenges the student may be facing."
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        return jsonify({"suggestion": "An error occurred while generating the suggestion."}), 500

@app.route('/get-student-attendance/<int:student_id>', methods=['GET'])
def get_student_attendance(student_id):
    try:
        response = supabase.from_('daily_attendance').select('date, status').eq('student_id', student_id).order('date', desc=True).limit(7).execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-student-full-attendance/<int:student_id>', methods=['GET'])
def get_student_full_attendance(student_id):
    try:
        response = supabase.from_('daily_attendance').select('date, status').eq('student_id', student_id).order('date', desc=False).execute()
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    records = request.get_json()
    today_str = date.today().strftime("%Y-%m-%d")
    data_to_upsert = [{'student_id': r['student_id'], 'date': today_str, 'status': r['status']} for r in records]
    try:
        supabase.from_('daily_attendance').upsert(data_to_upsert, on_conflict='student_id,date').execute()
        update_student_profiles()
        return jsonify({"status": "success", "message": "Attendance recorded and profiles updated!"})
    except Exception as e:
        return jsonify({"status": "error", "message": "Failed to save attendance."}), 500

@app.route('/update-historical-attendance', methods=['POST'])
def update_historical_attendance():
    records = request.get_json()
    if not records: return jsonify({"status": "error", "message": "No records to update."}), 400
    try:
        supabase.from_('daily_attendance').upsert(records, on_conflict='student_id,date').execute()
        update_student_profiles()
        return jsonify({"status": "success", "message": "Historical attendance updated successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to update history: {e}"}), 500

def send_email_alert(student):
    # ... email logic ...
    pass

@app.route('/send-bulk-alert', methods=['POST'])
def send_bulk_alert():
    # ... bulk alert logic ...
    pass

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    scheduler.add_job(scheduled_alert_job, 'cron', day_of_week='mon', hour=9)
    scheduler.start()
    print("\n⏰ Automated weekly alerts scheduled for every Monday at 9:00 AM.")
    print(f"\n🎉 Server is running! Access the dashboard at http://127.0.0.1:5000")
    
    # This part is for local development convenience, will be ignored by Gunicorn on Render
    is_local_run = os.environ.get("RENDER") is None
    if is_local_run:
        import webbrowser
        import threading
        url = "http://127.0.0.1:5000"
        threading.Timer(1.25, lambda: webbrowser.open(url)).start()
        app.run(debug=True, port=5000)

