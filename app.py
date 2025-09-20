import os
import pandas as pd
import smtplib
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify
from supabase import create_client, Client
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import date

# --- Basic Setup & Config ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# Supabase and Email Config...
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = os.getenv("EMAIL_PORT")

# --- ONE-TIME SETUP LOGIC (Previously process_data.py) ---
def run_initial_setup():
    """Consolidated function to set up the database on first run."""
    print("--- Starting Initial Data Setup ---")
    try:
        # Step 1: Load all data into memory first
        print("Step 1: Loading local CSV files...")
        historical_df = pd.read_csv('data/historical_attendance.csv')
        base_info_df = pd.read_csv('data/attendance.csv')
        scores_df = pd.read_csv('data/scores.csv')

        # Step 2: Upload historical attendance
        print("\nStep 2: Uploading historical attendance data...")
        supabase.from_('daily_attendance').delete().neq('student_id', 0).execute()
        for i in range(0, len(historical_df), 500):
            chunk = historical_df[i:i + 500]
            supabase.from_('daily_attendance').insert(chunk.to_dict(orient='records')).execute()
        print("✅ Historical attendance upload complete.")

        # --- FIX: Calculate attendance directly from the file to avoid DB lag ---
        print("\nStep 3: Calculating accurate attendance percentages from local file...")
        summary = historical_df.groupby('student_id')['status'].agg(
            total='count', 
            present=lambda x: (x == 'Present').sum()
        ).reset_index()
        summary['attendance_percentage'] = round((summary['present'] / summary['total']) * 100)
        attendance_summary_df = summary[['student_id', 'attendance_percentage']]
        print("✅ Attendance percentages calculated.")
        
        # Step 4: Create and upload final student profiles
        print("\nStep 4: Merging all data and creating final student profiles...")
        students_df = pd.merge(base_info_df, scores_df, on='student_id', how='left')
        students_df = pd.merge(students_df.drop(columns=['attendance_percentage']), attendance_summary_df, on='student_id', how='left')
        
        students_df['attendance_percentage'] = students_df['attendance_percentage'].fillna(0).astype(int)
        students_df['exam_attempts'] = students_df['exam_attempts'].fillna(0).astype(int)
        students_df['risk_level'] = students_df.apply(calculate_risk, axis=1)
        
        supabase.from_('students').upsert(students_df.to_dict(orient='records'), on_conflict='student_id').execute()
        print(f"✅ Successfully created profiles for {len(students_df)} students.")

    except Exception as e:
        print(f"❌ ERROR during initial setup: {e}")
    print("\n--- Data Setup Complete ---")

# --- REAL-TIME DATA PROCESSING ---
def calculate_overall_attendance():
    """Calculates attendance for all students directly from Supabase."""
    response = supabase.from_('daily_attendance').select('student_id, status').execute()
    if not response.data: return pd.DataFrame(columns=['student_id', 'attendance_percentage'])
    df = pd.DataFrame(response.data)
    summary = df.groupby('student_id')['status'].agg(
        total='count', present=lambda x: (x == 'Present').sum()
    ).reset_index()
    summary['attendance_percentage'] = round((summary['present'] / summary['total']) * 100)
    return summary[['student_id', 'attendance_percentage']]

def calculate_risk(row):
    """Determines a student's risk level."""
    if row['attendance_percentage'] < 70 and row['average_score'] < 50: return 'High'
    if row['attendance_percentage'] < 75 or row['average_score'] < 60 or row['exam_attempts'] > 3: return 'Medium'
    return 'Low'

def update_student_profiles():
    """Recalculates and updates profiles in Supabase after new attendance is marked."""
    try:
        print("Recalculating all student profiles...")
        attendance_summary_df = calculate_overall_attendance()
        base_info_res = supabase.from_('students').select('*').execute()
        if not base_info_res.data: return False

        students_df = pd.DataFrame(base_info_res.data)
        if not attendance_summary_df.empty:
            students_df = pd.merge(students_df.drop(columns=['attendance_percentage'], errors='ignore'), attendance_summary_df, on='student_id', how='left')
        
        students_df['attendance_percentage'] = students_df['attendance_percentage'].fillna(0).astype(int)
        students_df['risk_level'] = students_df.apply(calculate_risk, axis=1)
        
        columns_to_update = ['student_id', 'attendance_percentage', 'risk_level']
        data_to_upload = students_df[columns_to_update].to_dict(orient='records')
        
        supabase.from_('students').upsert(data_to_upload, on_conflict='student_id').execute()
        print("Successfully updated profiles.")
        return True
    except Exception as e:
        print(f"An error occurred while updating student profiles: {e}")
        return False

# --- FLASK ROUTES ---
@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html', date=date)

@app.route('/api/students', methods=['GET'])
def get_students():
    """Fetches a list of all students with their full details."""
    try:
        response = supabase.from_('students').select('*').order('student_id').execute()
        resp = jsonify(response.data)
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route('/get-student-suggestion/<int:student_id>', methods=['GET'])
def get_student_suggestion(student_id):
    # This function remains unchanged
    try:
        response = supabase.from_('students').select('attendance_percentage, average_score, exam_attempts').eq('student_id', student_id).single().execute()
        student = response.data
        suggestion = "Student is on track. Encourage continued engagement."
        if student['attendance_percentage'] < 75 and student['average_score'] < 60:
            suggestion = "High Priority: Both attendance and scores are low. Recommend immediate counseling and a revised study plan."
        elif student['attendance_percentage'] < 75:
            suggestion = "Attendance is a key concern. Follow-up to understand reasons for absence."
        elif student['average_score'] < 60:
            suggestion = "Academic scores are dropping. Suggest scheduling extra tutorial sessions."
        elif student['exam_attempts'] > 3:
            suggestion = "Multiple exam attempts indicate a foundational gap. Recommend a one-on-one concept review session."
        
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    # This function remains unchanged
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
    """Updates one or more past attendance records for a student."""
    records = request.get_json()
    if not records:
        return jsonify({"status": "error", "message": "No records to update."}), 400
    try:
        supabase.from_('daily_attendance').upsert(records, on_conflict='student_id,date').execute()
        update_student_profiles()
        return jsonify({"status": "success", "message": "Historical attendance updated successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to update history: {e}"}), 500

# --- EMAIL ALERT LOGIC ---
def send_email_alert(student):
    # This function remains unchanged
    subject = f"Academic Alert for Student: {student.get('name', 'N/A')}"
    html_body = f"""<h3>Academic Alert for {student.get('name', 'N/A')} (ID: {student.get('student_id')})</h3><p>A risk has been detected for this student based on the following data:</p><ul><li><strong>Risk Level:</strong> {student.get('risk_level', 'N/A')}</li><li><strong>Overall Attendance:</strong> {student.get('attendance_percentage', 0)}%</li><li><strong>Average Score:</strong> {student.get('average_score', 0)}%</li><li><strong>Exam Attempts:</strong> {student.get('exam_attempts', 0)}</li></ul><p>Please check the dashboard for more details and consider taking appropriate action.</p>"""
    to_emails = [email for email in [student.get('mentor_email'), student.get('guardian_email')] if email]
    if not to_emails:
        print(f"No valid emails found for student {student['student_id']}. Skipping email.")
        return
    msg = MIMEMultipart()
    msg['From'] = EMAIL_HOST_USER
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))
    try:
        server = smtplib.SMTP(EMAIL_HOST, int(EMAIL_PORT))
        server.starttls()
        server.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
        server.sendmail(EMAIL_HOST_USER, to_emails, msg.as_string())
        server.quit()
        print(f"Successfully sent email for student {student['student_id']}")
    except Exception as e:
        print(f"Failed to send email for student {student['student_id']}: {e}")

@app.route('/send-bulk-alert', methods=['POST'])
def send_bulk_alert():
    # This function remains unchanged
    student_ids = request.get_json().get('studentIds')
    if not student_ids:
        return jsonify({"status": "error", "message": "No student IDs provided."}), 400
    try:
        response = supabase.from_('students').select('*').in_('student_id', student_ids).execute()
        if not response.data:
            return jsonify({"status": "error", "message": "Students not found."}), 404
        for student in response.data:
            send_email_alert(student)
        return jsonify({"status": "success", "message": f"Alerts sent for {len(response.data)} students."})
    except Exception as e:
        print(f"Error sending bulk emails: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # This block is for local development only.
    # The live server will use Gunicorn to run the app.
    run_initial_setup()
    
    url = "http://127.0.0.1:5000"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    
    print(f"\n🎉 Server is running! Opening browser to {url}")
    app.run(debug=True, port=5000)

# The following is needed for the production server
# gunicorn app:app