import os
import sys
import json
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response, redirect, url_for
from flask_socketio import SocketIO, emit

from config import (
    BASE_DIR, AGE_GROUPS, MEDICAL_DISCLAIMER, LIVE_RECORDINGS_DIR, 
    UPLOAD_RECORDINGS_DIR, REPORTS_DIR, DATASETS_DIR, RECORDINGS_DIR
)
from database.models import init_db, get_db_connection
from ml.inference import InferencePipeline
from services.audio_service import AudioService
from services.ai_assistant import AIAssistantService
from services.report_service import ReportService

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pediatric-ai-secret-key-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

audio_service = AudioService()
assistant_service = AIAssistantService()
report_service = ReportService()

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

# Disable browser response caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# --- HTML FRONTEND ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', age_groups=AGE_GROUPS, disclaimer=MEDICAL_DISCLAIMER)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', age_groups=AGE_GROUPS, disclaimer=MEDICAL_DISCLAIMER)

@app.route('/live')
def live():
    return render_template('live.html', age_groups=AGE_GROUPS, disclaimer=MEDICAL_DISCLAIMER)

@app.route('/upload')
def upload():
    return render_template('upload.html', age_groups=AGE_GROUPS, disclaimer=MEDICAL_DISCLAIMER)

@app.route('/cry-model-test')
def cry_model_test():
    return render_template('cry_model_test.html', disclaimer=MEDICAL_DISCLAIMER)

@app.route('/dataset')
def dataset():
    return redirect(url_for('dashboard'))

@app.route('/history')
def history():
    return render_template('history.html', disclaimer=MEDICAL_DISCLAIMER)

@app.route('/analysis/<int:record_id>')
def analysis_detail(record_id):
    conn = get_db_connection()
    pred = conn.execute('''
        SELECT p.*, a.filename, a.filepath, a.audio_type, a.duration, a.source, s.input_type, s.child_id, c.age_months
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        LEFT JOIN monitoring_sessions s ON a.session_id = s.id
        LEFT JOIN children c ON s.child_id = c.id
        WHERE p.id = ?
    ''', (record_id,)).fetchone()
    conn.close()

    if not pred:
        return redirect(url_for('history'))