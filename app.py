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

    pred_dict = dict(pred)
    if pred_dict.get('probabilities_json'):
        pred_dict['probabilities'] = json.loads(pred_dict['probabilities_json'])
    else:
        pred_dict['probabilities'] = {}

    # AI assistant generation
    inference_stub = {
        "detected_sound": pred_dict.get("sound_type"),
        "sound_confidence": pred_dict.get("confidence", 0.0),
        "top_screening_pattern": pred_dict.get("screening_prediction"),
        "risk_result": {
            "risk_level": "HIGH" if pred_dict.get("risk_score", 0) >= 65 else "MODERATE" if pred_dict.get("risk_score", 0) >= 35 else "LOW",
            "risk_score": pred_dict.get("risk_score", 0),
            "risk_drivers": [pred_dict.get("screening_prediction")]
        }
    }
    ai_explanation = assistant_service.explain_results(inference_stub)

    return render_template(
        'analysis.html',
        record=pred_dict,
        explanation=ai_explanation,
        disclaimer=MEDICAL_DISCLAIMER
    )

# --- REST API ENDPOINTS ---

@app.route('/api/audio/upload', methods=['POST'])
def api_upload_audio():
    file = None
    for k in ['audio_file', 'audio', 'file']:
        if k in request.files:
            file = request.files[k]
            break

    if not file or file.filename == '':
        return jsonify({"error": "No valid audio file provided"}), 400

    age_group = request.form.get('age_group', '0-12m')
    audio_hint = request.form.get('audio_type_hint', 'UNKNOWN')

    # Caregiver optional symptoms
    symptoms = {
        "temperature": float(request.form.get('temperature')) if request.form.get('temperature') else None,
        "spo2": float(request.form.get('spo2')) if request.form.get('spo2') else None,
        "fever": request.form.get('fever') == 'yes',
        "difficulty_breathing": request.form.get('difficulty_breathing') == 'yes',
        "wheezing": request.form.get('wheezing') == 'yes',
        "reduced_feeding": request.form.get('reduced_feeding') == 'yes',
        "lethargy": request.form.get('lethargy') == 'yes',
        "blue_lips": request.form.get('blue_lips') == 'yes'
    }

    try:
        filename, filepath = audio_service.save_uploaded_file(file)

        # Execute Unified Inference dynamically
        pipeline = InferencePipeline()
        result = pipeline.process_audio(
            filepath,
            age_group=age_group,
            audio_type_hint=audio_hint,
            caregiver_symptoms=symptoms
        )
        print(f"DEBUG API UPLOAD: filepath={filepath}, detected={result.get('detected_sound')}, conf={result.get('sound_confidence')}, pattern={result.get('top_screening_pattern')}", flush=True)

        # Save to SQLite Database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO monitoring_sessions (input_type, risk_level, risk_score) VALUES (?, ?, ?)',
                       ('UPLOAD', result['risk_result']['risk_level'], result['risk_result']['risk_score']))
        session_id = cursor.lastrowid

        cursor.execute('''
            INSERT INTO audio_records (session_id, filename, filepath, audio_type, duration, source)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, filename, filepath, result['detected_sound'], result['duration'], 'UPLOAD'))
        audio_id = cursor.lastrowid

        probs_clean = {str(k): float(v) for k, v in result['screening_probabilities'].items()}
        probs_json = json.dumps(probs_clean)
        cursor.execute('''
            INSERT INTO predictions (audio_id, sound_type, confidence, screening_prediction, probabilities_json, risk_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (audio_id, result['detected_sound'], result['sound_confidence'], result['top_screening_pattern'], probs_json, result['risk_result']['risk_score']))
        pred_id = cursor.lastrowid

        conn.commit()
        conn.close()

        # Generate AI Clinical & IEEE Literature Analysis Explanation
        ai_explanation = assistant_service.explain_results(result, age_group=age_group)
        result['ai_explanation'] = ai_explanation
        result['prediction_id'] = pred_id
        result['audio_url'] = f"/recordings/uploads/{filename}"
        return jsonify(sanitize_for_json(result))
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"ERROR IN UPLOAD API: {err_msg}", flush=True)
        return jsonify({"error": str(e), "trace": err_msg}), 500

@app.route('/api/audio/live_analyze', methods=['POST'])
def api_live_analyze_audio():
    file = None
    for k in ['audio_file', 'audio', 'file']:
        if k in request.files:
            file = request.files[k]
            break

    if not file or file.filename == '':
        return jsonify({"error": "No valid audio file provided"}), 400
    age_group = request.form.get('age_group', '0-12m')
    audio_hint = request.form.get('audio_type_hint', 'UNKNOWN')

    try:
        filename, filepath = audio_service.save_uploaded_file(file)

        pipeline = InferencePipeline()
        result = pipeline.process_audio(
            filepath,
            age_group=age_group,
            audio_type_hint=audio_hint
        )

        ai_explanation = assistant_service.explain_results(result, age_group=age_group)
        result['ai_explanation'] = ai_explanation
        result['audio_url'] = f"/recordings/uploads/{filename}"
        return jsonify(sanitize_for_json(result))
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"ERROR IN LIVE ANALYZE API: {err_msg}", flush=True)
        return jsonify({"error": str(e), "trace": err_msg}), 500

@app.route('/api/datasets', methods=['GET'])
def api_get_datasets():
    manifest_path = os.path.join(DATASETS_DIR, "dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return jsonify(manifest)
    return jsonify([])

@app.route('/api/datasets/analyze', methods=['POST'])
def api_analyze_dataset_sample():
    data = request.json or {}
    filepath = data.get('filepath')
    age_group = data.get('age_group', '0-12m')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Dataset sample file not found"}), 404

    result = pipeline_instance.process_audio(filepath, age_group=age_group)

    # Save record to DB for demonstration log
    filename = os.path.basename(filepath)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO monitoring_sessions (input_type, risk_level, risk_score) VALUES (?, ?, ?)',
                   ('DATASET', result['risk_result']['risk_level'], result['risk_result']['risk_score']))
    session_id = cursor.lastrowid

    cursor.execute('''
        INSERT INTO audio_records (session_id, filename, filepath, audio_type, duration, source)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, filename, filepath, result['detected_sound'], result['duration'], 'DATASET'))
    audio_id = cursor.lastrowid

    probs_clean = {str(k): float(v) for k, v in result['screening_probabilities'].items()}
    probs_json = json.dumps(probs_clean)
    cursor.execute('''
        INSERT INTO predictions (audio_id, sound_type, confidence, screening_prediction, probabilities_json, risk_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (audio_id, result['detected_sound'], result['sound_confidence'], result['top_screening_pattern'], probs_json, result['risk_result']['risk_score']))
    pred_id = cursor.lastrowid

    conn.commit()
    conn.close()

    result['prediction_id'] = pred_id
    return jsonify(result)

@app.route('/api/history', methods=['GET'])
def api_get_history():
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT p.id, p.sound_type, p.confidence, p.screening_prediction, p.risk_score, p.created_at,
               a.filename, a.duration, a.source
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        ORDER BY p.id DESC
        LIMIT 50
    ''').fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        item = dict(r)
        item['risk_level'] = 'HIGH' if item['risk_score'] >= 65 else 'MODERATE' if item['risk_score'] >= 35 else 'LOW'
        history_list.append(item)

    return jsonify(history_list)

@app.route('/api/report/<int:pred_id>/pdf', methods=['GET'])
def api_download_pdf(pred_id):
    conn = get_db_connection()
    row = conn.execute('''
        SELECT p.*, a.filename, a.duration, a.source
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        WHERE p.id = ?
    ''', (pred_id,)).fetchone()
    conn.close()

    if not row:
        return "Record not found", 404

    row_dict = dict(row)
    row_dict['risk_level'] = 'HIGH' if row_dict['risk_score'] >= 65 else 'MODERATE' if row_dict['risk_score'] >= 35 else 'LOW'
    row_dict['probabilities'] = json.loads(row_dict['probabilities_json']) if row_dict.get('probabilities_json') else {}
    row_dict['explanation_summary'] = f"Model classified {row_dict['sound_type']} sound event with primary pattern '{row_dict['screening_prediction']}'."

    filename, filepath = report_service.generate_pdf_report(row_dict)
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/api/report/csv', methods=['GET'])
def api_download_csv():
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT p.id, p.sound_type, p.confidence, p.screening_prediction, p.risk_score, p.created_at,
               a.filename, a.source
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        ORDER BY p.id DESC
    ''').fetchall()
    conn.close()

    records = []
    for r in rows:
        d = dict(r)
        d['risk_level'] = 'HIGH' if d['risk_score'] >= 65 else 'MODERATE' if d['risk_score'] >= 35 else 'LOW'
        d['top_pattern'] = d['screening_prediction']
        records.append(d)

    csv_content = report_service.generate_csv_report(records)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=pediatric_ai_history.csv"}
    )

@app.route('/recordings/<path:subpath>')
def serve_recordings(subpath):
    import mimetypes
    filepath = os.path.join(RECORDINGS_DIR, subpath)
    if os.path.exists(filepath):
        mime, _ = mimetypes.guess_type(filepath)
        return send_file(filepath, mimetype=mime or 'audio/wav')
    return jsonify({"error": "File not found"}), 404

# --- WEBSOCKET EVENT HANDLERS ---

@socketio.on('connect')
def handle_connect():
    live_sessions[request.sid] = {
        "age_group": "0-12m",
        "buffer": [],
        "history": [],
        "cry_count": 0,
        "cough_count": 0,
        "resp_count": 0
    }
    emit('status', {'message': 'Connected to Pediatric AI Monitor WS'})

@socketio.on('join_monitoring')
def handle_join(data):
    age_group = data.get('age_group', '0-12m')
    if request.sid in live_sessions:
        live_sessions[request.sid]['age_group'] = age_group
    emit('status', {'message': f'Monitoring active for age group {age_group}'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    session = live_sessions.get(request.sid)
    if not session:
        return

    try:
        # Decode PCM float32 audio chunk from base64
        audio_bytes = base64.b64decode(data.get('pcm_data', ''))
        pcm_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        session['buffer'].extend(pcm_array)

        # Process sliding window when buffer reaches ~3 seconds (48000 samples @ 16kHz)
        if len(session['buffer']) >= 48000:
            window_audio = np.array(session['buffer'][:48000], dtype=np.float32)
            session['buffer'] = session['buffer'][16000:] # 1s hop

            # Save 16-bit integer WAV
            filename, filepath = audio_service.save_live_recording(
                window_audio, sr=16000, prefix="live"
            )

            # Run pipeline
            result = pipeline_instance.process_audio(
                filepath,
                age_group=session['age_group'],
                temporal_history=session['history']
            )

            sed_type = result['detected_sound']
            if sed_type == "CRY":
                session['cry_count'] += 1
            elif sed_type == "COUGH":
                session['cough_count'] += 1
            elif sed_type == "BREATHING":
                session['resp_count'] += 1

            session['history'].append(result['risk_result'])

            # Emit real-time monitoring update
            emit('live_detection', {
                'detected_sound': sed_type,
                'confidence': result['sound_confidence'],
                'top_pattern': result['top_screening_pattern'],
                'risk_level': result['risk_result']['risk_level'],
                'risk_score': result['risk_result']['risk_score'],
                'cry_count': session['cry_count'],
                'cough_count': session['cough_count'],
                'resp_count': session['resp_count'],
                'recording_file': filename,
                'audio_url': f"/recordings/live/{filename}",
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })

    except Exception as e:
        print(f"Error handling WebSocket audio chunk: {e}")

if __name__ == '__main__':
    print("Starting Pediatric AI Health Screening Web Application on port 5050...")
    socketio.run(app, host='0.0.0.0', port=5050, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
