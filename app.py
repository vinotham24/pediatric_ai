
import os
import sys
import json
import base64
import tempfile
import traceback
import numpy as np

from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, redirect, url_for
)
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
from services.prediction_status import normalize_prediction_record, prediction_metadata


# ============================================================
# FLASK APPLICATION INITIALIZATION
# ============================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pediatric-ai-secret-key-2026'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading'
)

audio_service = AudioService()
assistant_service = AIAssistantService()
report_service = ReportService()

# Original screening pipeline used by dataset and live monitoring.
pipeline_instance = InferencePipeline()

# Stores active WebSocket monitoring sessions.
live_sessions = {}

# ============================================================
# JSON CONVERSION HELPER
# ============================================================

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {
            str(k): sanitize_for_json(v)
            for k, v in obj.items()
        }

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


# ============================================================
# DISABLE BROWSER RESPONSE CACHING
# ============================================================

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = (
        'no-store, no-cache, must-revalidate, max-age=0'
    )
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


# ============================================================
# HTML FRONTEND ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template(
        'index.html',
        age_groups=AGE_GROUPS,
        disclaimer=MEDICAL_DISCLAIMER
    )


@app.route('/dashboard')
def dashboard():
    return render_template(
        'dashboard.html',
        age_groups=AGE_GROUPS,
        disclaimer=MEDICAL_DISCLAIMER
    )


@app.route('/live')
def live():
    return render_template(
        'live.html',
        age_groups=AGE_GROUPS,
        disclaimer=MEDICAL_DISCLAIMER
    )


@app.route('/upload')
def upload():
    return render_template(
        'upload.html',
        age_groups=AGE_GROUPS,
        disclaimer=MEDICAL_DISCLAIMER
    )


@app.route('/dataset')
def dataset():
    return redirect(url_for('dashboard'))


@app.route('/history')
def history():
    return render_template(
        'history.html',
        disclaimer=MEDICAL_DISCLAIMER
    )


# ------------------------------------------------------------
# NEW: SEPARATE CRY MODEL TEST PAGE
# ------------------------------------------------------------

@app.route('/cry-model-test')
def cry_model_test():
    return render_template('cry_model_test.html')


# ============================================================
# ANALYSIS DETAIL PAGE
# ============================================================

@app.route('/analysis/<int:record_id>')
def analysis_detail(record_id):

    conn = get_db_connection()

    pred = conn.execute('''
        SELECT p.*, a.filename, a.filepath, a.audio_type,
               a.duration, a.source, s.input_type, s.child_id,
               c.age_months
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        LEFT JOIN monitoring_sessions s ON a.session_id = s.id
        LEFT JOIN children c ON s.child_id = c.id
        WHERE p.id = ?
    ''', (record_id,)).fetchone()

    conn.close()

    if not pred:
        return redirect(url_for('history'))

    pred_dict = normalize_prediction_record(dict(pred))
    if pred_dict.get("source") in {"UPLOAD", "LIVE"} and pred_dict.get("filename"):
        pred_dict["audio_url"] = url_for(
            "serve_recordings", subpath=f"uploads/{pred_dict['filename']}"
        )

    inference_stub = {
        "detected_sound": pred_dict.get("sound_type"),
        "sound_classification": pred_dict.get("sound_classification"),
        "experimental_sound_candidate": pred_dict.get("experimental_sound_candidate"),
        "cry_classification_status": pred_dict.get("cry_classification_status"),
        "cough_classification_status": pred_dict.get("cough_classification_status"),
        "sound_confidence_is_calibrated": False,
        "prediction_status": pred_dict.get("prediction_status"),
        "possible_category": pred_dict.get("possible_category"),
        "top_screening_pattern": pred_dict.get("screening_prediction"),
        "risk_result": {
            "risk_available": False,
            "risk_level": "NOT CALCULATED",
            "risk_score": None,
            "risk_drivers": [],
        }
    }

    ai_explanation = assistant_service.explain_results(
        inference_stub
    )

    return render_template(
        'analysis.html',
        record=pred_dict,
        explanation=ai_explanation,
        disclaimer=MEDICAL_DISCLAIMER
    )


# ============================================================
# ORIGINAL AUDIO UPLOAD API
# DO NOT CONFUSE THIS WITH THE SEPARATE CRY MODEL API
# ============================================================

def _persist_analysis_result(result, filename, filepath, source):
    """Persist upload and live analyses through the same history schema."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO monitoring_sessions (input_type, risk_level, risk_score)
               VALUES (?, ?, ?)''',
            (source, result["risk_result"]["risk_level"], None),
        )
        session_id = cursor.lastrowid
        cursor.execute(
            '''INSERT INTO audio_records
               (session_id, filename, filepath, audio_type, duration, source)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (session_id, filename, filepath, result["detected_sound"], result["duration"], source),
        )
        audio_id = cursor.lastrowid
        metadata = json.dumps(prediction_metadata(result))
        # Legacy schema requires a non-null detector `confidence`; zero is a
        # storage sentinel only when no score exists. It is stripped from API,
        # history and reports by normalize_prediction_record.
        stored_score = result.get("sound_confidence")
        if stored_score is None:
            stored_score = 0.0
        cursor.execute(
            '''INSERT INTO predictions
               (audio_id, sound_type, confidence, screening_prediction,
                probabilities_json, risk_score)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (audio_id, result["detected_sound"], stored_score,
             result["top_screening_pattern"], metadata, None),
        )
        prediction_id = cursor.lastrowid
        conn.commit()
        return prediction_id
    except Exception:
        if hasattr(conn, "rollback"):
            conn.rollback()
        raise
    finally:
        conn.close()


def _process_audio_request(file_storage, source):
    filename, filepath = audio_service.save_uploaded_file(file_storage)
    try:
        result = InferencePipeline().process_audio(
            filepath,
            age_group=request.form.get("age_group", "0-12m"),
            audio_type_hint=request.form.get("audio_type_hint", "UNKNOWN"),
        )
        prediction_id = _persist_analysis_result(result, filename, filepath, source)
        result["ai_explanation"] = assistant_service.explain_results(
            result, age_group=request.form.get("age_group", "0-12m")
        )
        result["prediction_id"] = prediction_id
        result["audio_url"] = url_for("serve_recordings", subpath=f"uploads/{filename}")
        return sanitize_for_json(result)
    except Exception:
        audio_service.delete_recording(filepath)
        raise


def _request_audio_file():
    for key in ("audio_file", "audio", "file"):
        if key in request.files:
            candidate = request.files[key]
            if candidate and candidate.filename:
                return candidate
    return None

@app.route('/api/audio/upload', methods=['POST'])
def api_upload_audio():
    file = _request_audio_file()
    if not file or file.filename == '':
        return jsonify({
            "error": "No valid audio file provided",
            "prediction_status": "ERROR"
        }), 400

    try:
        return jsonify(_process_audio_request(file, "UPLOAD"))
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": f"Unsupported or invalid audio: {exc}", "prediction_status": "ERROR"}), 400
    except Exception:
        print(f"ERROR IN UPLOAD API: {traceback.format_exc()}", flush=True)
        return jsonify({
            "error": "Audio analysis could not be completed.",
            "prediction_status": "ERROR"
        }), 500


# ============================================================
# ORIGINAL LIVE AUDIO ANALYSIS API
# ============================================================

@app.route('/api/audio/live_analyze', methods=['POST'])
def api_live_analyze_audio():
    file = _request_audio_file()
    if not file or file.filename == '':
        return jsonify({
            "error": "No valid audio file provided",
            "prediction_status": "ERROR"
        }), 400

    try:
        return jsonify(_process_audio_request(file, "LIVE"))
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": f"Unsupported or invalid audio: {exc}", "prediction_status": "ERROR"}), 400
    except Exception:
        print(f"ERROR IN LIVE AUDIO API: {traceback.format_exc()}", flush=True)
        return jsonify({
            "error": "Live audio analysis could not be completed.",
            "prediction_status": "ERROR"
        }), 500


# ============================================================
# EXPERIMENTAL CRY ANALYSIS API
# ============================================================

@app.route('/api/cry-model/test', methods=['POST'])
def api_test_cry_model():

    audio_file = request.files.get('audio_file')

    if not audio_file or audio_file.filename == '':
        return jsonify({
            "success": False,
            "error": "Please select an audio file."
        }), 400

    source_path = wav_path = None

    try:
        source_path, wav_path = audio_service.decode_cry_upload(audio_file)
        result = InferencePipeline().process_audio(wav_path)
        cry_result = result.get("cry_analysis") or {}
        candidate_class = cry_result.get("candidate_class")
        status = result["prediction_status"]

        return jsonify({
            "success": True,
            "audio_analysis_status": result["audio_analysis_status"],
            "detected_sound": result["detected_sound"],
            "sound_classification": result["sound_classification"],
            "experimental_sound_candidate": result["experimental_sound_candidate"],
            "sound_detection_status": result["sound_detection_status"],
            "sound_detection_method": result["sound_detection_method"],
            "prediction_status": status,
            "predicted_label": candidate_class if status == "CONFIDENT" else status.lower(),
            "top_label": candidate_class,
            "possible_category": result["possible_category"],
            "model_score": result["model_score"],
            "model_score_interpretation": result["model_score_interpretation"],
            "cry_classification_status": result["cry_classification_status"],
            "cough_classification_status": result["cough_classification_status"],
            "medical_diagnosis": result["medical_diagnosis"],
            "warning": result["top_screening_pattern"],
            "class_scores": {},
            "disclaimer": (
                "Experimental research output only. This model is not clinically "
                "validated and does not provide a medical diagnosis."
            ),
        })

    except ValueError as exc:
        print(f"Invalid cry audio upload: {traceback.format_exc()}", flush=True)
        return jsonify({
            "success": False,
            "error": str(exc),
            "prediction_status": "ERROR",
        }), 400
    except Exception:
        print(f"ERROR IN CRY MODEL API: {traceback.format_exc()}", flush=True)
        return jsonify({
            "success": False,
            "error": "Audio analysis could not be completed.",
            "prediction_status": "ERROR",
        }), 500

    finally:
        for temporary_path in (source_path, wav_path):
            if temporary_path:
                try:
                    audio_service.delete_recording(temporary_path)
                except OSError:
                    print(f"Could not remove temporary audio file: {temporary_path}", flush=True)


# ============================================================
# DATASET API
# ============================================================

@app.route('/api/datasets', methods=['GET'])
def api_get_datasets():

    manifest_path = os.path.join(
        DATASETS_DIR,
        "dataset_manifest.json"
    )

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
        return jsonify({
            "error": "Dataset sample file not found"
        }), 404

    result = pipeline_instance.process_audio(
        filepath,
        age_group=age_group
    )

    filename = os.path.basename(filepath)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        '''
        INSERT INTO monitoring_sessions
        (input_type, risk_level, risk_score)
        VALUES (?, ?, ?)
        ''',
        (
            'DATASET',
            result['risk_result']['risk_level'],
                None
        )
    )

    session_id = cursor.lastrowid

    cursor.execute('''
        INSERT INTO audio_records
        (session_id, filename, filepath, audio_type, duration, source)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        session_id,
        filename,
        filepath,
        result['detected_sound'],
        result['duration'],
        'DATASET'
    ))

    audio_id = cursor.lastrowid

    probs_json = json.dumps(prediction_metadata(result))

    cursor.execute('''
        INSERT INTO predictions
        (audio_id, sound_type, confidence, screening_prediction,
         probabilities_json, risk_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        audio_id,
        result['detected_sound'],
        result['sound_confidence'] if result.get('sound_confidence') is not None else 0.0,
        result['top_screening_pattern'],
        probs_json,
        None
    ))

    pred_id = cursor.lastrowid

    conn.commit()
    conn.close()

    result['prediction_id'] = pred_id

    return jsonify(result)


# ============================================================
# HISTORY API
# ============================================================

@app.route('/api/history', methods=['GET'])
def api_get_history():

    conn = get_db_connection()

    rows = conn.execute('''
        SELECT p.id, p.sound_type, p.confidence,
               p.screening_prediction, p.risk_score, p.probabilities_json, p.created_at,
               a.filename, a.duration, a.source
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        ORDER BY p.id DESC
        LIMIT 50
    ''').fetchall()

    conn.close()

    history_list = []

    for r in rows:
        history_list.append(normalize_prediction_record(dict(r)))

    return jsonify(history_list)


# ============================================================
# PDF REPORT API
# ============================================================

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

    row_dict = normalize_prediction_record(dict(row))

    row_dict['explanation_summary'] = row_dict['experimental_notice']

    filename, filepath = report_service.generate_pdf_report(
        row_dict
    )

    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename
    )


# ============================================================
# CSV REPORT API
# ============================================================

@app.route('/api/report/csv', methods=['GET'])
def api_download_csv():

    conn = get_db_connection()

    rows = conn.execute('''
        SELECT p.id, p.sound_type, p.confidence,
               p.screening_prediction, p.risk_score, p.probabilities_json, p.created_at,
               a.filename, a.source
        FROM predictions p
        JOIN audio_records a ON p.audio_id = a.id
        ORDER BY p.id DESC
    ''').fetchall()

    conn.close()

    records = []

    for r in rows:
        records.append(normalize_prediction_record(dict(r)))

    csv_content = report_service.generate_csv_report(records)

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-disposition":
            "attachment; filename=pediatric_ai_history.csv"
        }
    )


# ============================================================
# SERVE RECORDINGS
# ============================================================

@app.route('/recordings/<path:subpath>')
def serve_recordings(subpath):
    import mimetypes
    recording_root = os.path.realpath(RECORDINGS_DIR)
    filepath = os.path.realpath(os.path.join(recording_root, subpath))
    if os.path.commonpath([recording_root, filepath]) == recording_root and os.path.isfile(filepath):
        mime, _ = mimetypes.guess_type(filepath)

        return send_file(
            filepath,
            mimetype=mime or 'audio/wav'
        )

    return jsonify({
        "error": "File not found"
    }), 404


# ============================================================
# WEBSOCKET EVENT HANDLERS
# ============================================================

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

    emit(
        'status',
        {'message': 'Connected to Pediatric AI Monitor WS'}
    )


@socketio.on('join_monitoring')
def handle_join(data):

    age_group = data.get('age_group', '0-12m')

    if request.sid in live_sessions:
        live_sessions[request.sid]['age_group'] = age_group

    emit(
        'status',
        {'message': f'Monitoring active for age group {age_group}'}
    )


@socketio.on('audio_chunk')
def handle_audio_chunk(data):

    session = live_sessions.get(request.sid)

    if not session:
        return

    try:
        # Decode PCM float32 audio chunk from base64.
        audio_bytes = base64.b64decode(
            data.get('pcm_data', '')
        )

        pcm_array = np.frombuffer(
            audio_bytes,
            dtype=np.float32
        )

        session['buffer'].extend(pcm_array)

        # Process sliding window when buffer reaches 3 seconds.
        if len(session['buffer']) >= 48000:

            window_audio = np.array(
                session['buffer'][:48000],
                dtype=np.float32
            )

            session['buffer'] = session['buffer'][16000:]

            # Save 16-bit integer WAV.
            filename, filepath = audio_service.save_live_recording(
                window_audio,
                sr=16000,
                prefix="live"
            )

            # Run original screening pipeline.
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

            session['history'].append(
                result['risk_result']
            )

            # Emit real-time monitoring update.
            emit('live_detection', {
                'detected_sound': sed_type,
                'sound_classification': result.get('sound_classification'),
                'experimental_sound_candidate': result.get('experimental_sound_candidate'),
                'prediction_status': result['prediction_status'],
                'possible_category': result['possible_category'],
                'cry_classification_status': result.get('cry_classification_status'),
                'cough_classification_status': result.get('cough_classification_status'),
                'model_score': result['model_score'],
                'model_score_interpretation': result['model_score_interpretation'],
                'top_pattern': result['top_screening_pattern'],
                'cry_count': session['cry_count'],
                'cough_count': session['cough_count'],
                'resp_count': session['resp_count'],
                'recording_file': filename,
                'audio_url': f"/recordings/live/{filename}",
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })

    except Exception as e:
        print(
            f"Error handling WebSocket audio chunk: {e}",
            flush=True
        )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == '__main__':

    print(
        "Starting Pediatric AI Health Screening "
        "Web Application on port 5051..."
    )

    socketio.run(
        app,
        host='0.0.0.0',
        port=5051,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )
