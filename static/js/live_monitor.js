document.addEventListener('DOMContentLoaded', () => {
    let mediaRecorder = null;
    let audioChunks = [];
    let audioContext = null;
    let analyser = null;
    let animFrameId = null;
    let timerInterval = null;
    let secondsElapsed = 0;

    const btnStart = document.getElementById('btnStartRecord');
    const btnStop = document.getElementById('btnStopRecord');
    const ageSelect = document.getElementById('liveAgeSelect');
    const statusText = document.getElementById('liveStatusText');
    const timerText = document.getElementById('liveTimerText');

    const resultContainer = document.getElementById('liveResultContainer');
    const liveAudioPlayer = document.getElementById('liveAudioPlayer');
    const canvas = document.getElementById('liveFullWaveform');
    const canvasCtx = canvas ? canvas.getContext('2d') : null;
    const micSecurityBanner = document.getElementById('micSecurityBanner');
    const micSecurityText = document.getElementById('micSecurityText');

    // Check Chrome Secure Context / MediaDevices availability on load
    checkMicAvailability();

    function checkMicAvailability() {
        const isNotSecureContext = !window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
        const noMediaDevices = !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia;

        if (isNotSecureContext || noMediaDevices) {
            if (micSecurityBanner) {
                micSecurityBanner.style.display = 'block';
                if (isNotSecureContext && micSecurityText) {
                    micSecurityText.innerHTML = `Chrome blocks microphone access on unencrypted IP links. Please open <a href="http://127.0.0.1:5051/live" style="font-weight: 700; color: #b45309; text-decoration: underline;">http://127.0.0.1:5051/live</a> or <a href="http://localhost:5051/live" style="font-weight: 700; color: #b45309; text-decoration: underline;">http://localhost:5051/live</a> in Chrome to enable microphone access.`;
                }
            }
        }
    }

    if (btnStart) {
        btnStart.addEventListener('click', startRecording);
    }
    if (btnStop) {
        btnStop.addEventListener('click', stopAndAnalyze);
    }

    async function startRecording() {
        try {
            audioChunks = [];
            
            // Check if navigator.mediaDevices exists
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                if (micSecurityBanner) micSecurityBanner.style.display = 'block';
                const isNotLocal = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
                let errMsg = 'Microphone API is disabled by Chrome on this connection.\n\n';
                if (isNotLocal && window.location.protocol === 'http:') {
                    errMsg += 'Chrome strictly requires accessing via http://127.0.0.1:5051/live or http://localhost:5051/live (or HTTPS) to allow microphone recording.\n\nPlease switch to http://127.0.0.1:5051/live in your browser address bar.';
                } else {
                    errMsg += 'Please ensure microphone permissions are granted in Chrome settings (click the lock/icon next to the URL address bar).';
                }
                alert(errMsg);
                return;
            }

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

            // Hide warning banner if mic access granted
            if (micSecurityBanner) micSecurityBanner.style.display = 'none';

            // Initialize AudioContext & Analyser for Waveform Visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 1024;
            source.connect(analyser);

            // Setup MediaRecorder with best supported browser MIME type
            let options = {};
            if (window.MediaRecorder) {
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    options = { mimeType: 'audio/webm;codecs=opus' };
                } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                    options = { mimeType: 'audio/webm' };
                } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                    options = { mimeType: 'audio/ogg;codecs=opus' };
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options = { mimeType: 'audio/mp4' };
                }
            }
            mediaRecorder = new MediaRecorder(stream, options);
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };
            mediaRecorder.start(250);

            // UI State updates
            btnStart.style.display = 'none';
            btnStop.style.display = 'inline-flex';
            if (statusText) {
                statusText.textContent = '🔴 RECORDING LIVE MICROPHONE...';
                statusText.style.color = '#dc2626';
            }

            secondsElapsed = 0;
            if (timerText) timerText.textContent = '00:00';
            timerInterval = setInterval(() => {
                secondsElapsed++;
                const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
                const secs = String(secondsElapsed % 60).padStart(2, '0');
                if (timerText) timerText.textContent = `${mins}:${secs}`;
            }, 1000);

            drawWaveform();
        } catch (err) {
            if (micSecurityBanner) micSecurityBanner.style.display = 'block';
            console.error('Microphone Access Error:', err);
            
            let userHelpMsg = 'Could not access your microphone in Chrome.\n\n';
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                userHelpMsg += 'Permission Denied: Please click the lock icon 🔒 (or site settings) next to the Chrome address bar and set Microphone to "Allow".';
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                userHelpMsg += 'No microphone found: Please ensure a microphone is connected to your system.';
            } else {
                userHelpMsg += `Error: ${err.message || err.name}\n\nMake sure to open http://127.0.0.1:5051/live in Chrome.`;
            }
            alert(userHelpMsg);
        }
    }

    function drawWaveform() {
        if (!canvasCtx || !analyser) return;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function render() {
            animFrameId = requestAnimationFrame(render);
            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = '#0f172a';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = '#38bdf8';
            canvasCtx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();
        }
        render();
    }

    async function stopAndAnalyze() {
        if (!mediaRecorder) return;

        btnStop.disabled = true;
        btnStop.textContent = '⏳ Analyzing Live Audio...';

        if (timerInterval) clearInterval(timerInterval);
        if (animFrameId) cancelAnimationFrame(animFrameId);

        mediaRecorder.onstop = async () => {
            // Stop mic track
            if (mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }

            const mimeType = mediaRecorder.mimeType || 'audio/webm';
            let filename = 'live_recording.webm';
            if (mimeType.includes('ogg')) filename = 'live_recording.ogg';
            else if (mimeType.includes('mp4') || mimeType.includes('aac')) filename = 'live_recording.mp4';
            else if (mimeType.includes('wav')) filename = 'live_recording.wav';

            const audioBlob = new Blob(audioChunks, { type: mimeType });
            
            // Set playback URL
            if (liveAudioPlayer) {
                liveAudioPlayer.src = URL.createObjectURL(audioBlob);
            }

            // Send blob to live analysis endpoint
            const formData = new FormData();
            formData.append('audio_file', audioBlob, filename);
            formData.append('audio', audioBlob, filename);
            formData.append('age_group', ageSelect ? ageSelect.value : '0-12m');

            try {
                const res = await fetch('/api/audio/live_analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    const errObj = await res.json().catch(() => ({}));
                    throw new Error(errObj.error || `Live analysis failed with HTTP ${res.status}`);
                }
                const data = await res.json();

                window.renderPredictionResult(data);
                if (liveAudioPlayer && data.audio_url) {
                    const previousUrl = liveAudioPlayer.src;
                    liveAudioPlayer.src = data.audio_url;
                    if (previousUrl && previousUrl.startsWith('blob:')) URL.revokeObjectURL(previousUrl);
                }
                const detailLink = document.getElementById('liveViewFullAnalysis');
                const pdfLink = document.getElementById('liveDownloadPDF');
                if (detailLink && data.prediction_id) detailLink.href = `/analysis/${data.prediction_id}`;
                if (pdfLink && data.prediction_id) pdfLink.href = `/api/report/${data.prediction_id}/pdf`;
                if (resultContainer) {
                    resultContainer.style.display = 'block';
                    resultContainer.scrollIntoView({ behavior: 'smooth' });
                }
            } catch (err) {
                window.renderPredictionResult({ prediction_status: 'ERROR' }, err.message);
                if (resultContainer) resultContainer.style.display = 'block';
                console.error(err);
            } finally {
                btnStart.style.display = 'inline-flex';
                btnStop.style.display = 'none';
                btnStop.disabled = false;
                btnStop.textContent = '⏹ STOP & ANALYZE DIRECTLY';
                if (statusText) {
                    statusText.textContent = '● MICROPHONE IDLE';
                    statusText.style.color = '#64748b';
                }
            }
        };

        mediaRecorder.stop();
    }
});
