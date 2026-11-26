document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    if (!form) return;
    const result = document.getElementById('uploadResultContainer');
    const button = document.getElementById('btnUploadSubmit');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        button.disabled = true;
        button.textContent = 'Analyzing…';
        try {
            const response = await fetch('/api/audio/upload', { method: 'POST', body: new FormData(form) });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Audio analysis failed.');
            window.renderPredictionResult(data);
            const player = document.getElementById('uploadAudioPlayer');
            if (player && data.audio_url) player.src = data.audio_url;
            document.getElementById('btnViewFullAnalysis').href = `/analysis/${data.prediction_id}`;
            document.getElementById('btnDownloadPDF').href = `/api/report/${data.prediction_id}/pdf`;
            result.style.display = 'block';
            result.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            window.renderPredictionResult({ prediction_status: 'ERROR' }, error.message);
            result.style.display = 'block';
        } finally {
            button.disabled = false;
            button.textContent = 'Analyze audio';
        }
    });
});
