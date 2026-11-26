document.addEventListener('DOMContentLoaded', () => {
    const tblBody = document.getElementById('tblDatasetBody');
    const evalCard = document.getElementById('datasetEvaluationCard');
    const fields = {
        file: document.getElementById('dsFileName'), source: document.getElementById('dsSource'),
        truth: document.getElementById('dsTrueLabel'), prediction: document.getElementById('dsAIPrediction'),
        score: document.getElementById('dsAIConfidence'), status: document.getElementById('dsAIStatus'),
        risk: document.getElementById('dsAIRiskScore')
    };
    loadDatasets();

    async function loadDatasets() {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();
            if (!Array.isArray(data) || !data.length) {
                tblBody.innerHTML = '<tr><td colspan="8" style="text-align:center">No dataset samples found.</td></tr>';
                return;
            }
            tblBody.replaceChildren();
            data.forEach((item, index) => {
                const tr = document.createElement('tr');
                [item.dataset, item.file, item.age, item.pattern, '-', '-', 'Not calculated'].forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value == null ? '' : String(value);
                    tr.appendChild(td);
                });
                const action = document.createElement('td');
                const button = document.createElement('button');
                button.className = 'btn btn-secondary btn-sm';
                button.textContent = 'Run experimental analysis';
                button.addEventListener('click', () => evaluateSample(item, index));
                action.appendChild(button);
                tr.appendChild(action);
                tr.dataset.index = index;
                tblBody.appendChild(tr);
            });
        } catch (err) { console.error('Error loading dataset manifest:', err); }
    }

    async function evaluateSample(item, index) {
        try {
            const res = await fetch('/api/datasets/analyze', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filepath: item.filepath, age_group: '0-12m'})
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Evaluation failed');
            const score = Number(data.model_score);
            const scoreText = Number.isFinite(score)
                ? `${(score * 100).toFixed(2)}% raw model score; not a probability of correctness.`
                : 'Unavailable';
            const pattern = data.top_screening_pattern || 'Not available.';
            const row = tblBody.querySelector(`tr[data-index="${index}"]`);
            const cells = row ? row.cells : null;
            if (cells) {
                cells[4].textContent = `${data.prediction_status || 'UNKNOWN'}: ${data.possible_category || 'No candidate'}`;
                cells[5].textContent = scoreText;
                cells[6].textContent = 'Not calculated';
            }
            fields.file.textContent = item.file;
            fields.source.textContent = item.dataset;
            fields.truth.textContent = item.pattern;
            fields.status.textContent = data.prediction_status || 'UNKNOWN';
            fields.prediction.textContent = pattern;
            fields.score.textContent = `${scoreText} ${data.model_score_interpretation || ''}`;
            fields.risk.textContent = 'Not calculated';
            evalCard.style.display = 'block';
        } catch (err) {
            console.error(err);
            alert('Unable to analyze this dataset sample.');
        }
    }
});
