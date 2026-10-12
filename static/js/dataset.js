document.addEventListener('DOMContentLoaded', () => {
    const tblBody = document.getElementById('tblDatasetBody');
    const evalCard = document.getElementById('datasetEvaluationCard');

    const dsFileName = document.getElementById('dsFileName');
    const dsSource = document.getElementById('dsSource');
    const dsTrueLabel = document.getElementById('dsTrueLabel');
    const dsAIPrediction = document.getElementById('dsAIPrediction');
    const dsAIConfidence = document.getElementById('dsAIConfidence');
    const dsAIRiskScore = document.getElementById('dsAIRiskScore');

    loadDatasets();

    async function loadDatasets() {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();

            if (!data || data.length === 0) {
                tblBody.innerHTML = `<tr><td colspan="8" style="text-align: center;">No dataset samples found.</td></tr>`;
                return;
            }

            tblBody.innerHTML = '';
            data.forEach((item, index) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${item.dataset}</strong></td>
                    <td><code>${item.file}</code></td>
                    <td>${item.age}</td>
                    <td><span style="color: #4ade80; font-weight: 700;">${item.pattern}</span></td>
                    <td id="aiPred_${index}">-</td>
                    <td id="aiConf_${index}">-</td>
                    <td id="aiRisk_${index}">-</td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="evaluateSample('${encodeURIComponent(item.filepath)}', '${item.file}', '${item.dataset}', '${item.pattern}', ${index})">
                            ▶ Run AI
                        </button>
                    </td>
                `;
                tblBody.appendChild(tr);
            });
        } catch (err) {
            console.error('Error loading dataset manifest:', err);
        }
    }

    window.evaluateSample = async function(filepath, filename, datasetName, trueLabel, index) {
        try {
            const res = await fetch('/api/datasets/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filepath: decodeURIComponent(filepath), age_group: '0-12m' })
            });

            if (!res.ok) throw new Error('Evaluation failed');
            const data = await res.json();

            document.getElementById(`aiPred_${index}`).innerHTML = `<strong style="color: #38bdf8;">${data.top_screening_pattern}</strong>`;
            document.getElementById(`aiConf_${index}`).textContent = `${(data.sound_confidence * 100).toFixed(1)}%`;
            document.getElementById(`aiRisk_${index}`).innerHTML = `<span class="badge badge-${data.risk_result.risk_level === 'HIGH' ? 'high' : data.risk_result.risk_level === 'MODERATE' ? 'mod' : 'low'}">${data.risk_result.risk_level} (${data.risk_result.risk_score}/100)</span>`;

            dsFileName.textContent = filename;
            dsSource.textContent = datasetName;
            dsTrueLabel.textContent = trueLabel;
            dsAIPrediction.textContent = data.top_screening_pattern;
            dsAIConfidence.textContent = `${(data.sound_confidence * 100).toFixed(1)}%`;
            dsAIRiskScore.textContent = `${data.risk_result.risk_score}/100 (${data.risk_result.risk_level})`;

            evalCard.style.display = 'block';

        } catch (err) {
            alert('Error running AI evaluation on sample.');
            console.error(err);
        }
    };
});
