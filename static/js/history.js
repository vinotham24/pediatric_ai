document.addEventListener('DOMContentLoaded', () => {
    const tblBody = document.getElementById('tblHistoryBody');
    loadHistory();

    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            if (!data || data.length === 0) {
                tblBody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text-muted)">No past analysis history records found.</td></tr>';
                return;
            }
            tblBody.replaceChildren();
            data.forEach(item => {
                const tr = document.createElement('tr');
                const values = [item.created_at, item.source, item.filename, item.sound_classification || "Unavailable: sound not reliably determined.",
                    item.prediction_status, item.possible_category || 'Not available',
                    item.screening_prediction, 'Not calculated', item.legacy_result_note || ''];
                values.forEach(value => {
                    const td = document.createElement('td');
                    td.textContent = value == null ? '' : String(value);
                    tr.appendChild(td);
                });
                const actions = document.createElement('td');
                actions.innerHTML = `<a href="/analysis/${encodeURIComponent(item.id)}" class="btn btn-secondary btn-sm">Details</a> <a href="/api/report/${encodeURIComponent(item.id)}/pdf" class="btn btn-primary btn-sm">PDF</a>`;
                tr.appendChild(actions);
                tblBody.appendChild(tr);
            });
        } catch (err) {
            console.error('Error loading history:', err);
        }
    }
});
