document.addEventListener('DOMContentLoaded', () => {
    const tblBody = document.getElementById('tblHistoryBody');

    loadHistory();

    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();

            if (!data || data.length === 0) {
                tblBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No past analysis history records found.</td></tr>`;
                return;
            }

            tblBody.innerHTML = '';
            data.forEach(item => {
                const tr = document.createElement('tr');
                const badgeClass = item.risk_level === 'HIGH' ? 'high' : item.risk_level === 'MODERATE' ? 'mod' : 'low';
                
                tr.innerHTML = `
                    <td>${item.created_at}</td>
                    <td><span class="badge badge-secondary">${item.source}</span></td>
                    <td><code>${item.filename}</code></td>
                    <td><strong>${item.sound_type}</strong></td>
                    <td>${item.screening_prediction}</td>
                    <td>${(item.confidence * 100).toFixed(1)}%</td>
                    <td><span class="badge badge-${badgeClass}">${item.risk_level} (${item.risk_score}/100)</span></td>
                    <td>
                        <a href="/analysis/${item.id}" class="btn btn-secondary btn-sm">🔍 Details</a>
                        <a href="/api/report/${item.id}/pdf" class="btn btn-primary btn-sm">📄 PDF</a>
                    </td>
                `;
                tblBody.appendChild(tr);
            });

        } catch (err) {
            console.error('Error loading history:', err);
        }
    }
});
