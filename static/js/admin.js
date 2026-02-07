document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.content-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.dataset.target + '-section';

            // Update active state
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Show section
            sections.forEach(s => s.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');

            // Update Breadcrumb
            document.getElementById('current-page').innerText = item.textContent.trim();
        });
    });

    // Mobile Menu Toggle
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.getElementById('sidebar');

    if (mobileBtn && sidebar) {
        mobileBtn.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !mobileBtn.contains(e.target) && sidebar.classList.contains('active')) {
                    sidebar.classList.remove('active');
                }
            }
        });
    }

    // Initial Load
    fetchStats();
    fetchHistory();

    // Stats Polling (every 5 seconds)
    setInterval(fetchStats, 5000);

    // File Upload Handler
    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }

    // Modal Elements
    const scanModal = document.getElementById('scan-modal');
    const openScanBtn = document.getElementById('open-scan-modal');
    const closeScanBtn = document.querySelectorAll('.close-modal');

    // Modal Toggle
    if (openScanBtn) {
        openScanBtn.addEventListener('click', () => {
            scanModal.classList.add('active');
        });
    }

    closeScanBtn.forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        });
    });

    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;

            // Toggle active buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle active content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`tab-${target}`).classList.add('active');
        });
    });

    // Quick Scan File Input
    const modalFileInput = document.getElementById('modal-file-upload');
    if (modalFileInput) {
        modalFileInput.addEventListener('change', (e) => {
            const fileName = e.target.files[0] ? e.target.files[0].name : "No file chosen";
            document.getElementById('file-name').innerText = fileName;
            document.getElementById('modal-upload-btn').disabled = !e.target.files[0];
        });

        document.getElementById('modal-upload-btn').addEventListener('click', () => {
            const file = modalFileInput.files[0];
            if (file) handleScanUpload(file);
        });
    }

    // Template Upload Logic
    const templateUploadZone = document.getElementById('template-upload-zone');
    const templateInput = document.getElementById('template-files');
    const saveTemplateBtn = document.getElementById('save-template-btn');
    let templateFiles = [];

    if (templateUploadZone) {
        templateUploadZone.addEventListener('click', () => templateInput.click());

        templateInput.addEventListener('change', (e) => {
            handleTemplateFiles(e.target.files);
        });

        // Drag & Drop
        templateUploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            templateUploadZone.style.backgroundColor = '#eef7fc';
        });

        templateUploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            templateUploadZone.style.backgroundColor = '#fafafa';
        });

        templateUploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            templateUploadZone.style.backgroundColor = '#fafafa';
            handleTemplateFiles(e.dataTransfer.files);
        });

        saveTemplateBtn.addEventListener('click', uploadTemplates);
    }

    function handleTemplateFiles(files) {
        const previewList = document.getElementById('template-preview-list');

        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;
            templateFiles.push(file);

            const reader = new FileReader();
            reader.onload = (e) => {
                const div = document.createElement('div');
                div.className = 'preview-item';
                div.innerHTML = `<img src="${e.target.result}">`;
                previewList.appendChild(div);
            };
            reader.readAsDataURL(file);
        });

        if (templateFiles.length >= 3) {
            saveTemplateBtn.classList.remove('hidden');
        }
    }

    async function uploadTemplates() {
        if (templateFiles.length < 3) {
            alert("Please upload at least 3 images.");
            return;
        }

        const formData = new FormData();
        templateFiles.forEach(file => {
            formData.append('files', file);
        });

        saveTemplateBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';

        try {
            // We need to update backend to handle multiple files
            // For now sending one-by-one or backend needs update
            let successCount = 0;
            for (const file of templateFiles) {
                const fd = new FormData();
                fd.append('file', file);
                const res = await fetch('/api/add_reference', {
                    method: 'POST',
                    body: fd
                });
                if (res.ok) successCount++;
            }

            alert(`Successfully added ${successCount} reference images!`);
            location.reload();

        } catch (e) {
            console.error(e);
            alert("Error uploading templates");
        } finally {
            saveTemplateBtn.innerHTML = '<i class="fa-solid fa-save"></i> Save Reference Model';
        }
    }
});

async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        document.getElementById('total-scans').innerText = data.total_scans;
        document.getElementById('perfect-count').innerText = data.perfect_count;
        document.getElementById('defected-count').innerText = data.defected_count;

        // Update list
        updateRecentList(data.last_24h);
    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}

function updateRecentList(items) {
    const list = document.getElementById('recent-history-list');
    list.innerHTML = '';

    if (!items || items.length === 0) {
        list.innerHTML = '<div class="empty-state">No recent scans</div>';
        return;
    }

    // Show last 5
    items.slice().reverse().slice(0, 5).forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item-row';
        const date = new Date(item.timestamp * 1000).toLocaleTimeString();
        div.innerHTML = `
            <span>${date}</span>
            <span class="status-tag ${item.status}">${item.status}</span>
        `;
        list.appendChild(div);
    });
}

async function fetchHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        updateHistoryTable(data);
    } catch (e) {
        console.error("Failed to fetch history", e);
    }
}

function updateHistoryTable(data) {
    const tbody = document.getElementById('history-table-body');
    tbody.innerHTML = '';

    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${row.id}</td>
            <td>${new Date(row.timestamp).toLocaleString()}</td>
            <td><span class="status-tag ${row.status}">${row.status}</span></td>
            <td>${row.defects}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Helper to update status display
function updateStatusDisplay(element, result) {
    if (result.block_status === 'PERFECT') {
        element.innerHTML = '<i class="fa-solid fa-check-circle"></i> Bearing Present';
        element.className = 'detection-status perfect';
        element.style.backgroundColor = '#dff9e8'; // Green light background
        element.style.color = '#2ecc71';
    } else {
        element.innerHTML = '<i class="fa-solid fa-times-circle"></i> Bearing Missed';
        element.className = 'detection-status defective';
        element.style.backgroundColor = '#fadbd8'; // Red light background
        element.style.color = '#e74c3c';
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const statusDiv = document.getElementById('detection-result');
    statusDiv.innerText = "Analyzing...";
    statusDiv.className = "detection-status waiting";

    try {
        const response = await fetch('/api/inspect_upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // Update Live Preview with analyzed result
        const liveFeed = document.getElementById('overview-live-feed');
        if (data.image && liveFeed) {
            liveFeed.src = data.image; // Temporarily show result in live feed area

            // Reset after 5 seconds to video feed
            setTimeout(() => {
                liveFeed.src = "/api/video_feed";
            }, 5000);
        }

        // Update Status
        const result = data.result;
        updateStatusDisplay(statusDiv, result);

        // Update Large Result Display if visible
        const largeResult = document.getElementById('large-result-display');
        if (largeResult) {
            updateStatusDisplay(largeResult, result);
        }

        // Refresh stats
        fetchStats();
        fetchHistory();

    } catch (e) {
        console.error("Upload failed", e);
        statusDiv.innerText = "Error";
    }
}

async function handleScanUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    // UI Feedback
    const btn = document.getElementById('modal-upload-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = 'Analyzing...';
    btn.disabled = true;

    const resultBox = document.getElementById('scan-result');
    resultBox.classList.remove('hidden');
    resultBox.innerHTML = '<div class="detection-status waiting">Processing...</div>';

    try {
        const response = await fetch('/api/inspect_upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        const result = data.result;

        // Update Modal Result
        updateStatusDisplay(resultBox, result);

        // Refresh
        fetchStats();
        fetchHistory();

    } catch (e) {
        console.error(e);
        resultBox.innerText = "Scan failed";
        resultBox.className = 'detection-status defective';
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function captureAndScan() {
    alert("Triggering live scan...");
    // Future: Trigger backend live scan logic
}

function exportHistory() {
    // Simple CSV export
    const rows = [];
    rows.push(["ID", "Time", "Status", "Defects"]);

    // Fetch latest history from table or store
    const tableRows = document.querySelectorAll('#history-table-body tr');
    tableRows.forEach(tr => {
        const cols = tr.querySelectorAll('td');
        rows.push([
            cols[0].innerText,
            cols[1].innerText,
            cols[2].innerText,
            cols[3].innerText
        ]);
    });

    let csvContent = "data:text/csv;charset=utf-8," +
        rows.map(e => e.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "inspection_history.csv");
    document.body.appendChild(link);
    link.click();
}

async function stopInspection() {
    try {
        const response = await fetch('/api/camera/stop', { method: 'POST' });
        const data = await response.json();
        alert(data.message);
        // Update UI to show stopped state if needed
        const liveFeed = document.getElementById('sidebar-live-feed');
        if (liveFeed) {
            // liveFeed.src = ""; // Option to clear feed
        }
    } catch (e) {
        console.error("Error stopping inspection:", e);
        alert("Failed to stop inspection");
    }
}

async function selectCamera(type) {
    let index = 0;
    if (type === 'ip') index = 1;
    // 'usb' defaults to 0

    try {
        const response = await fetch('/api/camera/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ index: index })
        });
        const data = await response.json();
        alert(data.message);
        // Refresh feed?
        const liveFeed = document.getElementById('sidebar-live-feed');
        if (liveFeed) {
            liveFeed.src = liveFeed.src.split('?')[0] + '?t=' + new Date().getTime();
        }
        const mainFeed = document.querySelector('.main-feed img');
        if (mainFeed) {
            mainFeed.src = mainFeed.src.split('?')[0] + '?t=' + new Date().getTime();
        }
    } catch (e) {
        console.error("Error selecting camera:", e);
        alert("Failed to switch camera");
    }
}
