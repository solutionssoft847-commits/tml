const CONFIG = {
    WS_RECONNECT_INTERVAL: 3000,  // ms
    WS_MAX_RETRIES: 10,
    API_TIMEOUT: 30000,  // 30s for inspection
    STATS_REFRESH_INTERVAL: 5000,  // 5s
    HISTORY_REFRESH_INTERVAL: 10000,  // 10s
    DEBOUNCE_DELAY: 300  // ms
};

// ==================== STATE MANAGEMENT ====================
const state = {
    ws: null,
    wsReconnectAttempts: 0,
    wsReconnectTimer: null,
    currentSection: 'overview',
    isProcessing: false,
    lastScanResult: null,
    templateFiles: []
};

// ==================== WEBSOCKET MANAGEMENT ====================
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(' Connecting to WebSocket:', wsUrl);

    try {
        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            console.log('✓ WebSocket connected');
            state.wsReconnectAttempts = 0;

            // Send ping to keep alive
            const pingInterval = setInterval(() => {
                if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                    state.ws.send(JSON.stringify({ type: 'ping' }));
                } else {
                    clearInterval(pingInterval);
                }
            }, 30000);
        };

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error(' WebSocket message parse error:', e);
            }
        };

        state.ws.onerror = (error) => {
            console.error(' WebSocket error:', error);
        };

        state.ws.onclose = (event) => {
            console.log(' WebSocket closed:', event.code, event.reason);
            scheduleReconnect();
        };

    } catch (e) {
        console.error(' WebSocket init error:', e);
        scheduleReconnect();
    }
}

function scheduleReconnect() {
    if (state.wsReconnectAttempts >= CONFIG.WS_MAX_RETRIES) {
        console.error(' Max WebSocket reconnect attempts reached');
        showNotification('WebSocket connection failed. Please refresh the page.', 'error');
        return;
    }

    state.wsReconnectAttempts++;
    console.log(`Reconnecting WebSocket in ${CONFIG.WS_RECONNECT_INTERVAL}ms (attempt ${state.wsReconnectAttempts}/${CONFIG.WS_MAX_RETRIES})`);

    if (state.wsReconnectTimer) {
        clearTimeout(state.wsReconnectTimer);
    }

    state.wsReconnectTimer = setTimeout(() => {
        initWebSocket();
    }, CONFIG.WS_RECONNECT_INTERVAL);
}

function handleWebSocketMessage(data) {
    console.log('WebSocket message:', data.type);

    if (data.type === 'stats_update') {
        updateStats(data.data);
    } else if (data.type === 'history_update') {
        if (data.data && data.data.length > 0) {
            prependHistoryItems(data.data);
        }
    } else if (data.type === 'camera_status') {
        state.activeCameraId = data.data.active ? data.data.id : null;
        console.log("Camera status synced:", state.activeCameraId);
        // If we have any UI update for camera card active class, trigger it here:
        if (state.currentSection === 'camera') loadCameras();
    } else if (data.type === 'pong') {
        // Keep-alive response
    } else {
        console.warn(' Unknown message type:', data.type);
    }
}

// ==================== API HELPERS ====================
async function apiCall(url, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || CONFIG.API_TIMEOUT);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        clearTimeout(timeout);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (e) {
        clearTimeout(timeout);
        if (e.name === 'AbortError') {
            throw new Error('Request timeout');
        }
        throw e;
    }
}

// Debounce helper
function debounce(func, delay) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
}

// ==================== UI UPDATES ====================
function updateStats(data) {
    const totalScans = document.getElementById('total-scans');
    const perfectCount = document.getElementById('perfect-count');
    const defectedCount = document.getElementById('defected-count');

    if (totalScans) animateNumber(totalScans, data.total_scans || 0);
    if (perfectCount) animateNumber(perfectCount, data.perfect_count || 0);
    if (defectedCount) animateNumber(defectedCount, data.defected_count || 0);

    // Update recent list if provided
    if (data.last_24h) {
        updateRecentList(data.last_24h);
    }
}

function getDisplayStatus(status) {
    if (!status) return 'N/A';
    const s = status.toUpperCase();
    if (s === 'PERFECT' || s === 'SUCCESS' || s === 'ERROR') return 'Pass';
    if (s === 'DEFECTIVE' || s === 'FAIL') return 'Fail';
    return status;
}

function updateRecentList(items) {
    const list = document.getElementById('recent-history-list');
    if (!list) return;

    list.innerHTML = '';
    if (!items || items.length === 0) {
        list.innerHTML = '<div class="empty-state">No recent scans</div>';
        return;
    }

    items.slice().reverse().slice(0, 5).forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item-row';
        const date = item.timestamp ? new Date(item.timestamp * 1000).toLocaleTimeString() : 'N/A';
        const displayStatus = getDisplayStatus(item.status);
        div.innerHTML = `
            <span>${date}</span>
            <span class="status-tag ${displayStatus.toLowerCase()}">${displayStatus}</span>
        `;
        list.appendChild(div);
    });
}

function animateNumber(element, targetValue) {
    const currentValue = parseInt(element.textContent) || 0;
    if (currentValue === targetValue) return;

    const duration = 500;
    const steps = 20;
    const increment = (targetValue - currentValue) / steps;
    const stepDuration = duration / steps;

    let current = currentValue;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        current += increment;
        if (step >= steps) {
            element.textContent = targetValue;
            clearInterval(timer);
        } else {
            element.textContent = Math.round(current);
        }
    }, stepDuration);
}

function updateLiveStatus(data) {
    const statusDisplay = document.getElementById('detection-result');
    const largeDisplay = document.getElementById('large-result-display');

    const status = data.status || 'UNKNOWN';
    const latency = data.latency_ms || 0;

    const statusHtml = `
        <div class="status-badge ${status.toLowerCase()}">
            <i class="fa-solid ${status === 'PERFECT' ? 'fa-check-circle' : 'fa-times-circle'}"></i> ${status}
        </div>
        <div class="latency">${latency.toFixed(0)}ms</div>
    `;

    if (statusDisplay) {
        statusDisplay.innerHTML = statusHtml;
        statusDisplay.className = `detection-status ${status.toLowerCase()}`;
    }

    if (largeDisplay) {
        largeDisplay.innerHTML = statusHtml;
    }

    state.lastScanResult = data;
}

// ==================== HISTORY MANAGEMENT ====================
async function loadHistory() {
    try {
        const data = await apiCall('/api/history');
        const tbody = document.getElementById('history-table-body');
        if (!tbody) return;

        if (!data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No history available</td></tr>';
            return;
        }

        tbody.innerHTML = data.map(record => {
            const displayStatus = getDisplayStatus(record.status);
            const imgHtml = record.image ? `<img src="${record.image}" class="history-preview-img" ondblclick="openLightbox('${record.image}')" title="Double click to enlarge">` : '<span class="no-img">N/A</span>';
            return `
                <tr class="${displayStatus.toLowerCase()}">
                    <td>#${record.id}</td>
                    <td>${formatDate(record.timestamp)}</td>
                    <td><span class="status-tag ${displayStatus.toLowerCase()}">${displayStatus}</span></td>
                    <td>${imgHtml}</td>
                </tr>
            `;
        }).join('');

    } catch (e) {
        console.error(' Failed to load history:', e);
    }
}

function prependHistoryItems(items) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    const emptyState = tbody.querySelector('.empty-state');
    if (emptyState) tbody.innerHTML = '';

    items.forEach(record => {
        const row = document.createElement('tr');
        const displayStatus = getDisplayStatus(record.status);
        row.className = displayStatus.toLowerCase();
        const imgHtml = record.image ? `<img src="${record.image}" class="history-preview-img" ondblclick="openLightbox('${record.image}')" title="Double click to enlarge">` : '<span class="no-img">N/A</span>';
        row.innerHTML = `
            <td>#${record.id || 'N/A'}</td>
            <td>${formatDate(record.timestamp)}</td>
            <td><span class="status-tag ${displayStatus.toLowerCase()}">${displayStatus}</span></td>
            <td>${imgHtml}</td>
        `;
        tbody.insertBefore(row, tbody.firstChild);
    });

    while (tbody.children.length > 50) {
        tbody.removeChild(tbody.lastChild);
    }
}

function formatDate(date) {
    if (!date) return 'N/A';
    try {
        return new Date(date).toLocaleString();
    } catch (e) {
        return date;
    }
}

function exportHistory() {
    const rows = [["ID", "Time", "Status", "Defects"]];
    const tableRows = document.querySelectorAll('#history-table-body tr');

    tableRows.forEach(tr => {
        const cols = tr.querySelectorAll('td');
        if (cols.length >= 4) {
            rows.push([
                cols[0].innerText,
                cols[1].innerText,
                cols[2].innerText,
                cols[3].innerText
            ]);
        }
    });

    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `inspection_history_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showNotification('History exported successfully', 'success');
}

// ==================== SCANNING FUNCTIONS ====================


async function uploadAndScan(file) {
    if (state.isProcessing || !file) return;
    state.isProcessing = true;

    const resultBox = document.getElementById('scan-result');
    if (resultBox) {
        resultBox.classList.remove('hidden');
        resultBox.innerHTML = '<div class="detection-status waiting">Analyzing image...</div>';
    }

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/inspect_upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        displayScanResult(data);
        showNotification('Scan complete', 'success');
        refreshStats();
        loadHistory();
    } catch (e) {
        console.error(' Scan error:', e);
        showNotification('Scan failed', 'error');
        if (resultBox) resultBox.innerHTML = '<div class="detection-status defective">Scan failed</div>';
    } finally {
        state.isProcessing = false;
        const btn = document.getElementById('modal-upload-btn');
        if (btn) btn.disabled = false;
    }
}

function displayScanResult(data) {
    const resultBox = document.getElementById('scan-result');
    if (!resultBox) return;

    const result = data.result || {};
    const status = result.block_status || 'UNKNOWN';
    const defects = result.defective_saddles || 0;

    resultBox.innerHTML = `
        <div class="result-header ${status.toLowerCase()}">
            <h3>${status}</h3>
            <p>Processing: ${result.processing_time_ms ? result.processing_time_ms.toFixed(0) : 0}ms</p>
        </div>
        <div class="result-details">
            <p><strong>Detected:</strong> ${result.detected_saddles || 0} / ${result.total_saddles || 0}</p>
            <p><strong>Defects:</strong> ${defects}</p>
        </div>
        ${data.image ? `<div class="result-image"><img src="${data.image}"></div>` : ''}
    `;

    // Update live feed temporarily if needed
    const liveFeed = document.getElementById('overview-live-feed') || document.getElementById('sidebar-live-feed');
    if (data.image && liveFeed) {
        const originalSrc = liveFeed.src;
        liveFeed.src = data.image;
        setTimeout(() => { liveFeed.src = originalSrc; }, 5000);
    }
}

function triggerManualScan() {
    showNotification('Live inspection started!', 'success');
}

async function captureAndScan() {
    if (state.isProcessing) return;

    const resultBox = document.getElementById('scan-result');
    const liveView = document.querySelector('.modal-live-view');

    // Visual flash effect
    if (liveView) {
        liveView.style.filter = 'brightness(3) contrast(1.5)';
        setTimeout(() => liveView.style.filter = '', 100);
    }

    if (resultBox) {
        resultBox.classList.remove('hidden');
        resultBox.innerHTML = '<div class="detection-status waiting"><i class="fa-solid fa-spinner fa-spin"></i> Capturing & Analyzing...</div>';
    }

    try {
        const img = document.querySelector('.modal-live-view img');
        if (!img) throw new Error("Live feed not found");

        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth || 640;
        canvas.height = img.naturalHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        canvas.toBlob(blob => {
            if (blob) {
                const file = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
                uploadAndScan(file);
            } else {
                showNotification('Could not capture frame blob', 'error');
            }
        }, 'image/jpeg', 0.95);

    } catch (e) {
        console.error('Capture error:', e);
        showNotification('Failed to capture frame', 'error');
    }
}

// ==================== TEMPLATE MANAGEMENT ====================
function openTemplateWizard() {
    document.getElementById('template-list-view').classList.add('hidden');
    document.getElementById('template-wizard-view').classList.remove('hidden');
}

function closeTemplateWizard() {
    document.getElementById('template-list-view').classList.remove('hidden');
    document.getElementById('template-wizard-view').classList.add('hidden');
    // Clear wizard state
    state.templateFiles = [];
    document.getElementById('template-preview-list').innerHTML = '';
    document.getElementById('save-template-btn').classList.add('hidden');
}

function removeTemplate(index, btn) {
    btn.parentElement.remove();
    // Mark as null in array, then filter when uploading
    state.templateFiles[index] = null;

    const activeFiles = state.templateFiles.filter(f => f !== null);
    const saveBtn = document.getElementById('save-template-btn');
    if (activeFiles.length < 3) {
        saveBtn.classList.add('hidden');
    }
}

async function loadRegisteredTemplates() {
    const container = document.getElementById('registered-templates-container');
    if (!container) return;

    try {
        const images = await apiCall('/api/templates');
        if (!images || images.length === 0) {
            container.innerHTML = '<div class="empty-state">No reference templates registered yet.</div>';
            return;
        }

        container.innerHTML = images.map((img, idx) => `
            <div class="preview-item">
                <img src="${img}" onclick="openLightbox('${img}')" title="Click to enlarge">
                <span style="position: absolute; bottom: 0; left: 0; background: rgba(0,0,0,0.6); color: white; padding: 2px 5px; font-size: 10px; border-radius: 0 4px 0 0;">REF #${idx + 1}</span>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load templates:', e);
        container.innerHTML = '<div class="empty-state">Error loading templates.</div>';
    }
}

function handleTemplateFiles(files) {
    const previewList = document.getElementById('template-preview-list');
    const saveBtn = document.getElementById('save-template-btn');

    Array.from(files).forEach(file => {
        if (!file.type.startsWith('image/')) return;
        state.templateFiles.push(file);
        const fileIdx = state.templateFiles.length - 1;

        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `
                <img src="${e.target.result}">
                <button class="remove-p" onclick="removeTemplate(${fileIdx}, this)">&times;</button>
            `;
            previewList.appendChild(div);
        };
        reader.readAsDataURL(file);
    });

    if (state.templateFiles.length >= 3) {
        saveBtn.classList.remove('hidden');
    }
}

async function uploadTemplates() {
    const validFiles = state.templateFiles.filter(f => f !== null);
    if (validFiles.length < 3) {
        showNotification("Please provide at least 3 images.", "warning");
        return;
    }

    const btn = document.getElementById('save-template-btn');
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    btn.disabled = true;

    try {
        let successCount = 0;
        for (const file of validFiles) {
            const fd = new FormData();
            fd.append('file', file);
            const res = await fetch('/api/add_reference', { method: 'POST', body: fd });
            if (res.ok) successCount++;
        }
        showNotification(`Successfully added ${successCount} reference images!`, "success");
        setTimeout(() => {
            loadRegisteredTemplates();
            closeTemplateWizard();
        }, 1500);
    } catch (e) {
        console.error(e);
        showNotification("Error uploading templates", "error");
    } finally {
        btn.innerHTML = originalContent;
        btn.disabled = false;
    }
}

// ==================== CAMERA MANAGEMENT ====================
async function loadCameras() {
    const container = document.getElementById('camera-list-container');
    if (!container) return;

    container.innerHTML = `
        <div class="empty-state" style="border:none; background:transparent;">
            <i class="fa-solid fa-spinner fa-spin"></i>
            <p>Scanning for cameras...</p>
        </div>
    `;

    try {
        const res = await apiCall('/api/cameras');
        container.innerHTML = '';

        let hasCameras = false;

        // Render Saved Cameras
        if (res.saved && res.saved.length > 0) {
            res.saved.forEach(cam => {
                const card = createCameraCard(cam, true);
                container.appendChild(card);
                hasCameras = true;
            });
        }

        // Add "Mobile Device" Camera Option
        const mobileCard = createCameraCard({
            id: 'mobile',
            name: 'This Device Camera',
            type: 'mobile',
            url: 'Browser Access'
        }, false);
        container.appendChild(mobileCard);


        // Render Physical Cameras
        if (res.physical && res.physical.length > 0) {
            res.physical.forEach(cam => {
                if (cam.id === 'ip') return;
                const card = createCameraCard(cam, false);
                container.appendChild(card);
                hasCameras = true;
            });
        }

        if (!hasCameras) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-video-slash"></i>
                    <h3>No Cameras Found</h3>
                    <p>Connect a USB camera or add an IP camera to get started.</p>
                    <button class="btn-primary" onclick="openAddCameraModal()" style="margin-top:1rem;">
                        <i class="fa-solid fa-plus"></i> Add IP Camera
                    </button>
                </div>
            `;
        }
    } catch (e) {
        console.error('Failed to load cameras:', e);
        container.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-triangle-exclamation" style="color:var(--danger-color)"></i>
                <h3>Connection Error</h3>
                <p>Could not load camera list. Please check the server.</p>
                <button class="btn-secondary" onclick="loadCameras()" style="margin-top:1rem;">Retry</button>
            </div>
        `;
    }
}

function createCameraCard(cam, isSaved) {
    const div = document.createElement('div');
    div.className = 'camera-card'; // Styling handled by camera_admin.css

    // Check if active (mock logic for now, ideally backend returns active ID)
    // We can assume if we just selected it, it's active locally? 
    // Or we fetch /api/cameras/active but that endpoint doesn't exist yet.
    // For now, let's just use a simple stored variable in state?
    // Or we can check if the cam.id matches state.activeCameraId if we add that.

    const isActive = (state.activeCameraId == cam.id);
    if (isActive) div.classList.add('active');

    const icon = cam.type === 'ip' ? 'fa-network-wired' : 'fa-video';
    const safeId = String(cam.id).replace(/'/g, "\\'");
    const deleteBtn = isSaved ? `
        <button class="btn-secondary btn-sm" onclick="event.stopPropagation(); deleteCamera(${cam.db_id})" title="Delete Camera" style="color:var(--danger-color); border-color:var(--danger-color);">
            <i class="fa-solid fa-trash"></i>
        </button>` : '';

    div.onclick = () => selectCamera(cam.id); // Click whole card to select

    div.innerHTML = `
        <div class="camera-status-badge">${isActive ? 'Active' : 'Standby'}</div>
        <div class="card-content">
            <div class="card-icon"><i class="fa-solid ${icon}"></i></div>
            <h3>${cam.name || 'Unknown Camera'}</h3>
            <p>${cam.type === 'ip' ? cam.url : 'USB Direct Connection'}</p>
        </div>
        <div class="camera-actions">
            <button class="btn-primary btn-sm" onclick="event.stopPropagation(); selectCamera('${safeId}')">
                ${isActive ? '<i class="fa-solid fa-check"></i> Connected' : 'Connect'}
            </button>
            <button class="btn-secondary btn-sm" onclick="event.stopPropagation(); testExistingCameraConnection('${safeId}')" title="Test Connection">
                <i class="fa-solid fa-flask"></i>
            </button>
            ${deleteBtn}
        </div>
    `;
    return div;
}

// ==================== MOBILE/BROWSER CAMERA ====================
let mobileStream = null;
let mobileInterval = null;

async function startMobileCamera() {
    try {
        mobileStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });

        // Hide backend stream images, show local video
        const feeds = document.querySelectorAll('img[src*="/api/video_feed"]');
        feeds.forEach(img => {
            // Create video element replacement if not exists
            let video = img.parentElement.querySelector('video.mobile-cam');
            if (!video) {
                video = document.createElement('video');
                video.className = 'mobile-cam';
                video.autoplay = true;
                video.playsInline = true;
                video.style.width = '100%';
                video.style.height = '100%';
                video.style.objectFit = 'cover';
                img.parentElement.appendChild(video);
            }
            img.style.display = 'none';
            video.style.display = 'block';
            video.srcObject = mobileStream;
        });

        startMobileProcessing();
        showNotification('Mobile camera started', 'success');

        // Update Active Camera State in UI
        state.activeCameraId = 'mobile';
        loadCameras();

    } catch (e) {
        console.error('Mobile camera error:', e);
        showNotification('Could not access camera', 'error');
    }
}

function stopMobileCamera() {
    if (mobileStream) {
        mobileStream.getTracks().forEach(track => track.stop());
        mobileStream = null;
    }
    if (mobileInterval) {
        clearInterval(mobileInterval);
        mobileInterval = null;
    }

    // Restore backend stream
    const feeds = document.querySelectorAll('img[src*="/api/video_feed"]');
    feeds.forEach(img => {
        const video = img.parentElement.querySelector('video.mobile-cam');
        if (video) video.style.display = 'none';
        img.style.display = 'block';
        img.src = img.src.split('?')[0] + '?t=' + new Date().getTime();
    });
}

async function startMobileProcessing() {
    const video = document.querySelector('video.mobile-cam');
    if (!video) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    mobileInterval = setInterval(async () => {
        if (!state.isProcessing && video.readyState === video.HAVE_ENOUGH_DATA) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            canvas.toBlob(async (blob) => {
                if (!blob) return;

                // Send to backend for inspection
                try {
                    const fd = new FormData();
                    fd.append('file', blob, 'mobile_frame.jpg');

                    // We use a flag 'no_history=true' if backend supports, but for now standard upload
                    const response = await fetch('/api/inspect_upload', {
                        method: 'POST',
                        body: fd
                    });

                    if (response.ok) {
                        const data = await response.json();
                        // Update UI with result (optional, or rely on WebSocket broadcasts which might be better)
                        // displayScanResult(data); 
                        // Note: displayScanResult updates the 'scan-modal' but we want live updates.
                        // WebSocket broadcast should handle 'live_status', so we don't need to manually update here
                        // IF the backend broadcasts for uploads too.
                    }
                } catch (e) {
                    console.error('Mobile processing error', e);
                }
            }, 'image/jpeg', 0.8);
        }
    }, 1000); // 1 FPS for mobile inspection to save bandwidth/processing
}

// Update selectCamera to handle "mobile"
async function testNewCameraConnection() {
    const url = document.getElementById('new-camera-url').value;
    if (!url) {
        showNotification('Please enter a URL to test', 'warning');
        return;
    }
    await performCameraTest(url);
}

async function testExistingCameraConnection(url) {
    if (url === 'mobile') {
        showNotification('Mobile camera test not required', 'info');
        return;
    }
    await performCameraTest(url);
}

async function performCameraTest(url) {
    showNotification('Testing connection...', 'info');
    try {
        const res = await apiCall('/api/camera/test', {
            method: 'POST',
            body: JSON.stringify({ url })
        });
        showNotification(res.message || 'Connection successful!', 'success');
    } catch (e) {
        showNotification('Connection failed. Check URL/Network.', 'error');
    }
}

async function selectCamera(cameraId) {
    if (cameraId === 'mobile') {
        await startMobileCamera();
        return;
    } else {
        // If switching away from mobile
        if (state.activeCameraId === 'mobile') {
            stopMobileCamera();
        }
    }

    try {
        // Determine if ID is int (USB) or str (URL)
        const payload = { index: null, url: null };
        if (typeof cameraId === 'number' || !isNaN(cameraId)) {
            payload.index = parseInt(cameraId);
        } else {
            payload.url = cameraId;
        }

        const res = await apiCall('/api/camera/select', {
            method: 'POST',
            body: JSON.stringify(payload)
        });

        showNotification(res.message || 'Camera selected', 'success');

        // Update local state and re-render to show active status
        state.activeCameraId = cameraId;
        loadCameras(); // re-render to update active class

        // Refresh feeds
        const feeds = document.querySelectorAll('img[src*="/api/video_feed"]');
        feeds.forEach(img => {
            const baseUrl = img.src.split('?')[0];
            img.src = `${baseUrl}?t=${new Date().getTime()}`;
        });
    } catch (e) {
        showNotification('Failed to switch camera', 'error');
    }
}

async function saveNewCamera() {
    const name = document.getElementById('new-camera-name').value;
    const type = document.getElementById('new-camera-type').value;
    const url = document.getElementById('new-camera-url').value;

    if (!name || !url) {
        showNotification('Please fill in Name and URL', 'warning');
        return;
    }

    try {
        await apiCall('/api/cameras/add', {
            method: 'POST',
            body: JSON.stringify({ name, type, url })
        });

        showNotification('Camera saved successfully', 'success');
        closeAddCameraModal();
        loadCameras();
    } catch (e) {
        showNotification('Failed to save camera', 'error');
    }
}

async function deleteCamera(dbId) {
    if (!confirm('Are you sure you want to delete this camera?')) return;

    try {
        await apiCall(`/api/cameras/${dbId}`, { method: 'DELETE' });
        showNotification('Camera deleted', 'success');
        loadCameras();
    } catch (e) {
        showNotification('Failed to delete camera', 'error');
    }
}

function openAddCameraModal() {
    document.getElementById('add-camera-modal').classList.add('active');
}

function closeAddCameraModal() {
    document.getElementById('add-camera-modal').classList.remove('active');
    // clear inputs
    document.getElementById('new-camera-name').value = '';
    document.getElementById('new-camera-url').value = '';
}

async function stopInspection() {
    try {
        const res = await apiCall('/api/camera/stop', { method: 'POST' });
        showNotification(res.message || 'Inspection stopped', 'info');
        // Force refresh of feeds to show NO SIGNAL
        refreshFeeds();
    } catch (e) {
        showNotification('Failed to stop inspection', 'error');
    }
}

async function startInspection() {
    try {
        const res = await apiCall('/api/camera/select', {
            method: 'POST',
            body: JSON.stringify({ index: 0 })
        });
        showNotification(res.message || 'Inspection started', 'success');
        refreshFeeds();
        state.activeCameraId = 0;
    } catch (e) {
        showNotification('Failed to start inspection', 'error');
    }
}

function refreshFeeds() {
    const feeds = document.querySelectorAll('img[src*="/api/video_feed"]');
    feeds.forEach(img => {
        const baseUrl = img.src.split('?')[0];
        img.src = `${baseUrl}?t=${new Date().getTime()}`;
    });
}

// ==================== NAVIGATION & UI ====================
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.content-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.dataset.target;

            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            sections.forEach(s => {
                s.classList.toggle('active', s.id === `${target}-section`);
            });

            const breadcrumb = document.getElementById('current-page');
            if (breadcrumb) breadcrumb.textContent = item.textContent.trim();

            state.currentSection = target;
            if (target === 'history') loadHistory();
            if (target === 'templates') loadRegisteredTemplates();
            if (target === 'camera') loadCameras();
        });
    });

    // Mobile Menu
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const sidebar = document.getElementById('sidebar');
    if (mobileBtn && sidebar) {
        mobileBtn.addEventListener('click', () => sidebar.classList.toggle('active'));
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 && !sidebar.contains(e.target) && !mobileBtn.contains(e.target)) {
                sidebar.classList.remove('active');
            }
        });
    }
}

function initModals() {
    const modal = document.getElementById('scan-modal');
    const openBtn = document.getElementById('open-scan-modal');
    const closeBtns = document.querySelectorAll('.close-modal');

    if (openBtn) openBtn.onclick = () => modal.classList.add('active');
    closeBtns.forEach(btn => btn.onclick = () => modal.classList.remove('active'));

    window.onclick = (e) => { if (e.target === modal) modal.classList.remove('active'); };

    // Tab buttons
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.onclick = () => {
            const tab = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tab-content').forEach(c => {
                c.classList.toggle('active', c.id === `tab-${tab}`);
            });
        };
    });

    // Modal File Input
    const modalFileInput = document.getElementById('modal-file-upload');
    const modalUploadBtn = document.getElementById('modal-upload-btn');
    if (modalFileInput) {
        modalFileInput.onchange = (e) => {
            const fileName = e.target.files[0] ? e.target.files[0].name : "No file chosen";
            document.getElementById('file-name').innerText = fileName;
            modalUploadBtn.disabled = !e.target.files[0];
        };
        modalUploadBtn.onclick = () => {
            const file = modalFileInput.files[0];
            if (file) uploadAndScan(file);
        };
    }
}

function initTemplateZone() {
    const zone = document.getElementById('template-upload-zone');
    const input = document.getElementById('template-files');
    const saveBtn = document.getElementById('save-template-btn');

    if (zone) {
        zone.onclick = () => input.click();
        input.onchange = (e) => handleTemplateFiles(e.target.files);

        zone.ondragover = (e) => { e.preventDefault(); zone.style.background = '#eef7fc'; };
        zone.ondragleave = (e) => { e.preventDefault(); zone.style.background = 'transparent'; };
        zone.ondrop = (e) => {
            e.preventDefault();
            zone.style.background = 'transparent';
            handleTemplateFiles(e.dataTransfer.files);
        };

        if (saveBtn) saveBtn.onclick = uploadTemplates;
    }
}

function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.innerHTML = `<i class="fa-solid ${type === 'success' ? 'fa-circle-check' : 'fa-info-circle'}"></i> ${message}`;
    document.body.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 100);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function openLightbox(src) {
    let lightbox = document.getElementById('lightbox-modal');
    if (!lightbox) {
        lightbox = document.createElement('div');
        lightbox.id = 'lightbox-modal';
        lightbox.className = 'lightbox-modal';
        lightbox.innerHTML = `<img src="" class="lightbox-content">`;
        lightbox.onclick = () => lightbox.classList.remove('active');
        document.body.appendChild(lightbox);
    }

    lightbox.querySelector('img').src = src;
    lightbox.classList.add('active');
}

async function refreshStats() {
    try {
        const data = await apiCall('/api/stats');
        updateStats(data);
    } catch (e) {
        console.error(' Stats refresh error:', e);
    }
}

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Dashboard initializing...');

    initWebSocket();
    initNavigation();
    initModals();
    initTemplateZone();

    // Initial data load
    refreshStats();
    loadHistory();
    loadRegisteredTemplates();

    // Set up periodic refresh as backup to WebSocket
    setInterval(refreshStats, CONFIG.STATS_REFRESH_INTERVAL);

    console.log('✅ Dashboard ready');
});

// Styles for notifications (Dynamic)
const style = document.createElement('style');
style.textContent = `
    .toast-notification {
        position: fixed; bottom: 20px; right: 20px; padding: 12px 24px;
        background: #333; color: white; border-radius: 8px; font-weight: 500;
        display: flex; align-items: center; gap: 10px; z-index: 9999;
        transform: translateY(100px); transition: 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .toast-notification.show { transform: translateY(0); }
    .toast-notification.success { background: #2ecc71; }
    .toast-notification.error { background: #e74c3c; }
    .toast-notification.warning { background: #f1c40f; color: #333; }
    .status-tag { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
    .status-tag.perfect, .status-tag.pass { background: #dff9e8; color: #2ecc71; }
    .status-tag.defective, .status-tag.fail { background: #fadbd8; color: #e74c3c; }
    .preview-item { position: relative; width: 60px; height: 60px; }
    .preview-item img { width: 100%; height: 100%; object-fit: cover; border-radius: 4px; }
    .remove-p { position: absolute; top: -5px; right: -5px; background: red; color: white; border: none; border-radius: 50%; width: 15px; height: 15px; font-size: 10px; cursor: pointer; }
`;
document.head.appendChild(style);
