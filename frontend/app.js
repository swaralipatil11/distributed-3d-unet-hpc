// Dynamic host resolution for API and WebSockets
// Allows local dev servers (e.g. Vite on port 5173) to communicate with FastAPI backend on port 8000
const isDevServer = window.location.hostname === "localhost" && window.location.port !== "8000" && window.location.port !== "";
const apiBase = isDevServer ? "http://localhost:8000" : "";
const wsHost = isDevServer ? "localhost:8000" : window.location.host;

// State variables
let socket = null;
let currentVolumeId = "";
let currentSliceIdx = 64;
let currentModalityIdx = 0;
let currentOpacity = 0.7;

// DOM Elements
const btnSimulate = document.getElementById("btn-simulate");
const selectVolume = document.getElementById("select-volume");
const connectionBadge = document.getElementById("connection-badge");

const ingestStatus = document.getElementById("ingest-status");
const ingestPercent = document.getElementById("ingest-percent");
const progressBarFill = document.getElementById("ingest-progress-bar");
const consoleLogs = document.getElementById("console-logs");

const mriSlice = document.getElementById("mri-slice");
const labelSlice = document.getElementById("label-slice");
const viewerPlaceholder = document.getElementById("viewer-placeholder");

const selectModality = document.getElementById("select-modality");
const sliderSlice = document.getElementById("slider-slice");
const sliceValText = document.getElementById("slice-val");
const sliderOpacity = document.getElementById("slider-opacity");
const opacityValText = document.getElementById("opacity-val");

const metricResize = document.getElementById("metric-resize");
const metricInference = document.getElementById("metric-inference");
const metricLatency = document.getElementById("metric-latency");

// Uploader Elements
const dropZone = document.getElementById("drop-zone");
const fileUpload = document.getElementById("file-upload");
const uploadProgressContainer = document.getElementById("upload-progress-container");
const uploadStatusText = document.getElementById("upload-status-text");
const uploadPercentText = document.getElementById("upload-percent-text");
const uploadProgressBar = document.getElementById("upload-progress-bar");

// Patient Info and Download Elements
const patientCard = document.getElementById("patient-card");
const patName = document.getElementById("pat-name");
const patId = document.getElementById("pat-id");
const patDate = document.getElementById("pat-date");
const patDesc = document.getElementById("pat-desc");
const btnDownload = document.getElementById("btn-download");


// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    connectWebSocket();
    loadVolumeList();
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    // Simulate Scan button
    btnSimulate.addEventListener("click", triggerSimulation);

    // Volume Select dropdown
    selectVolume.addEventListener("change", (e) => {
        const val = e.target.value;
        if (val) {
            currentVolumeId = val;
            enableControls();
            
            // Try to find selected volume stats to update metrics
            const selectedOption = selectVolume.options[selectVolume.selectedIndex];
            if (selectedOption.dataset.stats) {
                const stats = JSON.parse(selectedOption.dataset.stats);
                updateMetrics(stats);
                
                // Load patient metadata if present
                if (stats.patient_metadata) {
                    patName.textContent = stats.patient_metadata.patient_name || "Anonymous";
                    patId.textContent = stats.patient_metadata.patient_id || "N/A";
                    patDate.textContent = stats.patient_metadata.study_date || "N/A";
                    patDesc.textContent = stats.patient_metadata.study_description || "N/A";
                    patientCard.style.display = "block";
                } else {
                    patientCard.style.display = "none";
                }
            }
            
            updateSliceImages();
        } else {
            currentVolumeId = "";
            disableControls();
        }
    });

    // Download button event
    btnDownload.addEventListener("click", () => {
        if (currentVolumeId) {
            window.location.href = `${apiBase}/api/volume/${currentVolumeId}/download`;
        }
    });

    // Modality Select dropdown
    selectModality.addEventListener("change", (e) => {
        currentModalityIdx = parseInt(e.target.value);
        updateSliceImages();
    });

    // Slice Depth Slider
    sliderSlice.addEventListener("input", (e) => {
        currentSliceIdx = parseInt(e.target.value);
        sliceValText.textContent = `${currentSliceIdx} / 127`;
        updateSliceImages();
    });

    // Opacity Slider
    sliderOpacity.addEventListener("input", (e) => {
        const opacityVal = parseInt(e.target.value);
        currentOpacity = opacityVal / 100;
        opacityValText.textContent = `${opacityVal}%`;
        labelSlice.style.opacity = currentOpacity;
    });

    // Drag & Drop events
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    ["dragleave", "dragend"].forEach((type) => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileUpload.addEventListener("change", (e) => {
        if (fileUpload.files.length) {
            handleFileUpload(fileUpload.files[0]);
        }
    });
}

// Log line to terminal console
function logToTerminal(message, type = "packet") {
    const logLine = document.createElement("div");
    logLine.className = `log-line ${type}`;
    logLine.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    
    // Insert at front because console has flex-direction: column-reverse
    consoleLogs.insertBefore(logLine, consoleLogs.firstChild);
}

// Connect WebSocket for real-time progress updates
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${wsHost}/ws`;
    
    logToTerminal(`Connecting to WebSocket server...`, "system");
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        connectionBadge.textContent = "Connected";
        connectionBadge.className = "badge connected";
        logToTerminal("WebSocket connection established.", "system");
    };
    
    socket.onclose = () => {
        connectionBadge.textContent = "Disconnected";
        connectionBadge.className = "badge disconnected";
        logToTerminal("WebSocket connection closed. Retrying in 5 seconds...", "system");
        setTimeout(connectWebSocket, 5000);
    };
    
    socket.onerror = (err) => {
        logToTerminal(`WebSocket error occurred.`, "system");
    };
    
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleSocketMessage(data);
        } catch (e) {
            // Ignore non-json logs
        }
    };
}

// Handle socket events
function handleSocketMessage(msg) {
    if (msg.type === "slice_received") {
        const percent = Math.round((msg.packet_index + 1) / msg.total_packets * 100);
        
        ingestStatus.textContent = `Receiving scan slice ${msg.packet_index + 1} of ${msg.total_packets}...`;
        ingestPercent.textContent = `${percent}%`;
        progressBarFill.style.width = `${percent}%`;
        
        logToTerminal(
            `STREAM: Slice [${msg.packet_index + 1}/${msg.total_packets}] received for Volume: ${msg.volume_id} (${msg.payload_size_kb.toFixed(1)} KB)`,
            "packet"
        );
    } 
    else if (msg.type === "reassembly_started") {
        ingestStatus.textContent = "Reassembling slices...";
        logToTerminal(`SYSTEM: All slices received for volume ${msg.volume_id}. Reassembling 3D grid in memory...`, "system");
    }
    else if (msg.type === "inference_started") {
        ingestStatus.textContent = "Running 3D JIT Inference...";
        logToTerminal(`SYSTEM: Initializing 3D trilinear resizer and executing TorchScript 3D U-Net forward pass...`, "system");
    }
    else if (msg.type === "inference_completed") {
        ingestStatus.textContent = "Inference completed!";
        ingestPercent.textContent = "100%";
        progressBarFill.style.width = "100%";
        logToTerminal(`SUCCESS: Volumetric segmentation completed for Volume: ${msg.volume_id} (Model: ${msg.inference_time.toFixed(3)}s, Total: ${msg.total_latency.toFixed(2)}s). Saved output to disk.`, "completed");
        
        // Reload volume list and select the newly processed one
        loadVolumeList(msg.volume_id);
        
        // Re-enable simulation button
        btnSimulate.disabled = false;
    }
    else if (msg.type === "error") {
        ingestStatus.textContent = "Error occurred.";
        logToTerminal(`ERROR: ${msg.message}`, "system");
        btnSimulate.disabled = false;
    }
}

// Trigger simulation scan run
async function triggerSimulation() {
    btnSimulate.disabled = true;
    
    // Reset progress interface
    ingestStatus.textContent = "Initializing simulation...";
    ingestPercent.textContent = "0%";
    progressBarFill.style.width = "0%";
    
    logToTerminal("Simulating raw medical scan stream from scan machine...", "system");
    
    try {
        const response = await fetch(`${apiBase}/api/simulate`, { method: "POST" });
        const data = await response.json();
        if (data.status === "started") {
            logToTerminal(`Simulation started: streaming slices for scan volume.`, "system");
        } else {
            logToTerminal(`Simulation failed: ${data.message}`, "system");
            btnSimulate.disabled = false;
        }
    } catch (err) {
        logToTerminal(`Error triggering simulation: ${err.message}`, "system");
        btnSimulate.disabled = false;
    }
}

// Upload MRI Scan file or DICOM zip folder via API
function handleFileUpload(file) {
    const filename = file.name;
    if (!filename.endsWith(".nii.gz") && !filename.endsWith(".nii") && !filename.endsWith(".zip")) {
        logToTerminal(`ERROR: Unsupported file format. Please upload .nii.gz, .nii, or .zip.`, "system");
        alert("Unsupported file format. Please upload a NIfTI (.nii.gz) or DICOM ZIP (.zip) file.");
        return;
    }

    logToTerminal(`Uploading scan file: ${filename}...`, "system");
    
    // Reset progress UI
    ingestStatus.textContent = "Uploading scan...";
    ingestPercent.textContent = "0%";
    progressBarFill.style.width = "0%";
    
    uploadProgressContainer.style.display = "block";
    uploadStatusText.textContent = `Uploading ${filename}...`;
    uploadPercentText.textContent = "0%";
    uploadProgressBar.style.width = "0%";
    
    const formData = new FormData();
    formData.append("file", file);
    
    const xhr = new XMLHttpRequest();
    
    // Track upload progress
    xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            uploadPercentText.textContent = `${percent}%`;
            uploadProgressBar.style.width = `${percent}%`;
        }
    });
    
    // Complete upload
    xhr.onload = () => {
        if (xhr.status === 200) {
            const res = JSON.parse(xhr.responseText);
            logToTerminal(`Upload completed. Processing scan on worker engine... (Upload ID: ${res.upload_id})`, "completed");
            uploadStatusText.textContent = "Processing file on server...";
            
            // Hide progress container after a short delay
            setTimeout(() => {
                uploadProgressContainer.style.display = "none";
            }, 3000);
        } else {
            let errorMsg = "Upload failed.";
            try {
                const res = JSON.parse(xhr.responseText);
                errorMsg = res.message || errorMsg;
            } catch (e) {}
            logToTerminal(`ERROR: Upload failed: ${errorMsg}`, "system");
            uploadStatusText.textContent = "Upload failed.";
            uploadProgressContainer.style.display = "none";
        }
    };
    
    // Error handling
    xhr.onerror = () => {
        logToTerminal("ERROR: Network upload error occurred.", "system");
        uploadStatusText.textContent = "Upload failed.";
        uploadProgressContainer.style.display = "none";
    };
    
    xhr.open("POST", `${apiBase}/api/upload`);
    xhr.send(formData);
}

// Load List of processed volumes from API
async function loadVolumeList(selectedVolumeId = "") {
    try {
        const response = await fetch(`${apiBase}/api/volumes`);
        const volumes = await response.json();
        
        selectVolume.innerHTML = "";
        
        if (volumes.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "-- No Volumes Processed --";
            selectVolume.appendChild(opt);
            disableControls();
            return;
        }
        
        // Sort newest first
        volumes.sort((a, b) => b.timestamp - a.timestamp);
        
        volumes.forEach((vol) => {
            const opt = document.createElement("option");
            opt.value = vol.volume_id;
            opt.dataset.stats = JSON.stringify(vol);
            
            // Format time
            const dateStr = new Date(vol.timestamp * 1000).toLocaleTimeString();
            opt.textContent = `${vol.volume_id} (Processed at ${dateStr})`;
            
            selectVolume.appendChild(opt);
        });
        
        // If a specific volume needs selecting, select it; otherwise default to first
        if (selectedVolumeId) {
            selectVolume.value = selectedVolumeId;
        }
        
        // Trigger select change event
        selectVolume.dispatchEvent(new Event("change"));
        
    } catch (err) {
        logToTerminal(`Error loading volume list: ${err.message}`, "system");
    }
}

// Enable sliders and visualizer controls
function enableControls() {
    selectModality.disabled = false;
    sliderSlice.disabled = false;
    sliderOpacity.disabled = false;
    btnDownload.disabled = false;
    
    viewerPlaceholder.style.display = "none";
    mriSlice.style.display = "block";
    labelSlice.style.display = "block";
}

// Disable sliders and visualizer controls
function disableControls() {
    selectModality.disabled = true;
    sliderSlice.disabled = true;
    sliderOpacity.disabled = true;
    btnDownload.disabled = true;
    
    viewerPlaceholder.style.display = "flex";
    mriSlice.style.display = "none";
    labelSlice.style.display = "none";
    patientCard.style.display = "none";
    
    // Clear metrics
    metricResize.textContent = "0.000s";
    metricInference.textContent = "0.000s";
    metricLatency.textContent = "0.00s";
}

// Update slice image and overlay src paths
function updateSliceImages() {
    if (!currentVolumeId) return;
    
    // Update labels and MRI image sources
    const mriSrc = `${apiBase}/api/volume/${currentVolumeId}/slice/${currentSliceIdx}/modality/${currentModalityIdx}?t=${Date.now()}`;
    const labelSrc = `${apiBase}/api/volume/${currentVolumeId}/slice/${currentSliceIdx}/label?t=${Date.now()}`;
    
    mriSlice.src = mriSrc;
    labelSlice.src = labelSrc;
    labelSlice.style.opacity = currentOpacity;
}

// Update metrics panel
function updateMetrics(stats) {
    metricResize.textContent = `${stats.resize_time.toFixed(3)}s`;
    metricInference.textContent = `${stats.inference_time.toFixed(3)}s`;
    metricLatency.textContent = `${stats.total_latency.toFixed(2)}s`;
}
