// ===============================
// Konfigurasi
// ===============================
const BASE_URL = "http://103.191.92.79";
const YOLO_URL = `${BASE_URL}/predict`;
const KERAS_URL = `${BASE_URL}/predict-sentence`;

// ===============================
// State Management
// ===============================
let currentUser = null;
let webcamStream = null;
let detectionInterval = null;
let isDetecting = false;
let autoSpeakEnabled = true;
let menuItems = [];
let detectionHistory = [];

// ===============================
// Authentication
// ===============================
document.getElementById('loginForm')?.addEventListener('submit', function(e) {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    // Simple demo authentication
    if (username === 'admin' && password === 'admin123') {
        currentUser = username;
        document.getElementById('welcomeUser').textContent = `Selamat datang, ${username}`;
        switchPage('mainPage');
        loadMenuItems();
        loadHistory();
    } else {
        alert('Username atau password salah!');
    }
});

document.getElementById('logoutBtn')?.addEventListener('click', function() {
    currentUser = null;
    stopCamera();
    switchPage('loginPage');
});

function switchPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId)?.classList.add('active');
}

// ===============================
// Tab Navigation
// ===============================
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const tabName = this.dataset.tab;
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName + 'Tab')?.classList.add('active');
    });
});

// ===============================
// Camera & Detection
// ===============================
document.getElementById('startCamera')?.addEventListener('click', startCamera);
document.getElementById('stopCamera')?.addEventListener('click', stopCamera);

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' } 
        });
        const video = document.getElementById('webcam');
        video.srcObject = stream;
        webcamStream = stream;
        
        // Start auto detection
        isDetecting = true;
        detectionInterval = setInterval(captureAndDetect, 2000);
        
        document.getElementById('detectionOutput').textContent = 'Kamera aktif, mendeteksi...';
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Tidak dapat mengakses kamera. Pastikan izin kamera telah diberikan.');
    }
}

function stopCamera() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    
    isDetecting = false;
    document.getElementById('webcam').srcObject = null;
    document.getElementById('detectionOutput').textContent = 'Kamera berhenti';
}

async function captureAndDetect() {
    if (!isDetecting || !webcamStream) return;
    
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const base64Image = canvas.toDataURL('image/jpeg');
    
    // Try YOLO detection first (letter detection)
    try {
        const result = await detectWithYOLO(base64Image);
        if (result && result.prediction && result.prediction !== '-') {
            displayDetectionResult(result.prediction, 'huruf', 85);
            return;
        }
    } catch (error) {
        console.error('YOLO detection error:', error);
    }
    
    // Try Keras detection (word detection)
    try {
        const result = await detectWithKeras(base64Image);
        if (result && result.prediction && result.prediction !== '-') {
            displayDetectionResult(result.prediction, 'kata', 75);
        }
    } catch (error) {
        console.error('Keras detection error:', error);
    }
}

async function detectWithYOLO(base64Image) {
    const response = await fetch(YOLO_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

async function detectWithKeras(base64Image) {
    const response = await fetch(KERAS_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

function displayDetectionResult(prediction, type, confidence) {
    const outputElement = document.getElementById('detectionOutput');
    const speechElement = document.getElementById('speechText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceProgress = document.getElementById('confidenceProgress');
    
    // Display result
    const displayText = type === 'huruf' ? `Huruf: ${prediction.toUpperCase()}` : `Kata: ${prediction.toUpperCase()}`;
    outputElement.textContent = displayText;
    speechElement.textContent = prediction.toUpperCase();
    
    // Update confidence
    confidenceValue.textContent = `${confidence}%`;
    confidenceProgress.style.width = `${confidence}%`;
    
    // Auto speak
    if (autoSpeakEnabled) {
        speak(prediction);
    }
    
    // Add to history
    addToHistory(prediction, type, confidence);
}

function speak(text) {
    if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'id-ID';
        utterance.rate = 0.9;
        utterance.pitch = 1;
        
        window.speechSynthesis.speak(utterance);
    }
}

// ===============================
// Auto Speak Toggle
// ===============================
document.getElementById('autoSpeakToggle')?.addEventListener('click', function() {
    autoSpeakEnabled = !autoSpeakEnabled;
    this.textContent = autoSpeakEnabled ? '🔊 Auto Speak: ON' : '🔇 Auto Speak: OFF';
    this.classList.toggle('auto-speak-on');
    this.classList.toggle('auto-speak-off');
});

// ===============================
// History Management
// ===============================
function addToHistory(prediction, type, confidence) {
    const timestamp = new Date().toLocaleString('id-ID');
    const historyItem = {
        id: Date.now(),
        prediction,
        type,
        confidence,
        timestamp
    };
    
    detectionHistory.unshift(historyItem);
    
    // Keep only last 50 items
    if (detectionHistory.length > 50) {
        detectionHistory = detectionHistory.slice(0, 50);
    }
    
    saveHistory();
    renderHistory();
}

function loadHistory() {
    const saved = localStorage.getItem('detectionHistory');
    if (saved) {
        detectionHistory = JSON.parse(saved);
        renderHistory();
    }
}

function saveHistory() {
    localStorage.setItem('detectionHistory', JSON.stringify(detectionHistory));
}

function renderHistory() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    if (detectionHistory.length === 0) {
        historyList.innerHTML = '<p class="empty-state">Belum ada riwayat deteksi</p>';
        return;
    }
    
    historyList.innerHTML = detectionHistory.map(item => `
        <div class="history-item">
            <div class="history-icon">
                ${item.type === 'huruf' ? '🔤' : '📝'}
            </div>
            <div class="history-details">
                <div class="history-prediction">${item.prediction.toUpperCase()}</div>
                <div class="history-meta">
                    <span class="history-type">${item.type === 'huruf' ? 'Huruf' : 'Kata'}</span>
                    <span class="history-confidence">${item.confidence}%</span>
                    <span class="history-time">${item.timestamp}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// ===============================
// Menu Management
// ===============================
document.getElementById('addMenuBtn')?.addEventListener('click', function() {
    document.getElementById('addMenuModal').style.display = 'flex';
});

document.querySelector('.close')?.addEventListener('click', function() {
    document.getElementById('addMenuModal').style.display = 'none';
});

document.querySelector('.cancel-btn')?.addEventListener('click', function() {
    document.getElementById('addMenuModal').style.display = 'none';
});

document.getElementById('addMenuForm')?.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const menuItem = {
        id: Date.now(),
        name: formData.get('name'),
        price: parseFloat(formData.get('price')),
        category: formData.get('category'),
        image: formData.get('image') || 'https://via.placeholder.com/200x150?text=No+Image'
    };
    
    menuItems.push(menuItem);
    saveMenuItems();
    renderMenuItems();
    
    e.target.reset();
    document.getElementById('addMenuModal').style.display = 'none';
});

function loadMenuItems() {
    const saved = localStorage.getItem('menuItems');
    if (saved) {
        menuItems = JSON.parse(saved);
    } else {
        // Default menu items
        menuItems = [
            {
                id: 1,
                name: 'Nasi Goreng',
                price: 15000,
                category: 'makanan',
                image: 'https://via.placeholder.com/200x150?text=Nasi+Goreng'
            },
            {
                id: 2,
                name: 'Es Teh',
                price: 5000,
                category: 'minuman',
                image: 'https://via.placeholder.com/200x150?text=Es+Teh'
            },
            {
                id: 3,
                name: 'Kentang Goreng',
                price: 10000,
                category: 'snack',
                image: 'https://via.placeholder.com/200x150?text=Kentang+Goreng'
            }
        ];
    }
    renderMenuItems();
}

function saveMenuItems() {
    localStorage.setItem('menuItems', JSON.stringify(menuItems));
}

function renderMenuItems() {
    const menuGrid = document.getElementById('menuGrid');
    if (!menuGrid) return;
    
    if (menuItems.length === 0) {
        menuGrid.innerHTML = '<p class="empty-state">Belum ada menu. Tambahkan menu baru!</p>';
        return;
    }
    
    menuGrid.innerHTML = menuItems.map(item => `
        <div class="menu-card">
            <div class="menu-image" style="background-image: url('${item.image}')"></div>
            <div class="menu-info">
                <h4 class="menu-name">${item.name}</h4>
                <p class="menu-price">Rp ${item.price.toLocaleString('id-ID')}</p>
                <span class="menu-category">${item.category}</span>
            </div>
            <div class="menu-actions">
                <button class="delete-btn" onclick="deleteMenuItem(${item.id})">🗑️ Hapus</button>
            </div>
        </div>
    `).join('');
}

function deleteMenuItem(id) {
    if (confirm('Apakah Anda yakin ingin menghapus menu ini?')) {
        menuItems = menuItems.filter(item => item.id !== id);
        saveMenuItems();
        renderMenuItems();
    }
}

// ===============================
// Close modal when clicking outside
// ===============================
window.addEventListener('click', function(e) {
    const modal = document.getElementById('addMenuModal');
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// ===============================
// Initialize on page load
// ===============================
window.addEventListener('DOMContentLoaded', function() {
    // Check if user was logged in
    const savedUser = localStorage.getItem('currentUser');
    if (savedUser) {
        currentUser = savedUser;
        document.getElementById('welcomeUser').textContent = `Selamat datang, ${savedUser}`;
        switchPage('mainPage');
        loadMenuItems();
        loadHistory();
    }
});

// Save current user on login
function saveCurrentUser() {
    if (currentUser) {
        localStorage.setItem('currentUser', currentUser);
    } else {
        localStorage.removeItem('currentUser');
    }
}