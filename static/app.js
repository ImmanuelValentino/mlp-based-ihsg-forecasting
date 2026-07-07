// STOCKMIXER DASHBOARD FRONTEND LOGIC

document.addEventListener('DOMContentLoaded', () => {
    // State management
    let state = {
        selectedDate: '',
        history: [],
        currentData: null,
        searchQuery: '',
        currentFilter: 'all', // all, bullish, bearish
        sortColumn: 'AI_Score',
        sortDirection: 'desc', // desc or asc
        isPipelineRunning: false,
        currentModel: 'v1' // v1 or v3
    };


    // DOM Elements
    const pipelineDateInput = document.getElementById('pipeline-date');
    const btnRunV1 = document.getElementById('btn-run-v1');
    const btnRunV3 = document.getElementById('btn-run-v3');
    const historyList = document.getElementById('history-list');
    const historyLoading = document.getElementById('history-loading');
    const clockDisplay = document.getElementById('clock-display');
    const selectedDateTitle = document.getElementById('selected-date-title');
    const selectedDateSubtitle = document.getElementById('selected-date-subtitle');

    
    // Console DOM
    const consoleSection = document.getElementById('console-section');
    const consoleToggle = document.getElementById('console-toggle');
    const consoleToggleIcon = document.getElementById('console-toggle-icon');
    const consoleOutput = document.getElementById('console-output');
    const consoleBody = document.getElementById('console-body');
    const pipelineBadge = document.getElementById('pipeline-badge');
    const currentStepText = document.getElementById('current-step-text');
    
    // Stats DOM
    const statTotalTickers = document.getElementById('stat-total-tickers');
    const statBullishTickers = document.getElementById('stat-bullish-tickers');
    const statBearishTickers = document.getElementById('stat-bearish-tickers');
    const statTopScore = document.getElementById('stat-top-score');
    
    // Workspace DOM
    const visualImgContainer = document.getElementById('visual-img-container');
    const visualPlaceholder = document.getElementById('visual-placeholder');
    const btnZoomImage = document.getElementById('btn-zoom-image');
    const modelTabControl = document.getElementById('model-tab-control');
    const btnDownloadCsv = document.getElementById('btn-download-csv');
    
    // Table DOM
    const tableSearch = document.getElementById('table-search');
    const filterButtons = document.querySelectorAll('.filter-btn');
    const tableBody = document.getElementById('table-body');
    const tableHeaders = document.querySelectorAll('.recommendations-table th.sortable');
    
    // Modal DOM
    const imageModal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalCaption = document.getElementById('modal-caption');
    const closeModal = document.getElementById('close-modal');

    // --- SETUP & INITIALIZATION ---
    
    // Set default date input to today
    const todayStr = new Date().toISOString().split('T')[0];
    pipelineDateInput.value = todayStr;
    
    // Clock initialization
    setInterval(() => {
        const now = new Date();
        clockDisplay.textContent = now.toLocaleTimeString('id-ID') + ' - ' + now.toLocaleDateString('id-ID', { weekday: 'long' });
    }, 1000);

    // Initial load
    fetchHistory();
    checkActivePipeline();

    // --- EVENT LISTENERS ---

    // Collapsible Console
    consoleToggle.addEventListener('click', () => {
        consoleSection.classList.toggle('collapsed');
    });

    // Model tabs switcher (V1 vs V3)
    if (modelTabControl) {
        const modelTabBtns = modelTabControl.querySelectorAll('.tab-btn');
        modelTabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                modelTabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.currentModel = btn.getAttribute('data-model');
                renderData();
            });
        });
    }

    // CSV Download
    if (btnDownloadCsv) {
        btnDownloadCsv.addEventListener('click', () => {
            if (!state.selectedDate) {
                showNotification('Peringatan', 'Harap pilih tanggal terlebih dahulu.', 'error');
                return;
            }
            window.open(`/api/download/${state.selectedDate}/${state.currentModel}`, '_blank');
        });
    }


    // Run Pipeline V1 Trigger
    if (btnRunV1) {
        btnRunV1.addEventListener('click', () => {
            const targetDate = pipelineDateInput.value;
            if (!targetDate) {
                showNotification('Peringatan', 'Harap pilih tanggal dataset terlebih dahulu.', 'error');
                return;
            }
            runPipeline(targetDate, 'v1');
        });
    }

    // Run Pipeline V3 Trigger
    if (btnRunV3) {
        btnRunV3.addEventListener('click', () => {
            const targetDate = pipelineDateInput.value;
            if (!targetDate) {
                showNotification('Peringatan', 'Harap pilih tanggal dataset terlebih dahulu.', 'error');
                return;
            }
            runPipeline(targetDate, 'v3');
        });
    }

    // Table Search
    tableSearch.addEventListener('input', (e) => {
        state.searchQuery = e.target.value.trim().toLowerCase();
        renderTable();
    });

    // Table Filter Buttons
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentFilter = btn.getAttribute('data-filter');
            renderTable();
        });
    });

    // Table Sorting Headers
    tableHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const column = th.getAttribute('data-sort');
            if (state.sortColumn === column) {
                state.sortDirection = state.sortDirection === 'desc' ? 'asc' : 'desc';
            } else {
                state.sortColumn = column;
                state.sortDirection = 'desc'; // Default to highest first
            }
            
            // Update UI headers
            tableHeaders.forEach(h => {
                const icon = h.querySelector('i');
                icon.className = 'fa-solid fa-sort';
            });
            const activeIcon = th.querySelector('i');
            activeIcon.className = `fa-solid fa-sort-${state.sortDirection === 'desc' ? 'down' : 'up'}`;
            
            renderTable();
        });
    });

    // Lightbox Modal Zoom
    btnZoomImage.addEventListener('click', () => {
        const activeImg = visualImgContainer.querySelector('img');
        if (activeImg && activeImg.src) {
            imageModal.style.display = "block";
            modalImg.src = activeImg.src;
            modalCaption.textContent = `Visualisasi Momentum StockMixer ${state.currentModel.toUpperCase()} - ${state.selectedDate}`;
        }
    });

    
    // Zoom in on image direct click
    visualImgContainer.addEventListener('click', (e) => {
        if (e.target.tagName === 'IMG') {
            btnZoomImage.click();
        }
    });

    closeModal.addEventListener('click', () => {
        imageModal.style.display = "none";
    });

    // Close modal when clicking outside
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) {
            imageModal.style.display = "none";
        }
    });

    // --- API CALLS ---

    // Fetch History list of dates
    function fetchHistory(selectDate = null) {
        historyLoading.style.display = 'block';
        fetch('/api/history')
            .then(res => res.json())
            .then(dates => {
                state.history = dates;
                renderHistoryList();
                
                if (dates.length > 0) {
                    // Jika dilewati date spesifik, pilih date itu, jika tidak pilih yang terbaru
                    const dateToSelect = selectDate && dates.includes(selectDate) ? selectDate : dates[0];
                    selectHistoryDate(dateToSelect);
                } else {
                    selectedDateTitle.textContent = "Belum Ada Data";
                    selectedDateSubtitle.textContent = "Silakan jalankan pipeline untuk menghasilkan analisis pertama.";
                    tableBody.innerHTML = `<tr><td colspan="7" class="loading-row">Belum ada data visualisasi atau rekomendasi. Jalankan pipeline terlebih dahulu.</td></tr>`;
                    visualPlaceholder.style.display = 'block';
                    visualPlaceholder.innerHTML = '<i class="fa-solid fa-chart-line placeholder-icon"></i><p>Belum ada grafik visualisasi.</p>';
                }
                historyLoading.style.display = 'none';
            })
            .catch(err => {
                console.error("Gagal memuat history:", err);
                historyLoading.style.display = 'none';
                showNotification('Error', 'Gagal memuat riwayat analisis.', 'error');
            });
    }

    // Select and load specific date data
    function selectHistoryDate(date, retryCount = 0) {
        state.selectedDate = date;
        
        // Update active class in sidebar list
        const items = historyList.querySelectorAll('.history-item');
        items.forEach(item => {
            if (item.getAttribute('data-date') === date) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        selectedDateTitle.textContent = `Analisis Tanggal: ${date}`;
        selectedDateSubtitle.textContent = `Menampilkan rekomendasi model StockMixer untuk tanggal cutoff ${date}`;

        // Load dataset from backend
        tableBody.innerHTML = `<tr><td colspan="7" class="loading-row"><i class="fa-solid fa-spinner fa-spin"></i> Memuat data rekomendasi...</td></tr>`;
        
        fetch(`/api/outputs/${date}`)
            .then(res => {
                if (!res.ok) throw new Error("Data tidak ditemukan");
                return res.json();
            })
            .then(data => {
                // Check if data is empty/invalid (files still being written)
                if (!data.recommendations && !data.recommendations_v3) {
                    throw new Error("Data rekomendasi kosong");
                }
                state.currentData = data;
                renderData();
            })
            .catch(err => {
                console.error("Gagal memuat detail output:", err);
                
                // Retry jika masih dalam tahap penulis file (maksimal 3x dengan delay)
                if (retryCount < 3) {
                    setTimeout(() => {
                        selectHistoryDate(date, retryCount + 1);
                    }, 500);
                } else {
                    tableBody.innerHTML = `<tr><td colspan="7" class="loading-row text-danger">Gagal memuat data tanggal ${date}. Pastikan pipeline telah selesai dengan sukses.</td></tr>`;
                    showNotification('Error', `Gagal memuat data untuk tanggal ${date}.`, 'error');
                }
            });
    }

    // Check if pipeline is running currently (e.g. page refreshed)
    function checkActivePipeline() {
        fetch('/api/status')
            .then(res => res.json())
            .then(status => {
                if (status.is_running) {
                    state.isPipelineRunning = true;
                    updatePipelineUI(true, status.current_step, status.model_version || 'v1');
                    connectLogStream();
                }
            });
    }

    // Run Pipeline
    function runPipeline(date, model) {
        state.isPipelineRunning = true;
        updatePipelineUI(true, "Memulai pipeline...", model);
        consoleOutput.textContent = `Menghubungkan ke log pipeline ${model.toUpperCase()}...\n`;
        
        // Expand console
        consoleSection.classList.remove('collapsed');

        fetch('/api/run-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ date: date, model: model })
        })
        .then(res => res.json())
        .then(resData => {
            if (resData.error) {
                showNotification('Error', resData.error, 'error');
                updatePipelineUI(false);
                return;
            }
            showNotification('Sukses', `Pipeline ${model.toUpperCase()} untuk tanggal ${date} berhasil dipicu.`, 'info');
            connectLogStream(date);
        })
        .catch(err => {
            console.error("Gagal memicu pipeline:", err);
            showNotification('Error', 'Gagal memicu eksekusi pipeline.', 'error');
            updatePipelineUI(false);
        });
    }

    // Connect Server-Sent Events for live logs
    let eventSource = null;
    function connectLogStream(targetDate = null) {
        if (eventSource) {
            eventSource.close();
        }
        
        eventSource = new EventSource('/api/pipeline-logs');
        let hasError = false;
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.log) {
                // Tambahkan log ke konsol
                consoleOutput.textContent += data.log;
                // Auto scroll ke bawah
                consoleBody.scrollTop = consoleBody.scrollHeight;
                
                // Cari info langkah aktif dari log
                if (data.log.includes('[STEP:')) {
                    const stepName = data.log.match(/\[STEP:\s*([^\]]+)\]/)[1];
                    updatePipelineUI(true, stepName);
                }
                
                // Check for actual errors (not just ERROR mentions)
                if (data.log.includes('[ERROR]') || data.log.includes('EXCEPT:')) {
                    hasError = true;
                }
                
                if (data.finished) {
                    eventSource.close();
                    handlePipelineFinished(!hasError, targetDate);
                }
            }
            
            if (data.heartbeat) {
                // Heartbeat keep-alive, just continue
            }
        };

        eventSource.onerror = (err) => {
            console.error("SSE Error:", err);
            // Cek apakah pipeline sudah selesai dengan delay kecil
            setTimeout(() => {
                fetch('/api/status')
                    .then(res => res.json())
                    .then(status => {
                        if (!status.is_running) {
                            if (eventSource) eventSource.close();
                            handlePipelineFinished(status.success, targetDate || status.date);
                        }
                    });
            }, 500);
        };
    }

    function handlePipelineFinished(success, date) {
        state.isPipelineRunning = false;
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        
        updatePipelineUI(false);
        
        if (success) {
            showNotification('Selesai', `Pipeline selesai! Hasil tersimpan di outputs/${date}.`, 'success');
            // Add delay to ensure files are fully written to disk
            setTimeout(() => {
                // Refresh history list and auto select the new date
                fetchHistory(date);
            }, 1000);
        } else {
            showNotification('Gagal', `Pipeline gagal diproses. Silakan periksa konsol log.`, 'error');
        }
    }

    // --- UI UPDATERS & RENDERERS ---

    // Notification toast
    function showNotification(title, message, type = 'info') {
        const container = document.getElementById('notification-container');
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        let iconClass = 'fa-info-circle';
        if (type === 'success') iconClass = 'fa-check-circle';
        if (type === 'error') iconClass = 'fa-exclamation-circle';
        
        notification.innerHTML = `
            <i class="fa-solid ${iconClass} notification-icon"></i>
            <div>
                <strong style="display:block; font-size:14px; margin-bottom:2px;">${title}</strong>
                <span style="font-size:12px; opacity:0.8;">${message}</span>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Trigger reflow for animation
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Remove after 4 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 4000);
    }

    function updatePipelineUI(isRunning, stepName = 'Idle', model = 'v1') {
        if (btnRunV1) btnRunV1.disabled = isRunning;
        if (btnRunV3) btnRunV3.disabled = isRunning;
        
        if (isRunning) {
            pipelineBadge.textContent = "Running";
            pipelineBadge.className = "badge badge-running";
            currentStepText.textContent = `Langkah Aktif: ${stepName}`;
            if (model === 'v1' && btnRunV1) {
                btnRunV1.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running V1...`;
            } else if (model === 'v3' && btnRunV3) {
                btnRunV3.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running V3...`;
            }
        } else {
            // Cek status terakhir
            fetch('/api/status')
                .then(res => res.json())
                .then(status => {
                    if (status.success) {
                        pipelineBadge.textContent = "Success";
                        pipelineBadge.className = "badge badge-success";
                    } else if (status.error_message) {
                        pipelineBadge.textContent = "Failed";
                        pipelineBadge.className = "badge badge-error";
                    } else {
                        pipelineBadge.textContent = "Idle";
                        pipelineBadge.className = "badge badge-idle";
                    }
                    currentStepText.textContent = "";
                    if (btnRunV1) btnRunV1.innerHTML = `<i class="fa-solid fa-play"></i> Run Analisis Harian V1`;
                    if (btnRunV3) btnRunV3.innerHTML = `<i class="fa-solid fa-play"></i> Run Analisis 3 Harian V3`;
                });
        }
    }

    // Render History sidebar items
    function renderHistoryList() {
        historyList.innerHTML = '';
        state.history.forEach(date => {
            const li = document.createElement('li');
            li.className = `history-item ${date === state.selectedDate ? 'active' : ''}`;
            li.setAttribute('data-date', date);
            li.innerHTML = `
                <span><i class="fa-regular fa-calendar-check"></i> ${date}</span>
                <i class="fa-solid fa-chevron-right"></i>
            `;
            li.addEventListener('click', () => selectHistoryDate(date));
            historyList.appendChild(li);
        });
    }

    // Render main data (Image and Table)
    function renderData() {
        if (!state.currentData) return;
        
        const date = state.selectedDate;
        const model = state.currentModel;
        const hasVisual = model === 'v1' ? state.currentData.has_visual : state.currentData.has_visual_v3;
        
        // 1. Render Image Visual
        if (hasVisual) {
            visualPlaceholder.style.display = 'none';
            // Hindari caching gambar browser dengan menambahkan timestamp query param
            const imgSrc = `/api/visuals/${date}/${model}?t=${new Date().getTime()}`;
            
            const img = document.createElement('img');
            img.src = imgSrc;
            img.alt = `StockMixer ${model.toUpperCase()} Visual`;
            img.title = "Klik untuk memperbesar gambar";
            img.onerror = function() {
                this.style.display = 'none';
                visualPlaceholder.style.display = 'block';
                visualPlaceholder.innerHTML = `
                    <i class="fa-solid fa-triangle-exclamation placeholder-icon text-danger"></i>
                    <p>Gagal memuat gambar. File mungkin masih ditulis ke disk.</p>
                `;
            };
            
            visualImgContainer.innerHTML = '';
            visualImgContainer.appendChild(img);
            btnZoomImage.style.display = 'flex';
        } else {
            visualImgContainer.innerHTML = '';
            visualPlaceholder.style.display = 'block';
            visualPlaceholder.innerHTML = `
                <i class="fa-solid fa-triangle-exclamation placeholder-icon text-warning"></i>
                <p>Grafik visualisasi ${model.toUpperCase()} untuk tanggal ${date} tidak ditemukan.</p>
            `;
            btnZoomImage.style.display = 'none';
        }
        
        // 2. Render Table and Stats
        renderTable();
    }


    // Render stats cards & populate table rows
    function renderTable() {
        if (!state.currentData) return;
        
        const recList = state.currentModel === 'v1' ? state.currentData.recommendations : state.currentData.recommendations_v3;
        
        if (!recList || recList.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" class="loading-row text-warning">Tidak ada rekomendasi saham ${state.currentModel.toUpperCase()} yang lolos filter likuiditas pada tanggal ini.</td></tr>`;
            updateStatsCards(0, 0, 0, 0);
            return;
        }


        // Apply filters & search
        let filtered = recList.filter(item => {
            const matchSearch = item.Ticker.toLowerCase().includes(state.searchQuery);
            
            const isBullish = item.Harga > item.MA50;
            const matchFilter = 
                state.currentFilter === 'all' || 
                (state.currentFilter === 'bullish' && isBullish) ||
                (state.currentFilter === 'bearish' && !isBullish);
                
            return matchSearch && matchFilter;
        });

        // Apply Sorting
        filtered.sort((a, b) => {
            let valA = a[state.sortColumn];
            let valB = b[state.sortColumn];
            
            if (typeof valA === 'string') {
                return state.sortDirection === 'desc' 
                    ? valB.localeCompare(valA) 
                    : valA.localeCompare(valB);
            } else {
                return state.sortDirection === 'desc' 
                    ? valB - valA 
                    : valA - valB;
            }
        });

        // Compute Stats based on full recommendations list for this day
        const totalPicks = recList.length;
        const bullishPicks = recList.filter(item => item.Harga > item.MA50).length;
        const bearishPicks = totalPicks - bullishPicks;
        const maxScore = Math.max(...recList.map(item => item.AI_Score));
        
        updateStatsCards(totalPicks, bullishPicks, bearishPicks, maxScore);

        // Populate Table Body
        if (filtered.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" class="loading-row">Tidak ada saham yang cocok dengan filter pencarian Anda.</td></tr>`;
            return;
        }

        tableBody.innerHTML = '';
        filtered.forEach((item, index) => {
            const isBullish = item.Harga > item.MA50;
            const rank = index + 1;
            
            const scoreClass = item.AI_Score >= 0 ? 'score-positive' : 'score-negative';
            const trendText = isBullish ? '🚀 BULLISH' : '🩸 BEARISH';
            const trendClass = isBullish ? 'trend-bullish' : 'trend-bearish';
            
            // Format price to Rupiah style (without decimal unless needed)
            const formattedPrice = new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(item.Harga);
            const formattedMA50 = new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(item.MA50);
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="rank-cell">#${rank}</td>
                <td class="ticker-cell">${item.Ticker}</td>
                <td><span class="score-badge ${scoreClass}">${item.AI_Score.toFixed(4)}</span></td>
                <td>${formattedPrice}</td>
                <td>${formattedMA50}</td>
                <td>Rp ${item.Trx_Miliar.toFixed(2)} M</td>
                <td><span class="trend-badge ${trendClass}">${trendText}</span></td>
            `;
            tableBody.appendChild(tr);
        });
    }

    function updateStatsCards(total, bullish, bearish, topScore) {
        statTotalTickers.textContent = total;
        statBullishTickers.textContent = bullish;
        statBearishTickers.textContent = bearish;
        statTopScore.textContent = typeof topScore === 'number' && !isNaN(topScore) && topScore !== -Infinity 
            ? topScore.toFixed(4) 
            : '0.00';
    }
});
