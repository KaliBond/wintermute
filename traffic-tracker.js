// Simple traffic tracker for Neural Nations
// Privacy-friendly visitor counter with basic analytics

class TrafficTracker {
    constructor() {
        this.storageKey = 'neuralNationsTraffic';
        this.sessionKey = 'neuralNationsSession';
        this.init();
    }

    init() {
        this.trackVisit();
        this.displayCounter();
        this.trackPageViews();
    }

    trackVisit() {
        const now = new Date();
        const today = now.toDateString();
        const sessionId = this.generateSessionId();
        
        // Check if this is a new session
        const currentSession = sessionStorage.getItem(this.sessionKey);
        if (currentSession) {
            return; // Same session, don't count as new visit
        }
        
        // Mark new session
        sessionStorage.setItem(this.sessionKey, sessionId);
        
        // Get existing data
        let data = this.getData();
        
        // Update counters
        data.totalVisits++;
        data.dailyVisits[today] = (data.dailyVisits[today] || 0) + 1;
        data.lastVisit = now.toISOString();
        data.uniqueVisitors.add(this.getVisitorId());
        
        // Track page
        const page = window.location.pathname;
        data.pageViews[page] = (data.pageViews[page] || 0) + 1;
        
        // Save data
        this.saveData(data);
    }

    trackPageViews() {
        const page = window.location.pathname;
        let data = this.getData();
        data.pageViews[page] = (data.pageViews[page] || 0) + 1;
        this.saveData(data);
    }

    getData() {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                const data = JSON.parse(stored);
                // Convert Set back from Array for uniqueVisitors
                data.uniqueVisitors = new Set(data.uniqueVisitors || []);
                return data;
            }
        } catch (e) {
            console.warn('Error loading traffic data:', e);
        }
        
        return {
            totalVisits: 0,
            dailyVisits: {},
            pageViews: {},
            uniqueVisitors: new Set(),
            startDate: new Date().toISOString(),
            lastVisit: null
        };
    }

    saveData(data) {
        try {
            // Convert Set to Array for storage
            const toStore = {
                ...data,
                uniqueVisitors: Array.from(data.uniqueVisitors)
            };
            localStorage.setItem(this.storageKey, JSON.stringify(toStore));
        } catch (e) {
            console.warn('Error saving traffic data:', e);
        }
    }

    generateSessionId() {
        return Math.random().toString(36).substring(2) + Date.now().toString(36);
    }

    getVisitorId() {
        let visitorId = localStorage.getItem('neuralNationsVisitorId');
        if (!visitorId) {
            visitorId = this.generateSessionId();
            localStorage.setItem('neuralNationsVisitorId', visitorId);
        }
        return visitorId;
    }

    displayCounter() {
        const data = this.getData();
        const today = new Date().toDateString();
        const todayVisits = data.dailyVisits[today] || 0;
        
        // Create counter display
        this.createCounterWidget(data.totalVisits, todayVisits, data.uniqueVisitors.size);
    }

    createCounterWidget(total, today, unique) {
        // Remove existing counter if present
        const existing = document.getElementById('traffic-counter');
        if (existing) existing.remove();

        // Create counter widget
        const counter = document.createElement('div');
        counter.id = 'traffic-counter';
        counter.innerHTML = `
            <div class="traffic-widget">
                <div class="traffic-stats">
                    <div class="stat-item">
                        <span class="stat-number">${total.toLocaleString()}</span>
                        <span class="stat-label">Total Visits</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${today.toLocaleString()}</span>
                        <span class="stat-label">Today</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${unique.toLocaleString()}</span>
                        <span class="stat-label">Unique Visitors</span>
                    </div>
                </div>
                <div class="traffic-toggle" onclick="trafficTracker.toggleDetails()">📊</div>
            </div>
            <div id="traffic-details" class="traffic-details" style="display: none;">
                <h4>Site Analytics</h4>
                <div id="traffic-charts"></div>
                <div class="analytics-info">
                    <p><strong>Privacy Note:</strong> All data stored locally in your browser. No external tracking.</p>
                    <button onclick="trafficTracker.exportData()" class="export-btn">Export Data</button>
                    <button onclick="trafficTracker.clearData()" class="clear-btn">Clear Data</button>
                </div>
            </div>
        `;

        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .traffic-widget {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                min-width: 200px;
                backdrop-filter: blur(10px);
            }
            .traffic-stats {
                display: flex;
                gap: 15px;
                margin-bottom: 10px;
            }
            .stat-item {
                text-align: center;
                flex: 1;
            }
            .stat-number {
                display: block;
                font-size: 1.2em;
                font-weight: bold;
                color: #fff;
            }
            .stat-label {
                display: block;
                font-size: 0.8em;
                opacity: 0.9;
                margin-top: 2px;
            }
            .traffic-toggle {
                text-align: center;
                cursor: pointer;
                font-size: 1.2em;
                opacity: 0.8;
                transition: opacity 0.3s;
            }
            .traffic-toggle:hover {
                opacity: 1;
            }
            .traffic-details {
                position: fixed;
                bottom: 140px;
                right: 20px;
                background: white;
                color: #333;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                z-index: 1001;
                width: 300px;
                max-height: 400px;
                overflow-y: auto;
            }
            .traffic-details h4 {
                margin: 0 0 15px 0;
                color: #2c3e50;
            }
            .analytics-info {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #eee;
            }
            .analytics-info p {
                font-size: 0.9em;
                color: #666;
                margin: 10px 0;
            }
            .export-btn, .clear-btn {
                background: #3498db;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
                font-size: 0.9em;
            }
            .clear-btn {
                background: #e74c3c;
            }
            .export-btn:hover, .clear-btn:hover {
                opacity: 0.9;
            }
            @media (max-width: 768px) {
                .traffic-widget {
                    bottom: 10px;
                    right: 10px;
                    padding: 10px;
                    min-width: 150px;
                }
                .traffic-stats {
                    flex-direction: column;
                    gap: 8px;
                }
                .traffic-details {
                    width: 250px;
                    right: 10px;
                }
            }
        `;
        document.head.appendChild(style);

        // Add to page
        document.body.appendChild(counter);
        
        // Add details charts
        setTimeout(() => this.createAnalyticsCharts(), 100);
    }

    createAnalyticsCharts() {
        const data = this.getData();
        const chartsContainer = document.getElementById('traffic-charts');
        if (!chartsContainer) return;

        // Daily visits chart (last 7 days)
        const last7Days = this.getLast7Days();
        const dailyData = last7Days.map(date => ({
            date: date,
            visits: data.dailyVisits[date] || 0
        }));

        // Top pages
        const topPages = Object.entries(data.pageViews)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);

        chartsContainer.innerHTML = `
            <div class="chart-section">
                <h5>Last 7 Days</h5>
                <div class="mini-chart">
                    ${dailyData.map(d => `
                        <div class="chart-bar" style="height: ${Math.max(5, (d.visits / Math.max(...dailyData.map(x => x.visits), 1)) * 40)}px;">
                            <span class="chart-label">${d.date.split(' ')[1]?.substring(0,3) || 'Today'}</span>
                            <span class="chart-value">${d.visits}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div class="chart-section">
                <h5>Top Pages</h5>
                <div class="page-list">
                    ${topPages.map(([page, views]) => `
                        <div class="page-item">
                            <span class="page-name">${page === '/' ? 'Home' : page.replace(/^\//, '').replace('.html', '')}</span>
                            <span class="page-views">${views}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        // Add chart styles
        const chartStyle = document.createElement('style');
        chartStyle.textContent = `
            .chart-section {
                margin: 10px 0;
            }
            .chart-section h5 {
                margin: 0 0 8px 0;
                font-size: 0.9em;
                color: #2c3e50;
            }
            .mini-chart {
                display: flex;
                align-items: end;
                gap: 3px;
                height: 50px;
                margin: 10px 0;
            }
            .chart-bar {
                flex: 1;
                background: linear-gradient(to top, #3498db, #667eea);
                border-radius: 2px;
                position: relative;
                min-height: 5px;
                cursor: pointer;
            }
            .chart-bar:hover .chart-value {
                opacity: 1;
            }
            .chart-label {
                position: absolute;
                bottom: -15px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.7em;
                color: #666;
            }
            .chart-value {
                position: absolute;
                top: -20px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.7em;
                background: #333;
                color: white;
                padding: 2px 4px;
                border-radius: 3px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .page-list {
                max-height: 120px;
                overflow-y: auto;
            }
            .page-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 5px 0;
                border-bottom: 1px solid #f0f0f0;
                font-size: 0.9em;
            }
            .page-item:last-child {
                border-bottom: none;
            }
            .page-name {
                flex: 1;
                text-transform: capitalize;
            }
            .page-views {
                background: #ecf0f1;
                color: #2c3e50;
                padding: 2px 6px;
                border-radius: 10px;
                font-size: 0.8em;
            }
        `;
        document.head.appendChild(chartStyle);
    }

    getLast7Days() {
        const days = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            days.push(date.toDateString());
        }
        return days;
    }

    toggleDetails() {
        const details = document.getElementById('traffic-details');
        if (details) {
            details.style.display = details.style.display === 'none' ? 'block' : 'none';
        }
    }

    exportData() {
        const data = this.getData();
        const exportData = {
            ...data,
            uniqueVisitors: Array.from(data.uniqueVisitors),
            exportDate: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `neural-nations-analytics-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    clearData() {
        if (confirm('Are you sure you want to clear all analytics data? This cannot be undone.')) {
            localStorage.removeItem(this.storageKey);
            localStorage.removeItem('neuralNationsVisitorId');
            sessionStorage.removeItem(this.sessionKey);
            location.reload();
        }
    }

    // Public API for manual tracking
    trackEvent(event, data = {}) {
        const trackingData = this.getData();
        if (!trackingData.events) trackingData.events = {};
        
        const eventKey = `${event}_${new Date().toDateString()}`;
        trackingData.events[eventKey] = (trackingData.events[eventKey] || 0) + 1;
        
        this.saveData(trackingData);
        console.log(`Event tracked: ${event}`, data);
    }
}

// Initialize tracker when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.trafficTracker = new TrafficTracker();
    });
} else {
    window.trafficTracker = new TrafficTracker();
}

// Export for manual usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrafficTracker;
}