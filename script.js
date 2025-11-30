// ============================================================================
// NBA TEAMS DATA
// ============================================================================
const NBA_TEAMS = [
    // Atlantic Division
    { name: "Boston Celtics", abbr: "BOS", division: "Atlantic", rating: 85 },
    { name: "Brooklyn Nets", abbr: "BKN", division: "Atlantic", rating: 72 },
    { name: "New York Knicks", abbr: "NYK", division: "Atlantic", rating: 78 },
    { name: "Philadelphia 76ers", abbr: "PHI", division: "Atlantic", rating: 82 },
    { name: "Toronto Raptors", abbr: "TOR", division: "Atlantic", rating: 70 },
    
    // Central Division
    { name: "Chicago Bulls", abbr: "CHI", division: "Central", rating: 73 },
    { name: "Cleveland Cavaliers", abbr: "CLE", division: "Central", rating: 80 },
    { name: "Detroit Pistons", abbr: "DET", division: "Central", rating: 65 },
    { name: "Indiana Pacers", abbr: "IND", division: "Central", rating: 75 },
    { name: "Milwaukee Bucks", abbr: "MIL", division: "Central", rating: 84 },
    
    // Southeast Division
    { name: "Atlanta Hawks", abbr: "ATL", division: "Southeast", rating: 74 },
    { name: "Charlotte Hornets", abbr: "CHA", division: "Southeast", rating: 68 },
    { name: "Miami Heat", abbr: "MIA", division: "Southeast", rating: 79 },
    { name: "Orlando Magic", abbr: "ORL", division: "Southeast", rating: 76 },
    { name: "Washington Wizards", abbr: "WAS", division: "Southeast", rating: 66 },
    
    // Northwest Division
    { name: "Denver Nuggets", abbr: "DEN", division: "Northwest", rating: 88 },
    { name: "Minnesota Timberwolves", abbr: "MIN", division: "Northwest", rating: 77 },
    { name: "Oklahoma City Thunder", abbr: "OKC", division: "Northwest", rating: 81 },
    { name: "Portland Trail Blazers", abbr: "POR", division: "Northwest", rating: 69 },
    { name: "Utah Jazz", abbr: "UTA", division: "Northwest", rating: 67 },
    
    // Pacific Division
    { name: "Golden State Warriors", abbr: "GSW", division: "Pacific", rating: 83 },
    { name: "LA Clippers", abbr: "LAC", division: "Pacific", rating: 80 },
    { name: "Los Angeles Lakers", abbr: "LAL", division: "Pacific", rating: 81 },
    { name: "Phoenix Suns", abbr: "PHX", division: "Pacific", rating: 82 },
    { name: "Sacramento Kings", abbr: "SAC", division: "Pacific", rating: 76 },
    
    // Southwest Division
    { name: "Dallas Mavericks", abbr: "DAL", division: "Southwest", rating: 83 },
    { name: "Houston Rockets", abbr: "HOU", division: "Southwest", rating: 71 },
    { name: "Memphis Grizzlies", abbr: "MEM", division: "Southwest", rating: 75 },
    { name: "New Orleans Pelicans", abbr: "NOP", division: "Southwest", rating: 74 },
    { name: "San Antonio Spurs", abbr: "SAS", division: "Southwest", rating: 64 }
];

// ============================================================================
// DOM ELEMENTS
// ============================================================================
const team1Select = document.getElementById('team1');
const team2Select = document.getElementById('team2');
const predictionForm = document.getElementById('predictionForm');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resultsSection = document.getElementById('resultsSection');

// Results elements
const team1Name = document.getElementById('team1Name');
const team2Name = document.getElementById('team2Name');
const team1Badge = document.getElementById('team1Badge');
const team2Badge = document.getElementById('team2Badge');
const team1Probability = document.getElementById('team1Probability');
const team2Probability = document.getElementById('team2Probability');
const team1Bar = document.getElementById('team1Bar');
const team2Bar = document.getElementById('team2Bar');
const team1Result = document.getElementById('team1Result');
const team2Result = document.getElementById('team2Result');
const team1Stats = document.getElementById('team1Stats');
const team2Stats = document.getElementById('team2Stats');
const winnerName = document.getElementById('winnerName');
const confidenceValue = document.getElementById('confidenceValue');

// ============================================================================
// INITIALIZE APP
// ============================================================================
function initializeApp() {
    populateTeamSelects();
    setupEventListeners();
    setupCarousel();
}

// ============================================================================
// POPULATE TEAM DROPDOWNS
// ============================================================================
function populateTeamSelects() {
    // Sort teams alphabetically
    const sortedTeams = [...NBA_TEAMS].sort((a, b) => a.name.localeCompare(b.name));
    
    sortedTeams.forEach(team => {
        const option1 = document.createElement('option');
        option1.value = team.abbr;
        option1.textContent = team.name;
        team1Select.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = team.abbr;
        option2.textContent = team.name;
        team2Select.appendChild(option2);
    });
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function setupEventListeners() {
    predictionForm.addEventListener('submit', handlePrediction);
    
    // Hide error when user changes selections
    [team1Select, team2Select].forEach(select => {
        select.addEventListener('change', hideError);
    });
}

// ============================================================================
// HANDLE PREDICTION
// ============================================================================
function handlePrediction(e) {
    e.preventDefault();
    hideError();
    
    // Get form values
    const team1Abbr = team1Select.value;
    const team2Abbr = team2Select.value;
    const homeTeamValue = document.querySelector('input[name="homeTeam"]:checked')?.value;
    
    // Validation
    if (!team1Abbr || !team2Abbr) {
        showError('Please select both teams');
        return;
    }
    
    if (team1Abbr === team2Abbr) {
        showError('Please select two different teams');
        return;
    }
    
    if (!homeTeamValue) {
        showError('Please select which team is playing at home');
        return;
    }
    
    // Get team data
    const team1Data = NBA_TEAMS.find(t => t.abbr === team1Abbr);
    const team2Data = NBA_TEAMS.find(t => t.abbr === team2Abbr);
    
    // Determine home teams
    const team1IsHome = homeTeamValue === 'team1';
    const team2IsHome = homeTeamValue === 'team2';
    
    // Calculate prediction
    const prediction = predictGame(team1Data, team2Data, team1IsHome, team2IsHome);
    
    // Display results
    displayResults(prediction, team1Data, team2Data, team1IsHome, team2IsHome);
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// ============================================================================
// PREDICTION ALGORITHM
// ============================================================================
function predictGame(team1, team2, team1IsHome, team2IsHome) {
    // Base probabilities from team ratings (0-100 scale)
    let team1Score = team1.rating;
    let team2Score = team2.rating;
    
    // Home court advantage (+5 points)
    const HOME_ADVANTAGE = 5;
    if (team1IsHome) team1Score += HOME_ADVANTAGE;
    if (team2IsHome) team2Score += HOME_ADVANTAGE;
    
    // Add some randomness to simulate game variability (±3 points)
    const randomFactor = () => (Math.random() - 0.5) * 6;
    team1Score += randomFactor();
    team2Score += randomFactor();
    
    // Convert scores to probabilities (normalize to 100%)
    const totalScore = team1Score + team2Score;
    const team1WinProb = (team1Score / totalScore) * 100;
    const team2WinProb = (team2Score / totalScore) * 100;
    
    // Generate mock stats
    const team1StatsData = generateTeamStats(team1, team1IsHome);
    const team2StatsData = generateTeamStats(team2, team2IsHome);
    
    return {
        team1WinProb: Math.round(team1WinProb * 10) / 10,
        team2WinProb: Math.round(team2WinProb * 10) / 10,
        team1Stats: team1StatsData,
        team2Stats: team2StatsData,
        winner: team1WinProb > team2WinProb ? team1.name : team2.name,
        confidence: Math.max(team1WinProb, team2WinProb)
    };
}

// ============================================================================
// GENERATE TEAM STATS
// ============================================================================
function generateTeamStats(team, isHome) {
    // Base stats influenced by team rating
    const ratingFactor = team.rating / 100;
    const homeFactor = isHome ? 1.05 : 0.98;
    
    return {
        avgPoints: Math.round((105 + (team.rating - 70) * 0.8) * homeFactor * 10) / 10,
        avgRebounds: Math.round((42 + (team.rating - 70) * 0.3) * homeFactor * 10) / 10,
        avgAssists: Math.round((24 + (team.rating - 70) * 0.2) * homeFactor * 10) / 10,
        winStreak: Math.floor(Math.random() * 5)
    };
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================
function displayResults(prediction, team1, team2, team1IsHome, team2IsHome) {
    // Set team names
    team1Name.textContent = team1.name;
    team2Name.textContent = team2.name;
    
    // Set home/away badges
    team1Badge.textContent = team1IsHome ? 'HOME' : 'AWAY';
    team1Badge.className = `team-badge ${team1IsHome ? 'home' : 'away'}`;
    
    team2Badge.textContent = team2IsHome ? 'HOME' : 'AWAY';
    team2Badge.className = `team-badge ${team2IsHome ? 'home' : 'away'}`;
    
    // Set probabilities
    team1Probability.textContent = `${prediction.team1WinProb}%`;
    team2Probability.textContent = `${prediction.team2WinProb}%`;
    
    // Animate probability bars
    setTimeout(() => {
        team1Bar.style.width = `${prediction.team1WinProb}%`;
        team2Bar.style.width = `${prediction.team2WinProb}%`;
    }, 100);
    
    // Set winner styling
    team1Result.classList.remove('winner');
    team2Result.classList.remove('winner');
    
    if (prediction.winner === team1.name) {
        team1Result.classList.add('winner');
    } else {
        team2Result.classList.add('winner');
    }
    
    // Display stats
    team1Stats.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${prediction.team1Stats.avgPoints}</div>
            <div class="stat-label">Avg Points</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${prediction.team1Stats.avgRebounds}</div>
            <div class="stat-label">Avg Rebounds</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${prediction.team1Stats.avgAssists}</div>
            <div class="stat-label">Avg Assists</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${prediction.team1Stats.winStreak}</div>
            <div class="stat-label">Win Streak</div>
        </div>
    `;
    
    team2Stats.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${prediction.team2Stats.avgPoints}</div>
            <div class="stat-label">Avg Points</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${prediction.team2Stats.avgRebounds}</div>
            <div class="stat-label">Avg Rebounds</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${prediction.team2Stats.avgAssists}</div>
            <div class="stat-label">Avg Assists</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${prediction.team2Stats.winStreak}</div>
            <div class="stat-label">Win Streak</div>
        </div>
    `;
    
    // Set winner banner
    winnerName.textContent = prediction.winner;
    confidenceValue.textContent = `${Math.round(prediction.confidence)}%`;
    
    // Show results section
    resultsSection.classList.remove('hidden');
}

// ============================================================================
// ERROR HANDLING
// ============================================================================
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// ============================================================================
// IMAGE CAROUSEL
// ============================================================================
function setupCarousel() {
    const dots = document.querySelectorAll('.dot');
    const items = document.querySelectorAll('.viz-item');
    
    dots.forEach(dot => {
        dot.addEventListener('click', () => {
            const index = parseInt(dot.dataset.index);
            
            // Update active states
            dots.forEach(d => d.classList.remove('active'));
            items.forEach(item => item.classList.remove('active'));
            
            dot.classList.add('active');
            items[index].classList.add('active');
        });
    });
    
    // Auto-advance carousel every 5 seconds
    let currentIndex = 0;
    setInterval(() => {
        currentIndex = (currentIndex + 1) % items.length;
        dots[currentIndex].click();
    }, 5000);
}

// ============================================================================
// INITIALIZE ON PAGE LOAD
// ============================================================================
document.addEventListener('DOMContentLoaded', initializeApp);
