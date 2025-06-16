// Chess AI Web Interface - JavaScript Controller
// Version 2.0 - Refactored for Stability and Correctness

let game = new Chess();
let board = null;
let isEngineThinking = false;
let isAutoPlaying = false;
let neuralNetReady = false;
let stockfishReady = false;
let currentMoveTimeout = null;
let gameStartTime = null;
let gameTimer = null;
let gameMode = 'human-vs-neural'; // Initialize with a default
let whiteElo = 1500;
let blackElo = 1500;
const K_FACTOR = 32;

const API_TIMEOUT_MS = 60000; // 60-second timeout; HopeChess may need longer

// --- Initialization ---
$(document).ready(function() {
    console.log('Chess Interface Initializing...');
    initializeChessBoard();
    setupEventListeners();
    checkSystemStatus(); // Initial status check
    newGame(); // Start with a fresh game
});

function initializeChessBoard() {
    const config = {
        draggable: true,
        position: 'start',
        pieceTheme: '/static/img/chesspieces/wikipedia/{piece}.png',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
    };
    board = Chessboard('chess-board', config);
    $(window).on('resize', () => board.resize());
    $('#chess-board').addClass('theme-metal');
    console.log('✅ Chessboard created.');
}

function setupEventListeners() {
    $('#play-mode').on('change', () => {
        gameMode = $('#play-mode').val();
        console.log(`🔄 Game mode changed to: ${gameMode}`);
        updatePlayerIndicators();
    });
    $('#white-engine-strength, #black-engine-strength').on('change', updatePlayerIndicators);
    $('#new-game-btn').on('click', newGame);
    $('#auto-play-btn').on('click', toggleAutoPlay);
    $('#takeback-btn').on('click', takebackMove);
    $('#resign-btn').on('click', resignGame);
    $('#draw-btn').on('click', offerDraw);
    console.log('✅ Event listeners configured.');
}

// --- Game Controls ---
function newGame() {
    console.log('🎮 Starting new game...');
    if (isAutoPlaying) {
        stopAutoPlay();
    }
    clearTimeout(currentMoveTimeout);
    clearInterval(gameTimer);
    
    game = new Chess();
    board.start();
    isEngineThinking = false;
    gameStartTime = null;

    removeHighlights();
    updateGameStatus();
    updateMaterialCount();
    updateGameInfo();
    debounceEvaluation();
    updatePlayerIndicators();
    $('#game-time').text('0:00');
    // Hide previous game result row & overlay
    $('#game-result-row').hide();
    $('#game-result-value').text('-');
    $('#game-result-display').hide();
    whiteElo = 1500;
    blackElo = 1500;
    $('#elo-estimation').hide();
    console.log('✅ New game started.');
}

function toggleAutoPlay() {
    if (isAutoPlaying) {
        stopAutoPlay();
    } else {
        startAutoPlay();
    }
}

function startAutoPlay() {
    if (isAutoPlaying) return;

    const [whiteEngine, blackEngine] = parseGameModeEngines();
    if (whiteEngine === 'human' || blackEngine === 'human') {
        showNotification("Auto-play is only for engine vs. engine modes.", "warning");
        return;
    }
    if ((whiteEngine === 'neural' || blackEngine === 'neural') && !neuralNetReady) {
        showNotification("Cannot start auto-play: Neural Network model is not loaded.", "error");
        return;
    }
    
    isAutoPlaying = true;
    updateAutoPlayButton();
    console.log('🚀 Starting auto-play...');
    makeEngineMove();
}

function stopAutoPlay() {
    isAutoPlaying = false;
    clearTimeout(currentMoveTimeout);
    updateAutoPlayButton();
    console.log('⏹️ Auto-play stopped.');
}

// --- Core Move Logic ---
function onDragStart(source, piece) {
    if (game.game_over() || isEngineThinking) return false;
    const engineForTurn = parseGameModeEngines()[game.turn() === 'w' ? 0 : 1];
    if (engineForTurn !== 'human') return false;
    if ((game.turn() === 'b' && piece.search(/^w/) !== -1) || (game.turn() === 'w' && piece.search(/^b/) !== -1)) {
        return false;
    }
}

function onDrop(source, target) {
    const move = game.move({ from: source, to: target, promotion: 'q' });
    if (move === null) return 'snapback';
    
    console.log('✅ Human move:', move.san);
    highlightMove(source, target);
    processMove();
}

function onSnapEnd() {
    board.position(game.fen());
}

async function makeEngineMove() {
    const [whiteEngine, blackEngine] = parseGameModeEngines();
    const engineForTurn = (game.turn() === 'w') ? whiteEngine : blackEngine;

    if (game.game_over() || isEngineThinking || engineForTurn === 'human') {
        if (isAutoPlaying) stopAutoPlay();
        return;
    }

    isEngineThinking = true;
    updateEngineStatus('thinking', engineForTurn);

    try {
        let moveData = null;
        if (engineForTurn === 'neural') {
            moveData = await getNeuralMove();
        } else if (engineForTurn === 'leela') {
            moveData = await getLeelaMove();
        } else if (engineForTurn === 'hopechess') {
            moveData = await apiCall('/api/move/hopechess', { method: 'POST', body: { fen: game.fen(), strength: (game.turn()==='w' ? $('#white-engine-strength').val() : $('#black-engine-strength').val()) } });
        } else if (engineForTurn === 'llm') {
            moveData = await apiCall('/api/move/llm', { method: 'POST', body: { fen: game.fen() }, timeout: 0 }); // no timeout
        } else {
            moveData = await getStockfishMove(game.turn());
        }

        if (moveData && moveData.move) {
            console.log(`🎯 ${engineForTurn} move received:`, moveData.move);

            // Create a move object from the UCI string for chess.js
            const from = moveData.move.substring(0, 2);
            const to = moveData.move.substring(2, 4);
            const promotion = moveData.move.length > 4 ? moveData.move.substring(4, 5).toLowerCase() : undefined;

            const moveResult = game.move({
                from: from,
                to: to,
                promotion: promotion
            });

            // Check if the move was illegal
            if (moveResult === null) {
                console.error(`Engine proposed an illegal move or it was parsed incorrectly: ${moveData.move}`);
                showNotification(`Illegal move received: ${moveData.move}`, 'error');
                if (isAutoPlaying) stopAutoPlay();
                return; // Stop processing this turn
            }

            highlightMove(from, to);
            processMove();
        } else {
            // Handle case where engine returns no move
            showNotification(`${engineForTurn} did not return a move.`, 'warning');
            if (isAutoPlaying) stopAutoPlay();
        }
    } catch (error) {
        console.error(`❌ Engine move error for ${engineForTurn}:`, error);
        showNotification(`Error: ${error.message || 'Engine failure'}`, 'danger');
        if (isAutoPlaying) stopAutoPlay();
        // Declare engine error as resignation
        const engineName = getEngineDisplayName(engineForTurn);
        declareEngineErrorLoser(engineName);
    } finally {
        isEngineThinking = false;
        updateEngineStatus('ready', engineForTurn);

        if (isAutoPlaying && !game.game_over()) {
            currentMoveTimeout = setTimeout(makeEngineMove, 500); // Delay between auto-play moves
        } else if (isAutoPlaying) {
            stopAutoPlay();
        }
    }
}

function processMove() {
    board.position(game.fen());
    updateGameStatus();
    updateMaterialCount();
    updateGameInfo();
    debounceEvaluation();
    
    // Show game result once over
    if (game.game_over()) {
        let resultText = 'Draw';
        if (game.in_checkmate()) {
            resultText = game.turn() === 'w' ? 'Black wins by checkmate' : 'White wins by checkmate';
        } else if (game.in_stalemate()) {
            resultText = 'Stalemate';
        } else if (game.in_threefold_repetition()) {
            resultText = 'Draw by repetition';
        } else if (game.insufficient_material()) {
            resultText = 'Draw (insufficient material)';
        }

        // Update evaluation panel row
        $('#game-result-value').text(resultText);
        $('#game-result-row').show();

        // Ensure any old overlay is hidden
        $('#game-result-display').hide();

        // -------- Elo update ----------
        let scoreWhite;
        if (game.in_checkmate()) {
            scoreWhite = game.turn() === 'w' ? 0 : 1; // side to move is loser
        } else {
            scoreWhite = 0.5; // draw variants
        }
        const expectedWhite = 1 / (1 + Math.pow(10, (blackElo - whiteElo) / 400));
        const expectedBlack = 1 - expectedWhite;
        whiteElo = whiteElo + K_FACTOR * (scoreWhite - expectedWhite);
        blackElo = blackElo + K_FACTOR * ((1 - scoreWhite) - expectedBlack);

        $('#white-elo-estimate').text(Math.round(whiteElo));
        $('#black-elo-estimate').text(Math.round(blackElo));
        $('#elo-estimation').show();
    }

    if (!game.game_over()) {
        if (!isAutoPlaying) {
            // If not auto-playing, schedule the opponent's engine move
            const [whiteEngine, blackEngine] = parseGameModeEngines();
            const engineForTurn = (game.turn() === 'w') ? whiteEngine : blackEngine;
            if (engineForTurn !== 'human') {
                 currentMoveTimeout = setTimeout(makeEngineMove, 300);
            }
        }
    }
}

// --- API Communication ---
async function apiCall(endpoint, options = {}) {
    const controller = new AbortController();
    const customTimeout = (options.timeout !== undefined) ? options.timeout : API_TIMEOUT_MS;
    const timeoutId = customTimeout === 0 ? null : setTimeout(() => controller.abort(), customTimeout);
    
    try {
        const fetchOptions = { ...options, signal: controller.signal, headers: { 'Content-Type': 'application/json', ...options.headers } };
        if (options.body) fetchOptions.body = JSON.stringify(options.body);

        const response = await fetch(endpoint, fetchOptions);
        const responseData = await response.json();
        if (!response.ok) throw new Error(responseData.error || `HTTP ${response.status}`);
        return responseData;
    } catch (error) {
        if (timeoutId) clearTimeout(timeoutId);
        if (error.name === 'AbortError') throw new Error('Request timeout');
        throw error;
    }
}

async function getNeuralMove() {
    return await apiCall('/api/move/ai', { method: 'POST', body: { fen: game.fen() } });
}

async function getLeelaMove() {
    showNotification("LeelaZero/LC0 engine is not yet implemented.", "warning");
    return Promise.resolve(null);
}

async function getStockfishMove(color) {
    const strengthSelector = color === 'w' ? '#white-engine-strength' : '#black-engine-strength';
    const strength = $(strengthSelector).val();
    return await apiCall('/api/move/stockfish', { method: 'POST', body: { fen: game.fen(), strength } });
}

// --- UI Update Functions ---
function updateGameStatus() {
    let statusText;
    let turnColor = game.turn() === 'w' ? 'White' : 'Black';
    
    if (game.in_checkmate()) {
        const winner = turnColor === 'White' ? 'Black' : 'White';
        statusText = `Checkmate! ${winner} wins.`;
    } else if (game.in_draw()) {
        let reason = '';
        if (game.in_stalemate()) reason = 'by Stalemate';
        else if (game.in_threefold_repetition()) reason = 'by Threefold Repetition';
        else if (game.insufficient_material()) reason = 'by Insufficient Material';
        // chess.js does not expose fivefold repetition separately; threefold detection is sufficient.
        else reason = 'Draw';
        statusText = `Draw ${reason}.`;
    } else {
        statusText = `${turnColor} to move`;
        if (game.in_check()) {
            statusText += ' (in check)';
        }
    }
    $('#game-status').text(statusText);

    if (game.history().length === 1 && !gameTimer) {
        gameStartTime = Date.now();
        gameTimer = setInterval(updateGameInfo, 1000);
    }
    if (game.game_over()) {
            clearInterval(gameTimer);
        if (isAutoPlaying) stopAutoPlay();
    }
}

function updateGameInfo() {
    if (gameStartTime) {
        const elapsedMs = Date.now() - gameStartTime;
        const minutes = Math.floor(elapsedMs / 60000);
        const seconds = Math.floor((elapsedMs % 60000) / 1000);
        $('#game-time').text(`${minutes}:${seconds.toString().padStart(2, '0')}`);
    }
    $('#total-moves').text(game.history().length);
    $('#legal-moves-count').text(game.moves().length);
}

function updateAutoPlayButton() {
    const button = $('#auto-play-btn');
    if (isAutoPlaying) {
        button.html('<i class="fas fa-stop"></i> Stop Auto-Play').removeClass('btn-info').addClass('btn-warning');
    } else {
        button.html('<i class="fas fa-play-circle"></i> Start Auto-Play').removeClass('btn-warning').addClass('btn-info');
    }
}

function updatePlayerIndicators() {
    const [whiteEngine, blackEngine] = parseGameModeEngines();
    const whiteStrength = $('#white-engine-strength option:selected').text();
    const blackStrength = $('#black-engine-strength option:selected').text();
    
    const updateIndicator = (side, engine, strength) => {
        const indicator = $(`#${side}-player-indicator`);
        indicator.find('.player-name').text(getEngineDisplayName(engine));
        indicator.find('.player-strength').text(engine === 'human' ? '' : strength);
    };

    updateIndicator('white', whiteEngine, whiteStrength);
    updateIndicator('black', blackEngine, blackStrength);

    // Update evaluation labels
    $('#white-eval-label').text(`${getEngineDisplayName(whiteEngine)}:`);
    $('#black-eval-label').text(`${getEngineDisplayName(blackEngine)}:`);
}

let evaluationDebounceTimer;
function debounceEvaluation() {
    clearTimeout(evaluationDebounceTimer);
    evaluationDebounceTimer = setTimeout(() => {
        if (!game.game_over()) evaluatePosition();
    }, 350);
}

async function evaluatePosition() {
    if (isEngineThinking) return;
    try {
        const data = await apiCall('/api/evaluate', { method: 'POST', body: { fen: game.fen() } });
        const evaluation = data.evaluation;

        // evaluation is centipawns from white's perspective, can be null
        if (evaluation === null || evaluation === undefined) {
            // Reset bar and numeric displays
            $('#evaluation-fill').css('width', '50%');
            $('#white-eval').text('N/A');
            $('#black-eval').text('N/A');
            return;
        }
        
        const cpScore = parseFloat(evaluation); // centipawns (white perspective)

        // Logistic scaling to 0-100% bar filled towards white
        const barPercent = 50 * (1 + (2 / (1 + Math.exp(-0.004 * cpScore)) - 1));
        $('#evaluation-fill').css('width', `${barPercent}%`);

        // Numeric displays for both sides
        $('#white-eval').text(`${cpScore} cp`);
        $('#black-eval').text(`${-cpScore} cp`);

    } catch (error) {
        console.error('Evaluation error:', error);
        $('#white-eval').text('Err');
        $('#black-eval').text('Err');
    }
}

async function checkSystemStatus() {
    try {
        const data = await apiCall('/api/status');
        neuralNetReady = data.model_loaded;
        stockfishReady = data.stockfish_available;

        const updateStatusIndicator = (name, isReady) => {
            const el = $(`.status-indicator:contains("${name}")`);
            const icon = name === 'Neural Network' ? 'fa-brain' : 'fa-chess-rook';
            el.removeClass('status-online status-offline status-loading')
              .addClass(isReady ? 'status-online' : 'status-offline')
              .html(`<i class="fas ${icon}"></i> ${name}: ${isReady ? 'Ready' : 'Unavailable'}`);
        };
        updateStatusIndicator('Neural Network', neuralNetReady);
        updateStatusIndicator('Stockfish', stockfishReady);
        
        $('#device-info').text(data.device || 'CPU');
        
        // Disable relevant dropdown options if engines not ready
        $('#play-mode option[value*="neural"]').prop('disabled', !neuralNetReady);
        $('#play-mode option[value*="stockfish"]').prop('disabled', !stockfishReady);
        $('#play-mode option[value*="leela"]').prop('disabled', true); // Always disabled for now
        $('#play-mode option[value*="hopechess"]').prop('disabled', true); // HopeChess temporarily disabled

        // LLM enabled only if backend has config
        const llmCfg = await apiCall('/api/llm/config', {method:'GET'}).catch(()=>null);
        const llmEnabled = llmCfg && llmCfg.has_key && llmCfg.model;
        $('#play-mode option[value*="llm"]').prop('disabled', !llmEnabled);

        updatePlayerIndicators();
        console.log('✅ System status updated.');
        } catch (error) {
        console.error('❌ System status error:', error);
    }
}

function updateMaterialCount() {
    // Count captures via move history containing 'x'
    const captures = game.history({ verbose: true }).filter(m => m.flags.includes('c')).length;
    $('#captures-count').text(captures);
    // Existing material advantage logic can remain if desired
}

function highlightMove(from, to) {
    removeHighlights();
    $(`[data-square="${from}"]`).addClass('highlight-white');
    $(`[data-square="${to}"]`).addClass('highlight-black');
}

function removeHighlights() {
    $('.square-55d63').removeClass('highlight-white highlight-black');
}

function showNotification(message, type = 'info') {
    const alertClass = `alert-${type}`;
    const notification = $(`<div class="alert ${alertClass} alert-dismissible fade show">${message}</div>`);
    $('#notification-area').append(notification);
    setTimeout(() => notification.alert('close'), 5000);
}

function parseGameModeEngines() {
    const mode = $('#play-mode').val() || 'human-vs-neural';
    if (mode.includes('human-vs-')) {
        return ['human', mode.split('-vs-')[1]];
    } else if (mode.includes('-vs-human')) {
        return [mode.split('-vs-')[0], 'human'];
    } else {
        return mode.split('-vs-');
    }
}

function getEngineDisplayName(engine) {
    switch(engine) {
        case 'neural':
            return 'Neural Network';
        case 'stockfish':
            return 'Stockfish';
        case 'leela':
            return 'LeelaZero';
        case 'hopechess':
            return 'HopeChess';
        case 'llm':
            return 'Gemini LLM';
        case 'human':
            return 'Human Player';
        default:
            return 'Unknown';
    }
}

function updateEngineStatus(status, engine) {
    // This is a placeholder for more detailed status updates in the UI
}

// --- Utility Controls ---
function takebackMove() {
    if (game.history().length === 0 || game.game_over()) {
        showNotification('No moves to take back.', 'warning');
        return;
    }
    // Undo last two moves if last move was engine (common for human vs engine)
    game.undo();
    if (!parseGameModeEngines().includes('human') && game.history().length > 0) {
        game.undo();
    }
    board.position(game.fen());
    removeHighlights();
    updateGameStatus();
    updateGameInfo();
    debounceEvaluation();
    console.log('↩️ Move taken back.');
}

function resignGame() {
    if (game.game_over()) {
        showNotification('Game already over.', 'info');
        return;
    }
    const resigningSide = game.turn() === 'w' ? 'White' : 'Black';
    const winner = resigningSide === 'White' ? 'Black' : 'White';
    showNotification(`${resigningSide} resigned. ${winner} wins.`, 'danger');
    // Mark game as over by setting insufficient material artificially
    // simpler: clear interval timers and stop auto play
    clearInterval(gameTimer);
    if (isAutoPlaying) stopAutoPlay();
    $('#game-status').text(`Resignation! ${winner} wins.`);
    console.log(`🏳️ ${resigningSide} resigned.`);
}

function offerDraw() {
    if (game.game_over()) {
        showNotification('Game already over.', 'info');
        return;
    }
    showNotification('Draw agreed.', 'primary');
    clearInterval(gameTimer);
    if (isAutoPlaying) stopAutoPlay();
    $('#game-status').text('Draw agreed.');
    console.log('🤝 Draw agreed.');
}

function declareEngineErrorLoser(engineName) {
    const winner = engineName === 'White Engine' ? 'Black' : 'White';
    $('#game-result-value').text(`${engineName} error – ${winner} wins.`);
    $('#game-result-row').show();
    $('#elo-estimation').hide();
}

// LLM settings save handler
$(document).on('click', '#save-llm', async function() {
    const key = $('#llm-api-key').val().trim();
    const model = $('#llm-model').val();
    if(!model){ showNotification('Please select a model before saving.','warning'); return; }
    try {
        await apiCall('/api/llm/config', {method:'POST', body:{api_key:key, model:model}, timeout:0});
        showNotification('LLM configuration saved.', 'success');
        $('#settingsModal').modal('hide');
        await checkSystemStatus(); // refresh dropdown enablement
    } catch(e) {
        showNotification('Failed to save LLM config: '+e.message, 'danger');
    }
});

// Pre-fill modal when opened (show existing config but don't fetch models automatically)
$('#settingsModal').on('show.bs.modal', async function(){
    $('#llm-model').prop('disabled', true).empty().append('<option selected disabled value="">Select a model…</option>');
    $('#save-llm').prop('disabled', true);
    try {
        const cfg = await apiCall('/api/llm/config?include_key=1',{method:'GET', timeout:0});
        if(cfg.api_key){
            $('#llm-api-key').val(cfg.api_key);
            // If a model already saved, fetch list automatically so user sees it
            if(cfg.model){
                await fetchAndPopulateModels(cfg.api_key, cfg.model.replace('models/',''));
            }
        }
    } catch(e){ console.error(e); }
});

async function fetchAndPopulateModels(apiKey, preselect=null){
    try{
        // Temporarily disable button during fetch
        $('#llm-fetch-models').prop('disabled', true).text('Fetching…');

        // Persist key (without model) so backend can use it for /api/llm/models
        await apiCall('/api/llm/config', {method:'POST', body:{api_key: apiKey}, timeout:0});

        const mdlResp = await apiCall('/api/llm/models',{method:'GET', timeout:0});
        const select = $('#llm-model');
        select.prop('disabled', false).empty();
        mdlResp.models.forEach(m=>select.append(`<option value="${m}">${m}</option>`));
        if(preselect){ select.val(preselect); }
        $('#save-llm').prop('disabled', false);
        showNotification('Model list updated.', 'success');
    }catch(e){
        showNotification('Failed to fetch models: '+e.message, 'danger');
        console.error(e);
    }finally{
        $('#llm-fetch-models').prop('disabled', false).text('Confirm Key & Fetch Models');
    }
}

// Fetch models on button click (after user entered API key)
$(document).on('click', '#llm-fetch-models', async function(){
    const key = $('#llm-api-key').val().trim();
    if(!key){ showNotification('Please enter an API key first.','warning'); return; }
    await fetchAndPopulateModels(key, null);
}); 