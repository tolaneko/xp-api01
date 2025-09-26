import express from "express";
import fetch from "node-fetch";
import math from "mathjs";
import { NeuralNetwork } from "brain.js";

const app = express();
const PORT = process.env.PORT || 3000;

const url = "https://app-tai-xiu-default-rtdb.firebaseio.com/taixiu_sessions.json";

let latestPhien = null;
let latestData = null;
let predictionHistory = [];

class OptimizedDicePredictor {
    constructor() {
        this.history = [];
        this.modelWeights = {};
        this.modelPerformance = {};
        this.patternDatabase = {};
        this.marketRegime = 'normal';
        this.initializeLayeredModels();
        this.setupNeuralNetworks();
    }

    initializeLayeredModels() {
        this.tier1Models = {
            model101: { name: 'Data Normalization', weight: 0.9 },
            model102: { name: 'Feature Extraction', weight: 0.9 },
            model103: { name: 'Sequence Encoding', weight: 0.85 },
            model104: { name: 'Volatility Calculation', weight: 0.8 }
        };

        this.tier2Models = {
            model201: { name: 'Basic Pattern Recognition', weight: 1.2 },
            model202: { name: 'Advanced Pattern Matching', weight: 1.3 },
            model203: { name: 'Markov Chain Analysis', weight: 1.1 },
            model204: { name: 'Sequential Pattern Mining', weight: 1.0 },
            model205: { name: 'Fourier Transform Analysis', weight: 0.9 }
        };

        this.tier3Models = {
            model301: { name: 'Statistical Arbitrage', weight: 1.4 },
            model302: { name: 'Bayesian Inference', weight: 1.3 },
            model303: { name: 'Monte Carlo Simulation', weight: 1.1 },
            model304: { name: 'Regression Analysis', weight: 1.2 },
            model305: { name: 'Time Series Analysis', weight: 1.1 }
        };

        this.allModels = {
            ...this.tier1Models,
            ...this.tier2Models,
            ...this.tier3Models
        };

        for (const model in this.allModels) {
            this.modelPerformance[model] = {
                correct: 0,
                total: 0,
                accuracy: 0,
                recentAccuracy: 0,
                streak: 0,
                lastPrediction: null,
                recentPredictions: []
            };
            this.modelWeights[model] = this.allModels[model].weight;
        }
    }

    setupNeuralNetworks() {
        this.patternNN = new NeuralNetwork({
            hiddenLayers: [8, 6],
            activation: 'sigmoid',
            learningRate: 0.3
        });

        this.trendNN = new NeuralNetwork({
            hiddenLayers: [6, 4],
            activation: 'tanh',
            learningRate: 0.25
        });

        this.initializeNeuralNetworks();
    }

    initializeNeuralNetworks() {
        const patternTrainingData = [
            { input: [1, 0, 1, 0], output: { Tai: 0.7, Xiu: 0.3 } },
            { input: [0, 1, 0, 1], output: { Tai: 0.3, Xiu: 0.7 } },
            { input: [1, 1, 0, 0], output: { Tai: 0.4, Xiu: 0.6 } },
            { input: [0, 0, 1, 1], output: { Tai: 0.6, Xiu: 0.4 } }
        ];

        const trendTrainingData = [
            { input: [0.8, 0.6, 0.7], output: { up: 0.8, down: 0.2 } },
            { input: [0.3, 0.4, 0.2], output: { up: 0.2, down: 0.8 } },
            { input: [0.5, 0.5, 0.5], output: { up: 0.5, down: 0.5 } }
        ];

        this.patternNN.train(patternTrainingData, { iterations: 1000 });
        this.trendNN.train(trendTrainingData, { iterations: 800 });
    }

    // TẦNG 1: DATA PREPROCESSING
    normalizeData(sequence) {
        if (sequence.length === 0) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const mean = math.mean(encoded);
        const std = math.std(encoded);
        
        const normalized = encoded.map(val => std !== 0 ? (val - mean) / std : 0);
        
        return {
            normalized,
            stats: { mean, std },
            reason: `Dữ liệu chuẩn hóa (mean: ${mean.toFixed(3)}, std: ${std.toFixed(3)})`
        };
    }

    extractFeatures(sequence) {
        if (sequence.length < 5) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        
        const features = {
            mean: math.mean(encoded),
            variance: math.variance(encoded),
            volatility: this.calculateVolatility(sequence),
            trendStrength: this.calculateTrendStrength(encoded),
            autocorrelation: this.calculateAutocorrelation(encoded, 1),
            entropy: this.calculateEntropy(encoded)
        };
        
        return {
            features,
            reason: 'Đã trích xuất 6 đặc trưng quan trọng'
        };
    }

    calculateAdvancedVolatility(sequence, window = 5) {
        if (sequence.length < window) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const volatilities = [];
        
        for (let i = window; i <= encoded.length; i++) {
            const windowData = encoded.slice(i - window, i);
            const changes = windowData.filter((val, idx, arr) => 
                idx > 0 && val !== arr[idx - 1]).length;
            volatilities.push(changes / (window - 1));
        }
        
        const currentVolatility = volatilities[volatilities.length - 1] || 0.5;
        let volatilityRegime = 'normal';
        
        if (currentVolatility > 0.7) volatilityRegime = 'high';
        else if (currentVolatility < 0.3) volatilityRegime = 'low';
        
        this.marketRegime = volatilityRegime;
        
        return {
            volatility: currentVolatility,
            regime: volatilityRegime,
            reason: `Biến động: ${(currentVolatility * 100).toFixed(1)}% (${volatilityRegime})`
        };
    }

    // TẦNG 2: PATTERN RECOGNITION
    enhancedBasicPatternRecognition(sequence) {
        if (sequence.length < 4) return null;
        
        const recent = sequence.slice(-4);
        const encoded = recent.map(item => item === 'Tài' ? 1 : 0);
        
        const nnResult = this.patternNN.run(encoded);
        const prediction = nnResult.Tai > nnResult.Xiu ? 'Tài' : 'Xỉu';
        const confidence = Math.max(nnResult.Tai, nnResult.Xiu);
        
        return {
            prediction,
            confidence,
            reason: `Neural Network pattern recognition`,
            details: { nnResult, pattern: encoded.join('') }
        };
    }

    markovChainAnalysis(sequence) {
        if (sequence.length < 10) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const transitions = { 0: { 0: 0, 1: 0 }, 1: { 0: 0, 1: 0 } };
        
        for (let i = 1; i < encoded.length; i++) {
            const from = encoded[i - 1];
            const to = encoded[i];
            transitions[from][to]++;
        }
        
        for (const from of [0, 1]) {
            const total = transitions[from][0] + transitions[from][1];
            if (total > 0) {
                transitions[from][0] /= total;
                transitions[from][1] /= total;
            }
        }
        
        const lastState = encoded[encoded.length - 1];
        const taiProbability = transitions[lastState][1];
        const xiuProbability = transitions[lastState][0];
        
        const prediction = taiProbability > xiuProbability ? 'Tài' : 'Xỉu';
        const confidence = Math.max(taiProbability, xiuProbability);
        
        return {
            prediction,
            confidence,
            reason: `Markov Chain: P(Tài)=${taiProbability.toFixed(3)}`,
            transitions
        };
    }

    sequentialPatternMining(sequence, minSupport = 0.3) {
        if (sequence.length < 8) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const patterns = this.findFrequentPatterns(encoded, 2, 3, minSupport);
        
        if (patterns.length === 0) return null;
        
        patterns.sort((a, b) => b.support - a.support);
        const bestPattern = patterns[0];
        const nextResults = this.findNextResultsForPattern(encoded, bestPattern.pattern);
        
        if (nextResults.total > 0) {
            const taiProbability = nextResults.tai / nextResults.total;
            const prediction = taiProbability > 0.5 ? 'Tài' : 'Xỉu';
            const confidence = Math.max(taiProbability, 1 - taiProbability);
            
            return {
                prediction,
                confidence,
                reason: `Sequential pattern (support: ${bestPattern.support.toFixed(3)})`,
                pattern: bestPattern.pattern.join('')
            };
        }
        
        return null;
    }

    // TẦNG 3: STATISTICAL ANALYSIS
    statisticalArbitrage(sequence) {
        if (sequence.length < 15) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const mean = math.mean(encoded);
        const std = math.std(encoded);
        
        const zScore = Math.abs((mean - 0.5) / (std / Math.sqrt(encoded.length)));
        
        if (zScore > 1.5) {
            const prediction = mean > 0.5 ? 'Xỉu' : 'Tài';
            const confidence = Math.min(0.8, 0.5 + zScore * 0.2);
            
            return {
                prediction,
                confidence,
                reason: `Statistical arbitrage (z-score: ${zScore.toFixed(2)})`,
                stats: { mean, std, zScore }
            };
        }
        
        return null;
    }

    bayesianInference(sequence) {
        if (sequence.length < 8) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const taiCount = encoded.reduce((sum, val) => sum + val, 0);
        const total = encoded.length;
        
        const priorStrength = this.marketRegime === 'high' ? 2 : 5;
        const taiPosterior = (taiCount + priorStrength) / (total + 2 * priorStrength);
        
        if (Math.abs(taiPosterior - 0.5) > 0.15) {
            const prediction = taiPosterior > 0.5 ? 'Xỉu' : 'Tài';
            const confidence = 0.5 + Math.abs(taiPosterior - 0.5);
            
            return {
                prediction,
                confidence,
                reason: `Bayesian inference (P(Tài)=${taiPosterior.toFixed(3)})`
            };
        }
        
        return null;
    }

    monteCarloSimulation(sequence, simulations = 500) {
        if (sequence.length < 10) return null;
        
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        const transitions = { 0: { 0: 0, 1: 0 }, 1: { 0: 0, 1: 0 } };
        
        for (let i = 1; i < encoded.length; i++) {
            const from = encoded[i - 1];
            const to = encoded[i];
            transitions[from][to]++;
        }
        
        for (const from of [0, 1]) {
            const total = transitions[from][0] + transitions[from][1];
            if (total > 0) {
                transitions[from][0] /= total;
                transitions[from][1] /= total;
            }
        }
        
        const lastState = encoded[encoded.length - 1];
        let taiResults = 0;
        
        for (let i = 0; i < simulations; i++) {
            let currentState = lastState;
            for (let step = 0; step < 3; step++) {
                const rand = Math.random();
                currentState = rand < transitions[currentState][1] ? 1 : 0;
            }
            if (currentState === 1) taiResults++;
        }
        
        const taiProbability = taiResults / simulations;
        const prediction = taiProbability > 0.5 ? 'Tài' : 'Xỉu';
        const confidence = Math.max(taiProbability, 1 - taiProbability);
        
        return {
            prediction,
            confidence,
            reason: `Monte Carlo (${simulations} runs, P(Tài)=${taiProbability.toFixed(3)})`
        };
    }

    // UTILITY FUNCTIONS
    findFrequentPatterns(sequence, minLength, maxLength, minSupport) {
        const patterns = [];
        const n = sequence.length;
        
        for (let length = minLength; length <= maxLength; length++) {
            for (let start = 0; start <= n - length; start++) {
                const pattern = sequence.slice(start, start + length);
                const patternString = pattern.join('');
                
                let count = 0;
                for (let i = 0; i <= n - length; i++) {
                    const candidate = sequence.slice(i, i + length);
                    if (candidate.join('') === patternString) count++;
                }
                
                const support = count / (n - length + 1);
                if (support >= minSupport) {
                    patterns.push({ pattern, support, count });
                }
            }
        }
        
        return patterns;
    }

    findNextResultsForPattern(sequence, pattern) {
        const result = { tai: 0, xiu: 0, total: 0 };
        const patternString = pattern.join('');
        const patternLength = pattern.length;
        
        for (let i = 0; i <= sequence.length - patternLength - 1; i++) {
            const candidate = sequence.slice(i, i + patternLength);
            if (candidate.join('') === patternString) {
                const nextValue = sequence[i + patternLength];
                if (nextValue === 1) result.tai++;
                else result.xiu++;
                result.total++;
            }
        }
        
        return result;
    }

    calculateVolatility(sequence) {
        if (sequence.length < 2) return 0;
        const encoded = sequence.map(item => item === 'Tài' ? 1 : 0);
        let changes = 0;
        for (let i = 1; i < encoded.length; i++) {
            if (encoded[i] !== encoded[i - 1]) changes++;
        }
        return changes / (encoded.length - 1);
    }

    calculateTrendStrength(data) {
        if (data.length < 2) return 0;
        let ups = 0, downs = 0;
        for (let i = 1; i < data.length; i++) {
            if (data[i] > data[i - 1]) ups++;
            else if (data[i] < data[i - 1]) downs++;
        }
        return Math.abs(ups - downs) / (data.length - 1);
    }

    calculateAutocorrelation(data, lag) {
        if (data.length <= lag) return 0;
        const mean = math.mean(data);
        let numerator = 0, denominator = 0;
        for (let i = lag; i < data.length; i++) {
            numerator += (data[i] - mean) * (data[i - lag] - mean);
        }
        for (let i = 0; i < data.length; i++) {
            denominator += Math.pow(data[i] - mean, 2);
        }
        return denominator !== 0 ? numerator / denominator : 0;
    }

    calculateEntropy(data) {
        const counts = { 0: 0, 1: 0 };
        data.forEach(val => counts[val]++);
        const p0 = counts[0] / data.length;
        const p1 = counts[1] / data.length;
        let entropy = 0;
        if (p0 > 0) entropy -= p0 * Math.log2(p0);
        if (p1 > 0) entropy -= p1 * Math.log2(p1);
        return entropy;
    }

    // MAIN PREDICTION METHOD
    async predict() {
        if (this.history.length < 5) {
            return this.getRandomPrediction();
        }

        const modelPredictions = [];
        
        // Chạy các models
        const models = [
            () => this.enhancedBasicPatternRecognition(this.history),
            () => this.markovChainAnalysis(this.history),
            () => this.sequentialPatternMining(this.history),
            () => this.statisticalArbitrage(this.history),
            () => this.bayesianInference(this.history),
            () => this.monteCarloSimulation(this.history)
        ];

        for (const model of models) {
            try {
                const result = model();
                if (result) modelPredictions.push(result);
            } catch (error) {
                console.error('Model error:', error);
            }
        }

        if (modelPredictions.length === 0) {
            return this.getRandomPrediction();
        }

        // Kết hợp predictions
        return this.combinePredictions(modelPredictions);
    }

    getRandomPrediction() {
        const randomPred = Math.random() > 0.5 ? 'Tài' : 'Xỉu';
        return {
            prediction: randomPred,
            confidence: 0.5,
            reason: 'Dự đoán ngẫu nhiên (không đủ dữ liệu)',
            model: 'random',
            probabilities: {
                Tai: 0.5,
                Xiu: 0.5
            }
        };
    }

    combinePredictions(predictions) {
        let taiScore = 0;
        let xiuScore = 0;
        let totalWeight = 0;
        const modelDetails = {};

        predictions.forEach((pred, index) => {
            const weight = pred.confidence;
            if (pred.prediction === 'Tài') {
                taiScore += weight;
            } else {
                xiuScore += weight;
            }
            totalWeight += weight;
            
            modelDetails[`model_${index}`] = {
                prediction: pred.prediction,
                confidence: pred.confidence,
                reason: pred.reason
            };
        });

        const taiProbability = taiScore / totalWeight;
        const xiuProbability = xiuScore / totalWeight;
        
        const finalPrediction = taiProbability > xiuProbability ? 'Tài' : 'Xỉu';
        const finalConfidence = Math.max(taiProbability, xiuProbability);

        return {
            prediction: finalPrediction,
            confidence: finalConfidence,
            reason: `Kết hợp ${predictions.length} mô hình`,
            details: modelDetails,
            probabilities: {
                Tai: taiProbability,
                Xiu: xiuProbability
            }
        };
    }

    addResult(result) {
        this.history.push(result);
        // Giữ lịch sử tối đa 50 phiên
        if (this.history.length > 50) {
            this.history.shift();
        }
    }

    getHistory() {
        return this.history;
    }

    getHistoryStats() {
        if (this.history.length === 0) return null;
        
        const encoded = this.history.map(item => item === 'Tài' ? 1 : 0);
        const taiCount = encoded.reduce((sum, val) => sum + val, 0);
        const xiuCount = this.history.length - taiCount;
        
        return {
            total: this.history.length,
            tai: taiCount,
            xiu: xiuCount,
            taiRatio: taiCount / this.history.length,
            xiuRatio: xiuCount / this.history.length,
            currentStreak: this.calculateCurrentStreak()
        };
    }

    calculateCurrentStreak() {
        if (this.history.length === 0) return 0;
        
        let streak = 1;
        const lastResult = this.history[this.history.length - 1];
        
        for (let i = this.history.length - 2; i >= 0; i--) {
            if (this.history[i] === lastResult) {
                streak++;
            } else {
                break;
            }
        }
        
        return streak;
    }
}

// Khởi tạo predictor
const predictor = new OptimizedDicePredictor();

async function checkSession() {
    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data) {
            const sessions = Object.values(data).filter(x => x.Phien !== undefined);

            if (sessions.length > 0) {
                const latestSession = sessions.reduce((a, b) =>
                    a.Phien > b.Phien ? a : b
                );

                if (latestPhien === null || latestSession.Phien > latestPhien) {
                    // Thêm kết quả mới vào predictor
                    const result = latestSession.tong >= 11 ? 'Tài' : 'Xỉu';
                    predictor.addResult(result);
                    
                    // Dự đoán phiên tiếp theo
                    const prediction = await predictor.predict();
                    
                    latestPhien = latestSession.Phien;
                    latestData = {
                        ...latestSession,
                        prediction: prediction.prediction,
                        confidence: prediction.confidence
                    };

                    // Lưu lịch sử dự đoán
                    predictionHistory.push({
                        phien: latestSession.Phien,
                        actual: result,
                        prediction: prediction.prediction,
                        confidence: prediction.confidence,
                        timestamp: new Date()
                    });

                    // Giữ tối đa 100 bản ghi
                    if (predictionHistory.length > 100) {
                        predictionHistory.shift();
                    }

                    console.log("Phiên mới:", latestSession.Phien);
                    console.log("Kết quả:", result, "- Tổng:", latestSession.tong);
                    console.log("Dự đoán phiên tiếp theo:", prediction.prediction);
                    console.log("Độ tin cậy:", (prediction.confidence * 100).toFixed(1) + "%");
                    console.log("Lý do:", prediction.reason);
                    console.log("-------------------------------");
                }
            }
        }
    } catch (err) {
        console.error("Lỗi:", err.message);
    }
}

// API endpoint chính với định dạng bạn yêu cầu
app.get("/api/taixiu", async (req, res) => {
    if (latestData) {
        try {
            // Lấy dự đoán mới nhất
            const prediction = await predictor.predict();
            const stats = predictor.getHistoryStats();
            
            // Format kết quả theo đúng yêu cầu
            const responseData = {
                "Phien": latestData.Phien,
                "Xuc_xac_1": latestData.xuc_xac_1,
                "Xuc_xac_2": latestData.xuc_xac_2,
                "Xuc_xac_3": latestData.xuc_xac_3,
                "Tong": latestData.tong,
                "Ket_qua": latestData.tong >= 11 ? 'Tài' : 'Xỉu',
                "Phien_hien_tai": latestData.Phien + 1,
                "Du_doan": `>>${prediction.prediction}<<`,
                "Ti_le": `>>${(prediction.confidence * 100).toFixed(1)}%<<`,
                "id": "Cstooldudoan11",
                "thong_tin_them": {
                    "so_luong_mau": stats ? stats.total : 0,
                    "ti_le_tai": stats ? (stats.taiRatio * 100).toFixed(1) + '%' : '0%',
                    "ti_le_xiu": stats ? (stats.xiuRatio * 100).toFixed(1) + '%' : '0%',
                    "do_tin_cay": prediction.reason
                }
            };
            
            res.json(responseData);
            
        } catch (error) {
            res.json({
                "Phien": latestData.Phien,
                "Xuc_xac_1": latestData.xuc_xac_1,
                "Xuc_xac_2": latestData.xuc_xac_2,
                "Xuc_xac_3": latestData.xuc_xac_3,
                "Tong": latestData.tong,
                "Ket_qua": latestData.tong >= 11 ? 'Tài' : 'Xỉu',
                "Phien_hien_tai": latestData.Phien + 1,
                "Du_doan": ">>Đang phân tích...<<",
                "Ti_le": ">>0%<<",
                "id": "Cstooldudoan11"
            });
        }
    } else {
        res.json({ 
            message: "Chưa có dữ liệu",
            Du_doan: ">>Chờ dữ liệu...<<",
            Ti_le: ">>0%<<"
        });
    }
});

// Các API endpoints phụ
app.get("/api/prediction", async (req, res) => {
    try {
        const prediction = await predictor.predict();
        const stats = predictor.getHistoryStats();
        
        res.json({
            prediction: prediction.prediction,
            confidence: prediction.confidence,
            reason: prediction.reason,
            probabilities: prediction.probabilities,
            history_stats: stats,
            model_details: prediction.details
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get("/api/history", (req, res) => {
    const stats = predictor.getHistoryStats();
    const recentHistory = predictor.getHistory().slice(-10);
    
    res.json({
        history: recentHistory,
        stats: stats,
        prediction_history: predictionHistory.slice(-20)
    });
});

app.get("/api/stats", (req, res) => {
    const stats = predictor.getHistoryStats();
    const totalPredictions = predictionHistory.length;
    const correctPredictions = predictionHistory.filter(p => 
        p.prediction === p.actual).length;
    const accuracy = totalPredictions > 0 ? (correctPredictions / totalPredictions) * 100 : 0;

    res.json({
        accuracy: accuracy.toFixed(1) + '%',
        total_predictions: totalPredictions,
        correct_predictions: correctPredictions,
        current_streak: stats ? stats.currentStreak : 0,
        market_regime: predictor.marketRegime
    });
});

// check session mỗi 3 giây
setInterval(checkSession, 3000);
checkSession();

app.listen(PORT, () => {
    console.log(`Server đang chạy tại http://localhost:${PORT}`);
    console.log(`API endpoints:
- /api/taixiu : Dữ liệu phiên hiện tại + dự đoán (định dạng chính)
- /api/prediction : Chi tiết dự đoán
- /api/history : Lịch sử kết quả và dự đoán
- /api/stats : Thống kê độ chính xác`);
});
