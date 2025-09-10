// script.js - Medical Pose Detection & Analysis Suite
let collectedSamples = [];   // store all samples here

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

// Core state variables
let detector = null;
let isDetecting = false;
let animationId = null;
let startTime = null;
let sessionTimer = null;
let exerciseMode = false;

// Analytics
let frameCount = 0;
let totalConfidence = 0;
let poseCount = 0;
let lastFpsTime = Date.now();
let exerciseCount = 0;

// Squat detection variables
let squatState = 'up';
let stateFrameCount = 0;
let minStateFrames = 3;
let lastSquatTime = 0;

// Medical assessment variables
let medicalMode = 'posture';
let postureAssessments = [];
let fallRiskHistory = [];
let exerciseSession = null;
let scoliosisReadings = [];
let balanceTestActive = false;
let balanceTestStartTime = null;
let balanceTestDuration = 30000; // 30 seconds

// Patient data
let currentPatientData = {
    name: 'Patient',
    age: 0,
    conditions: [],
    lastAssessment: null
};

// Medical assessment thresholds
const POSTURE_THRESHOLDS = {
    excellent: { head: 15, shoulder: 10, spine: 20 },
    good: { head: 25, shoulder: 20, spine: 35 },
    fair: { head: 40, shoulder: 35, spine: 50 }
};

const FALL_RISK_FACTORS = {
    sway: 30,
    stability: 0.7,
    balance: 25
};

// Utility functions
function log(msg, type = 'info') {
    console.log(msg);
    status.textContent = msg;
    if (type === 'error') {
        status.style.background = '#f44336';
    } else if (type === 'success') {
        status.style.background = '#4CAF50';
    } else {
        status.style.background = '#444';
    }
}




// Camera functions
async function startCamera() {
    try {
        log('Starting camera...', 'info');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480,
                facingMode: 'user'
            }
        });
        
        video.srcObject = stream;
        
        video.onloadeddata = () => {
            log('Camera ready! Load AI Model to continue', 'success');
            document.getElementById('aiBtn').disabled = false;
        };
        
    } catch (error) {
        log('Camera error: ' + error.message, 'error');
    }
}

async function loadAI() {
    try {
        log('Loading TensorFlow.js...', 'info');
        
        await tf.ready();
        log('TensorFlow.js ready!', 'success');
        
        log('Loading pose detection model...', 'info');
        
        // Use MoveNet with proper configuration
        detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            { 
                modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
                enableSmoothing: true
            }
        );
        
        log('AI model loaded! Start detection to begin analysis', 'success');
        document.getElementById('detectBtn').disabled = false;
        
    } catch (error) {
        console.error('AI loading error:', error);
        log('AI loading error: ' + error.message, 'error');
    }
}
// Detection control functions
async function startDetection() {
    if (!detector) {
        log('AI model not loaded yet', 'error');
        return;
    }
    
    isDetecting = true;
    startTime = Date.now();
    document.getElementById('detectBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    log('Analyzing poses and posture...', 'success');
    
    // Reset detection variables
    squatState = 'up';
    stateFrameCount = 0;
    
    startSessionTimer();
    detectLoop();
}

function stopDetection() {
    isDetecting = false;
    document.getElementById('detectBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    
    if (sessionTimer) {
        clearInterval(sessionTimer);
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    log('Detection stopped', 'info');
}

function startSessionTimer() {
    sessionTimer = setInterval(() => {
        if (startTime) {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('session-time').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }, 1000);
}
let lastKeypoints=[];
// Main detection loop
async function detectLoop() {
    if (!isDetecting) return;
    
    try {
        const poses = await detector.estimatePoses(video);
        
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (poses.length > 0) {
            const pose = poses[0];
            lastKeypoints = pose.keypoints;
            drawPose(pose);
            analyzePose(pose);
            analyzeMedicalPose(pose);
            updateStats(pose);
            
            if (exerciseMode) {
                countSquats(pose);
            }
            
            if (balanceTestActive) {
                updateBalanceTest(pose);
            }
        }
        
        updateFPS();
        
    } catch (error) {
        console.error('Detection error:', error);
        log('Detection error: ' + error.message, 'error');
        stopDetection();
        return;
    }
    
    animationId = requestAnimationFrame(detectLoop);
}

// Drawing functions
function drawPose(pose) {
    const keypoints = pose.keypoints;
    
    // Draw keypoints
    keypoints.forEach(keypoint => {
        if (keypoint.score > 0.3) {
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = keypoint.score > 0.6 ? '#00ff00' : '#ffaa00';
            ctx.fill();
        }
    });
    
    // Draw skeleton
    const connections = [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // arms
        [11, 12], [5, 11], [6, 12],  // torso
        [11, 13], [13, 15], [12, 14], [14, 16]  // legs
    ];
    
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 2;
    
    connections.forEach(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];
        
        if (kp1 && kp2 && kp1.score > 0.3 && kp2.score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
        }
    });
    
    // Highlight key points for squat detection when exercise mode is on
    if (exerciseMode) {
        const leftHip = keypoints[11];
        const rightHip = keypoints[12];
        const leftKnee = keypoints[13];
        const rightKnee = keypoints[14];
        
        [leftHip, rightHip, leftKnee, rightKnee].forEach(kp => {
            if (kp && kp.score > 0.5) {
                ctx.beginPath();
                ctx.arc(kp.x, kp.y, 8, 0, 2 * Math.PI);
                ctx.strokeStyle = '#ff00ff';
                ctx.lineWidth = 3;
                ctx.stroke();
            }
        });
    }
}

// Enhanced posture analysis that includes sitting
// Replace the existing analyzePose function with this corrected version

function analyzePose(pose) {
    const keypoints = pose.keypoints;
    
    // Get key body parts
    const nose = keypoints[0];
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    
    // Determine body position first
    determineBodyPosition(keypoints);
    
    // Get current position
    const currentPosition = document.getElementById('body-position').textContent;
    
    // Analyze posture based on position
    if (currentPosition.includes('Sitting')) {
        // Use sitting-specific posture analysis
        analyzeSittingPosture(keypoints);
    } else {
        // Use existing standing posture analysis
        analyzeStandingPosture(nose, leftShoulder, rightShoulder, leftHip, rightHip);
    }
}
// Add sitting-specific exercise counting
// Add this function to handle sitting exercises

function countSittingExercises(pose) {
    const keypoints = pose.keypoints;
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const nose = keypoints[0];
    
    if (!leftShoulder || !rightShoulder || !nose) return;
    
    const shoulderCenter = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2
    };
    
    // Detect neck stretches or shoulder rolls while sitting
    const headMovement = Math.abs(nose.y - shoulderCenter.y);
    
    // Simple sitting exercise detection (can be expanded)
    if (headMovement > 60) {
        // Could be a neck stretch
        updateSquatIndicator('Neck exercise detected', 'squat-transition');
    }
}

function analyzeStandingPosture(nose, leftShoulder, rightShoulder, leftHip, rightHip) {
    if (!nose || !leftShoulder || !rightShoulder || !leftHip || !rightHip) {
        updatePostureDisplay('Unknown', 'Need clearer view of upper body', 'warning');
        return;
    }
    
    // Calculate shoulder alignment
    const shoulderSlope = Math.abs(leftShoulder.y - rightShoulder.y);
    const shoulderCenter = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2
    };
    
    // Calculate hip alignment
    const hipCenter = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
    };
    
    // Check head position relative to shoulders
    const headForward = Math.abs(nose.x - shoulderCenter.x) > 30;
    
    // Check spine alignment
    const spineDeviation = Math.abs(shoulderCenter.x - hipCenter.x);
    
    // Determine posture quality
    let postureStatus, advice, level;
    
    if (shoulderSlope < 15 && !headForward && spineDeviation < 25) {
        postureStatus = 'Excellent Standing Posture';
        advice = 'Great job! Keep maintaining this alignment';
        level = 'good';
    } else if (shoulderSlope < 25 && spineDeviation < 40) {
        postureStatus = 'Good Standing Posture';
        advice = headForward ? 'Try to keep head above shoulders' : 'Minor adjustments needed';
        level = 'warning';
    } else {
        postureStatus = 'Poor Standing Posture';
        const issues = [];
        if (shoulderSlope > 25) issues.push('uneven shoulders');
        if (headForward) issues.push('forward head');  
        if (spineDeviation > 40) issues.push('spine misalignment');
        advice = `Address: ${issues.join(', ')}`;
        level = 'bad';
    }
    
    updatePostureDisplay(postureStatus, advice, level);
}
// Improved sitting detection in determineBodyPosition
function determineBodyPosition(keypoints) {
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    const leftKnee = keypoints[13];
    const rightKnee = keypoints[14];
    const leftAnkle = keypoints[15];
    const rightAnkle = keypoints[16];
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    
    let position = 'Unknown';
    let confidence = 0;
    
    if (leftHip && rightHip && leftKnee && rightKnee && 
        leftHip.score > 0.4 && rightHip.score > 0.4 && 
        leftKnee.score > 0.4 && rightKnee.score > 0.4) {
        
        const avgHip = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        };
        const avgKnee = {
            x: (leftKnee.x + rightKnee.x) / 2,
            y: (leftKnee.y + rightKnee.y) / 2
        };
        
        // Calculate key measurements
        const hipKneeVerticalDiff = avgKnee.y - avgHip.y;
        const hipKneeHorizontalDiff = Math.abs(avgKnee.x - avgHip.x);
        
        // Additional measurements for better sitting detection
        let avgAnkle = null;
        let avgShoulder = null;
        
        if (leftAnkle && rightAnkle && leftAnkle.score > 0.3 && rightAnkle.score > 0.3) {
            avgAnkle = {
                x: (leftAnkle.x + rightAnkle.x) / 2,
                y: (leftAnkle.y + rightAnkle.y) / 2
            };
        }
        
        if (leftShoulder && rightShoulder && leftShoulder.score > 0.4 && rightShoulder.score > 0.4) {
            avgShoulder = {
                x: (leftShoulder.x + rightShoulder.x) / 2,
                y: (leftShoulder.y + rightShoulder.y) / 2
            };
        }
        
        // Enhanced position detection logic
        if (avgAnkle && avgShoulder) {
            const kneeAngleDegrees = calculateKneeAngle(avgHip, avgKnee, avgAnkle);
            const hipShoulderVerticalDiff = avgShoulder.y - avgHip.y;
            const kneeAnkleVerticalDiff = avgAnkle.y - avgKnee.y;
            const shoulderKneeHorizontalDiff = Math.abs(avgShoulder.x - avgKnee.x);
            
            // Improved sitting detection criteria
            const isSittingAngle = kneeAngleDegrees > 60 && kneeAngleDegrees < 130;
            const isSittingHeight = hipKneeVerticalDiff > 30 && hipKneeVerticalDiff < 150;
            const isSittingPosture = hipShoulderVerticalDiff < -10; // shoulders above hips
            const hasFootContact = kneeAnkleVerticalDiff > 10; // feet likely on ground
            
            if (isSittingAngle && isSittingHeight && isSittingPosture && hasFootContact) {
                confidence = 85;
                
                // Sub-categorize sitting posture
                if (hipShoulderVerticalDiff < -60) {
                    position = 'Sitting - Upright';
                    confidence = 90;
                } else if (hipShoulderVerticalDiff > -30) {
                    position = 'Sitting - Slouched';
                    confidence = 80;
                } else {
                    position = 'Sitting';
                    confidence = 85;
                }
            }
            // Standing detection (refined)
            else if (hipKneeVerticalDiff > 80 && kneeAngleDegrees > 160) {
                position = 'Standing';
                confidence = 90;
            }
            // Squatting detection (refined)  
            else if (hipKneeVerticalDiff < 30 && kneeAngleDegrees < 90) {
                position = 'Squatting';
                confidence = 85;
            }
            // Partial squat
            else if (hipKneeVerticalDiff < 60 && kneeAngleDegrees < 130) {
                position = 'Partial Squat';
                confidence = 75;
            }
            // Leaning/Bending
            else if (hipShoulderVerticalDiff > -10 && hipKneeVerticalDiff > 60) {
                position = 'Leaning/Bending';
                confidence = 70;
            }
            // Transitioning
            else {
                position = 'Transitioning';
                confidence = 60;
            }
            
            // Update debug information with more details
            document.getElementById('position-debug').textContent = 
                `Hip-Knee: ${Math.round(hipKneeVerticalDiff)} | Angle: ${Math.round(kneeAngleDegrees)}° | Hip-Shoulder: ${Math.round(hipShoulderVerticalDiff)}`;
        }
        // Fallback to original logic if ankle/shoulder data unavailable
        else {
            if (hipKneeVerticalDiff > 80) {
                position = 'Standing';
                confidence = 90;
            } else if (hipKneeVerticalDiff < 30) {
                position = 'Squatting';
                confidence = 85;
            } else if (hipKneeVerticalDiff < 60) {
                position = 'Partial Squat';
                confidence = 75;
            } else {
                position = 'Transitioning';
                confidence = 60;
            }
            
            document.getElementById('position-debug').textContent = 
                `Hip Y: ${Math.round(avgHip.y)} | Knee Y: ${Math.round(avgKnee.y)} | Diff: ${Math.round(hipKneeVerticalDiff)}`;
        }
    }
    
    document.getElementById('body-position').textContent = position;
    document.getElementById('position-confidence').textContent = Math.round(confidence) + '%';
    
    // Update position metric styling based on posture quality
    const positionMetric = document.getElementById('position-metric');
    positionMetric.className = 'metric';
    
    if (position.includes('Slouched')) {
        positionMetric.classList.add('warning');
    } else if (position === 'Sitting - Upright' || position === 'Standing') {
        positionMetric.classList.add('good');
    }
}
// Helper function to calculate knee angle
function calculateKneeAngle(hip, knee, ankle) {
    // Vector from knee to hip
    const hipVector = {
        x: hip.x - knee.x,
        y: hip.y - knee.y
    };
    
    // Vector from knee to ankle
    const ankleVector = {
        x: ankle.x - knee.x,
        y: ankle.y - knee.y
    };
    
    // Calculate angle between vectors
    const dotProduct = hipVector.x * ankleVector.x + hipVector.y * ankleVector.y;
    const hipMagnitude = Math.sqrt(hipVector.x ** 2 + hipVector.y ** 2);
    const ankleMagnitude = Math.sqrt(ankleVector.x ** 2 + ankleVector.y ** 2);
    
    if (hipMagnitude === 0 || ankleMagnitude === 0) return 0;
    
    const cosAngle = dotProduct / (hipMagnitude * ankleMagnitude);
    const angleRadians = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    
    return angleRadians * (180 / Math.PI);
}


// Enhanced sitting posture analysis
// Add this new function to your script.js

function analyzeSittingPosture(keypoints) {
    const nose = keypoints[0];
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    const leftKnee = keypoints[13];
    const rightKnee = keypoints[14];
    
    if (!nose || !leftShoulder || !rightShoulder || !leftHip || !rightHip) {
        updatePostureDisplay('Unknown Sitting Posture', 'Need clearer view of upper body', 'warning');
        return;
    }
    
    const shoulderCenter = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2
    };
    
    const hipCenter = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
    };
    
    // Calculate sitting-specific metrics
    const headForwardDistance = nose.x - shoulderCenter.x;
    const shoulderHipAlignment = Math.abs(shoulderCenter.x - hipCenter.x);
    const backAngle = Math.atan2(shoulderCenter.x - hipCenter.x, hipCenter.y - shoulderCenter.y) * 180 / Math.PI;
    const shoulderSlope = Math.abs(leftShoulder.y - rightShoulder.y);
    
    // Check if shoulders are rounded forward (common in sitting)
    const shoulderRounding = shoulderCenter.x > hipCenter.x + 20;
    
    // Determine sitting posture quality
    let sittingPosture = 'Good Sitting Posture';
    let sittingAdvice = 'Excellent sitting posture maintained';
    let sittingLevel = 'good';
    
    const issues = [];
    
    // Forward head posture (more lenient for sitting)
    if (Math.abs(headForwardDistance) > 50) {
        issues.push('forward head posture');
    }
    
    // Back not straight (shoulders not over hips)
    if (shoulderHipAlignment > 40) {
        issues.push('slouched back');
    }
    
    // Rounded shoulders
    if (shoulderRounding) {
        issues.push('rounded shoulders');
    }
    
    // Leaning to one side
    if (Math.abs(backAngle) > 20) {
        issues.push('leaning to one side');
    }
    
    // Uneven shoulders
    if (shoulderSlope > 20) {
        issues.push('uneven shoulders');
    }
    
    // Determine overall sitting posture grade
    if (issues.length > 2) {
        sittingPosture = 'Poor Sitting Posture';
        sittingAdvice = `Multiple issues: ${issues.slice(0, 3).join(', ')}`;
        sittingLevel = 'bad';
    } else if (issues.length > 0) {
        sittingPosture = 'Fair Sitting Posture';
        sittingAdvice = `Address: ${issues.join(', ')}`;
        sittingLevel = 'warning';
    }
    
    // Add specific sitting advice
    if (sittingLevel !== 'good') {
        const sittingTips = [];
        if (issues.includes('forward head posture')) sittingTips.push('chin tuck exercises');
        if (issues.includes('slouched back')) sittingTips.push('sit up straight');
        if (issues.includes('rounded shoulders')) sittingTips.push('shoulder blade squeezes');
        
        if (sittingTips.length > 0) {
            sittingAdvice += ` | Try: ${sittingTips.join(', ')}`;
        }
    }
    
    // Update the posture display with sitting-specific feedback
    updatePostureDisplay(sittingPosture, sittingAdvice, sittingLevel);
}

// Exercise counting - Complete the function
function countSquats(pose) {
    const keypoints = pose.keypoints;
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    const leftKnee = keypoints[13];
    const rightKnee = keypoints[14];
    
    if (!leftHip || !rightHip || !leftKnee || !rightKnee ||
        leftHip.score < 0.4 || rightHip.score < 0.4 || 
        leftKnee.score < 0.4 || rightKnee.score < 0.4) {
        updateSquatIndicator('Waiting for clear pose...', 'squat-transition');
        return;
    }
    
    const avgHipY = (leftHip.y + rightHip.y) / 2;
    const avgKneeY = (leftKnee.y + rightKnee.y) / 2;
    const hipKneeDiff = avgKneeY - avgHipY;
    
    let newState = squatState;
    
    // More reliable state detection
    if (hipKneeDiff < 50) {
        newState = 'down';
    } else if (hipKneeDiff > 90) {
        newState = 'up';
    }
    
    // Update debug info
    document.getElementById('squat-debug').textContent = 
        `State: ${newState} | Hip: ${Math.round(avgHipY)} | Knee: ${Math.round(avgKneeY)} | Diff: ${Math.round(hipKneeDiff)}`;
    
    // State change detection with stability check
    if (newState !== squatState) {
        if (stateFrameCount < minStateFrames) {
            stateFrameCount++;
            return; // Wait for stability
        }
        
        // State confirmed, check for squat completion
        if (squatState === 'down' && newState === 'up') {
            const timeSinceLastSquat = Date.now() - lastSquatTime;
            if (timeSinceLastSquat > 1500) { // Prevent double counting
                exerciseCount++;
                lastSquatTime = Date.now();
                document.getElementById('exercise-count').textContent = exerciseCount;
                updateSquatIndicator('Squat completed! 🎉', 'squat-up');
                
                // Visual feedback
                const counter = document.getElementById('exercise-count');
                counter.style.transform = 'scale(1.2)';
                counter.style.color = '#4CAF50';
                setTimeout(() => {
                    counter.style.transform = 'scale(1)';
                    counter.style.color = 'inherit';
                }, 300);
            }
        }
        
        // Update state
        squatState = newState;
        stateFrameCount = 0;
    } else {
        stateFrameCount++;
    }
    
    // Update indicator based on current state
    if (squatState === 'down') {
        updateSquatIndicator('In squat position - stand up to count', 'squat-down');
    } else if (squatState === 'up') {
        updateSquatIndicator('Standing - squat down to continue', 'squat-up');
    } else {
        updateSquatIndicator('Moving...', 'squat-transition');
    }
}

// Medical analysis functions
function analyzeMedicalPose(pose) {
    switch (medicalMode) {
        case 'posture':
            assessPosture(pose);
            break;
        case 'fall':
            assessFallRisk(pose);
            break;
        case 'therapy':
            assessTherapySession(pose);
            break;
        case 'scoliosis':
            assessScoliosis(pose);
            break;
    }
}

function assessPosture(pose) {
    const keypoints = pose.keypoints;
    const nose = keypoints[0];
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    
    if (!nose || !leftShoulder || !rightShoulder || !leftHip || !rightHip) return;
    
    // Calculate clinical measurements
    const shoulderSlope = Math.abs(leftShoulder.y - rightShoulder.y);
    const headDeviation = Math.abs(nose.x - (leftShoulder.x + rightShoulder.x) / 2);
    const spineDeviation = Math.abs((leftShoulder.x + rightShoulder.x) / 2 - (leftHip.x + rightHip.x) / 2);
    
    // Calculate overall posture score (0-100)
    let score = 100;
    score -= Math.min(shoulderSlope * 0.5, 25);
    score -= Math.min(headDeviation * 0.3, 25);
    score -= Math.min(spineDeviation * 0.4, 25);
    score = Math.max(0, Math.round(score));
    
    // Determine clinical grade
    let grade = 'A';
    if (score < 90) grade = 'B';
    if (score < 75) grade = 'C';
    if (score < 60) grade = 'D';
    if (score < 45) grade = 'F';
    
    // Update displays
    document.getElementById('posture-score').querySelector('.large-number').textContent = score;
    document.getElementById('clinical-grade').querySelector('div').textContent = grade;
    
    // Clinical assessment text
    let assessment = `Posture Score: ${score}/100 (Grade ${grade})`;
    if (shoulderSlope > 20) assessment += '\n• Significant shoulder asymmetry detected';
    if (headDeviation > 25) assessment += '\n• Forward head posture present';
    if (spineDeviation > 30) assessment += '\n• Lateral spine deviation noted';
    
    document.getElementById('posture-clinical').textContent = assessment;
    
    // Store assessment
    postureAssessments.push({
        timestamp: Date.now(),
        score: score,
        grade: grade,
        shoulderSlope: shoulderSlope,
        headDeviation: headDeviation,
        spineDeviation: spineDeviation
    });
}

function assessFallRisk(pose) {
    const keypoints = pose.keypoints;
    const leftAnkle = keypoints[15];
    const rightAnkle = keypoints[16];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    
    if (!leftAnkle || !rightAnkle || !leftHip || !rightHip) return;
    
    // Calculate center of mass and base of support
    const com = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
    };
    
    const baseCenter = {
        x: (leftAnkle.x + rightAnkle.x) / 2,
        y: (leftAnkle.y + rightAnkle.y) / 2
    };
    
    // Calculate sway and stability metrics
    const lateralSway = Math.abs(com.x - baseCenter.x);
    const baseWidth = Math.abs(leftAnkle.x - rightAnkle.x);
    const stability = baseWidth > 0 ? Math.min(baseWidth / lateralSway, 2) : 0;
    
    // Calculate fall risk percentage
    let fallRisk = 0;
    if (lateralSway > 20) fallRisk += 30;
    if (stability < 1) fallRisk += 25;
    if (baseWidth < 50) fallRisk += 20;
    
    fallRisk = Math.min(100, fallRisk);
    
    // Balance score (inverse of fall risk)
    const balanceScore = Math.max(0, 100 - fallRisk);
    
    // Update displays
    document.getElementById('fall-risk-score').querySelector('.large-number').textContent = fallRisk;
    document.getElementById('balance-score').querySelector('.large-number').textContent = balanceScore;
    
    // Update risk level styling
    const riskDisplay = document.getElementById('fall-risk-score');
    riskDisplay.className = 'score-display';
    if (fallRisk > 60) riskDisplay.classList.add('risk-high');
    else if (fallRisk > 30) riskDisplay.classList.add('risk-medium');
    else riskDisplay.classList.add('risk-low');
    
    // Store assessment
    fallRiskHistory.push({
        timestamp: Date.now(),
        fallRisk: fallRisk,
        balanceScore: balanceScore,
        lateralSway: lateralSway,
        stability: stability
    });
}

function assessTherapySession(pose) {
    if (!exerciseSession) return;
    
    // Track exercise accuracy based on form
    const keypoints = pose.keypoints;
    let formAccuracy = 80; // Base accuracy
    
    // Analyze form quality (simplified)
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    
    if (leftShoulder && rightShoulder) {
        const shoulderLevel = Math.abs(leftShoulder.y - rightShoulder.y);
        if (shoulderLevel > 30) formAccuracy -= 15;
    }
    
    // Update PT session display
    document.getElementById('pt-exercises').textContent = exerciseCount;
    document.getElementById('pt-accuracy').textContent = Math.round(formAccuracy) + '%';
    
    exerciseSession.accuracy = formAccuracy;
    exerciseSession.exercises = exerciseCount;
}

function assessScoliosis(pose) {
    const keypoints = pose.keypoints;
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return;
    
    // Calculate Cobb angle approximation
    const shoulderAngle = Math.atan2(rightShoulder.y - leftShoulder.y, rightShoulder.x - leftShoulder.x) * 180 / Math.PI;
    const hipAngle = Math.atan2(rightHip.y - leftHip.y, rightHip.x - leftHip.x) * 180 / Math.PI;
    const cobbAngle = Math.abs(shoulderAngle - hipAngle);
    
    // Calculate trunk rotation
    const shoulderCenter = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
    const hipCenter = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2 };
    const trunkRotation = Math.atan2(hipCenter.x - shoulderCenter.x, shoulderCenter.y - hipCenter.y) * 180 / Math.PI;
    
    // Calculate shoulder level difference
    const shoulderLevelDiff = Math.abs(leftShoulder.y - rightShoulder.y);
    
    // Update displays
    document.getElementById('cobb-angle').textContent = Math.round(cobbAngle) + '°';
    document.getElementById('trunk-rotation').textContent = Math.round(Math.abs(trunkRotation)) + '°';
    document.getElementById('shoulder-level').textContent = Math.round(shoulderLevelDiff) + 'px';
    
    // Alert for significant findings
    let alertText = '';
    if (cobbAngle > 10) alertText = `Potential scoliosis detected (Cobb: ${Math.round(cobbAngle)}°)`;
    else if (shoulderLevelDiff > 20) alertText = 'Shoulder asymmetry detected';
    else alertText = 'No significant findings';
    
    document.getElementById('scoliosis-alert').textContent = alertText;
}

// UI Helper functions
function updatePostureDisplay(status, advice, level) {
    const indicator = document.querySelector('.posture-indicator');
    const statusEl = document.getElementById('posture-status');
    const adviceEl = document.getElementById('posture-advice');
    
    indicator.className = 'posture-indicator ' + level;
    statusEl.textContent = status;
    adviceEl.textContent = advice;
}

function updateSquatIndicator(message, className) {
    const indicator = document.getElementById('squat-indicator');
    indicator.textContent = message;
    indicator.className = 'squat-indicator ' + className;
}

function updateStats(pose) {
    frameCount++;
    poseCount++;
    totalConfidence += pose.score || 0.5;
    
    document.getElementById('pose-count').textContent = poseCount;
    document.getElementById('avg-confidence').textContent = 
        Math.round((totalConfidence / poseCount) * 100) + '%';
}

function updateFPS() {
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
        const fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
        document.getElementById('fps-display').textContent = fps;
        frameCount = 0;
        lastFpsTime = now;
    }
}

// Control functions
function resetCounters() {
    exerciseCount = 0;
    poseCount = 0;
    totalConfidence = 0;
    frameCount = 0;
    
    document.getElementById('exercise-count').textContent = '0';
    document.getElementById('pose-count').textContent = '0';
    document.getElementById('avg-confidence').textContent = '0%';
    document.getElementById('fps-display').textContent = '0';
    
    log('Counters reset', 'info');
}

function toggleExerciseMode() {
    exerciseMode = !exerciseMode;
    const btn = document.getElementById('exerciseBtn');
    
    if (exerciseMode) {
        btn.textContent = 'Disable Counting';
        btn.style.background = '#f44336';
        log('Exercise counting enabled', 'success');
    } else {
        btn.textContent = 'Enable Counting';
        btn.style.background = '#4CAF50';
        log('Exercise counting disabled', 'info');
    }
}

// Medical panel functions
function switchMedicalTab(mode) {
    medicalMode = mode;
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    // Find and activate the clicked tab
    const clickedTab = Array.from(document.querySelectorAll('.tab-btn')).find(btn => 
        btn.textContent.toLowerCase().includes(mode.toLowerCase()) ||
        btn.onclick.toString().includes(mode)
    );
    if (clickedTab) {
        clickedTab.classList.add('active');
    }
    
    // Update content panels
    document.querySelectorAll('.medical-content').forEach(content => content.classList.remove('active'));
    const targetPanel = document.getElementById(mode + '-medical');
    if (targetPanel) {
        targetPanel.classList.add('active');
    }
    
    log(`Switched to ${mode} assessment mode`, 'info');
}
// Fix for medical tab switching
function setupMedicalTabs() {
    const tabs = document.querySelectorAll('.medical-tabs .tab-btn');
    tabs.forEach((tab, index) => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            const modes = ['posture', 'fall', 'therapy', 'scoliosis'];
            if (modes[index]) {
                switchMedicalTab(modes[index]);
            }
        });
    });
}

function generatePostureReport() {
    if (postureAssessments.length === 0) {
        alert('No posture data available. Start detection first.');
        return;
    }
    
    const latest = postureAssessments[postureAssessments.length - 1];
    const report = `
POSTURE ASSESSMENT REPORT
========================
Date: ${new Date().toLocaleDateString()}
Patient: ${currentPatientData.name}

MEASUREMENTS:
- Overall Score: ${latest.score}/100 (Grade ${latest.grade})
- Shoulder Asymmetry: ${latest.shoulderSlope.toFixed(1)}px
- Head Deviation: ${latest.headDeviation.toFixed(1)}px  
- Spine Deviation: ${latest.spineDeviation.toFixed(1)}px

RECOMMENDATIONS:
${latest.score > 85 ? '• Maintain current posture habits' : ''}
${latest.shoulderSlope > 20 ? '• Consider shoulder alignment exercises' : ''}
${latest.headDeviation > 25 ? '• Practice chin tuck exercises' : ''}
${latest.spineDeviation > 30 ? '• Core strengthening recommended' : ''}

Session Duration: ${document.getElementById('session-time').textContent}
Total Poses Analyzed: ${poseCount}
    `;
    
    // Create downloadable report
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `posture_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    log('Posture report generated', 'success');
}

function startBalanceTest() {
    if (balanceTestActive) {
        alert('Balance test already in progress');
        return;
    }
    
    balanceTestActive = true;
    balanceTestStartTime = Date.now();
    
    log('Balance test started - maintain position for 30 seconds', 'info');
    
    // Progress bar animation
    const progressBar = document.getElementById('balance-progress');
    progressBar.style.width = '0%';
    
    const interval = setInterval(() => {
        const elapsed = Date.now() - balanceTestStartTime;
        const progress = Math.min((elapsed / balanceTestDuration) * 100, 100);
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(interval);
            balanceTestActive = false;
            log('Balance test completed!', 'success');
        }
    }, 100);
}

function updateBalanceTest(pose) {
    if (!balanceTestActive) return;
    
    const elapsed = Date.now() - balanceTestStartTime;
    const remaining = Math.max(0, (balanceTestDuration - elapsed) / 1000);
    
    if (remaining <= 0) {
        balanceTestActive = false;
        return;
    }
    
    // Add balance stability analysis during test
    assessFallRisk(pose);
}

function startPTSession() {
    if (exerciseSession) {
        alert('PT session already in progress');
        return;
    }
    
    exerciseSession = {
        startTime: Date.now(),
        exercises: 0,
        accuracy: 0,
        notes: []
    };
    
    // Enable exercise mode automatically
    if (!exerciseMode) {
        toggleExerciseMode();
    }
    
    // Switch to therapy medical mode
    medicalMode = 'therapy';
    switchMedicalTab('therapy');
    
    document.getElementById('pt-notes').textContent = 'PT session active - tracking exercises and form';
    log('Physical therapy session started', 'success');
}

function endPTSession() {
    if (!exerciseSession) {
        alert('No active PT session');
        return;
    }
    
    const sessionDuration = Math.floor((Date.now() - exerciseSession.startTime) / 1000);
    const minutes = Math.floor(sessionDuration / 60);
    const seconds = sessionDuration % 60;
    
    const report = `
PHYSICAL THERAPY SESSION REPORT
==============================
Date: ${new Date().toLocaleDateString()}
Patient: ${currentPatientData.name}
Duration: ${minutes}:${seconds.toString().padStart(2, '0')}

PERFORMANCE METRICS:
- Exercises Completed: ${exerciseSession.exercises}
- Average Form Accuracy: ${Math.round(exerciseSession.accuracy)}%
- Total Poses Analyzed: ${poseCount}

SESSION NOTES:
- Session completed successfully
- ${exerciseSession.exercises > 10 ? 'Good exercise volume achieved' : 'Consider increasing exercise volume'}
- ${exerciseSession.accuracy > 80 ? 'Excellent form maintained' : 'Focus on form improvement'}

RECOMMENDATIONS:
- Continue current exercise regimen
- Monitor form quality in future sessions
- ${exerciseSession.exercises < 5 ? 'Gradually increase repetitions' : 'Maintain current intensity'}
    `;
    
    // Create downloadable report
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pt_session_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    // Reset session
    exerciseSession = null;
    document.getElementById('pt-notes').textContent = 'Session completed and documented';
    
    log('PT session completed and documented', 'success');
}

function startScoliosisScreen() {
    medicalMode = 'scoliosis';
    switchMedicalTab('scoliosis');
    
    document.getElementById('scoliosis-alert').textContent = 'Scoliosis screening active - analyzing spinal alignment';
    log('Scoliosis screening started', 'success');
}

function captureReading() {
    if (medicalMode !== 'scoliosis') {
        alert('Switch to scoliosis screening first');
        return;
    }
    
    const cobbAngle = parseFloat(document.getElementById('cobb-angle').textContent);
    const trunkRotation = parseFloat(document.getElementById('trunk-rotation').textContent);
    const shoulderLevel = parseFloat(document.getElementById('shoulder-level').textContent);
    
    const reading = {
        timestamp: Date.now(),
        cobbAngle: cobbAngle,
        trunkRotation: trunkRotation,
        shoulderLevel: shoulderLevel
    };
    
    scoliosisReadings.push(reading);
    
    document.getElementById('scoliosis-alert').textContent = 
        `Reading captured: Cobb ${cobbAngle}°, Rotation ${trunkRotation}°, Shoulder ${shoulderLevel}px`;
    
    log(`Scoliosis reading captured (#${scoliosisReadings.length})`, 'success');
}

// Patient data management
function setPatientData(name, age, conditions = []) {
    currentPatientData = {
        name: name,
        age: age,
        conditions: conditions,
        lastAssessment: Date.now()
    };
    
    log(`Patient data updated: ${name}, Age ${age}`, 'info');
}

// Advanced analysis functions
function calculateMovementQuality(pose) {
    const keypoints = pose.keypoints;
    let qualityScore = 100;
    
    // Check joint alignment
    const leftShoulder = keypoints[5];
    const leftElbow = keypoints[7];
    const leftWrist = keypoints[9];
    
    if (leftShoulder && leftElbow && leftWrist) {
        // Calculate arm alignment
        const upperArmAngle = Math.atan2(leftElbow.y - leftShoulder.y, leftElbow.x - leftShoulder.x);
        const forearmAngle = Math.atan2(leftWrist.y - leftElbow.y, leftWrist.x - leftElbow.x);
        const jointAngle = Math.abs(upperArmAngle - forearmAngle) * 180 / Math.PI;
        
        // Penalize extreme angles
        if (jointAngle < 30 || jointAngle > 150) {
            qualityScore -= 15;
        }
    }
    
    return qualityScore;
}

function detectMovementPatterns(pose) {
    const keypoints = pose.keypoints;
    const patterns = [];
    
    // Detect reaching pattern
    const leftWrist = keypoints[9];
    const rightWrist = keypoints[10];
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    
    if (leftWrist && leftShoulder) {
        const reachDistance = Math.sqrt(
            Math.pow(leftWrist.x - leftShoulder.x, 2) + 
            Math.pow(leftWrist.y - leftShoulder.y, 2)
        );
        
        if (reachDistance > 150) {
            patterns.push('Extended Reach');
        }
    }
    
    return patterns;
}

// Rehabilitation-specific functions
function assessRangeOfMotion(pose) {
    const keypoints = pose.keypoints;
    const assessments = {};
    
    // Shoulder ROM
    const leftShoulder = keypoints[5];
    const leftElbow = keypoints[7];
    
    if (leftShoulder && leftElbow) {
        const shoulderAngle = Math.atan2(
            leftElbow.y - leftShoulder.y, 
            leftElbow.x - leftShoulder.x
        ) * 180 / Math.PI;
        
        assessments.shoulderFlexion = Math.abs(shoulderAngle);
    }
    
    // Hip ROM
    const leftHip = keypoints[11];
    const leftKnee = keypoints[13];
    
    if (leftHip && leftKnee) {
        const hipAngle = Math.atan2(
            leftKnee.y - leftHip.y, 
            leftKnee.x - leftHip.x
        ) * 180 / Math.PI;
        
        assessments.hipFlexion = Math.abs(hipAngle);
    }
    
    return assessments;
}

function monitorPainIndicators(pose) {
    const keypoints = pose.keypoints;
    const indicators = [];
    
    // Check for guarding postures
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    
    if (leftShoulder && rightShoulder) {
        const shoulderAsymmetry = Math.abs(leftShoulder.y - rightShoulder.y);
        
        if (shoulderAsymmetry > 30) {
            indicators.push('Shoulder guarding detected');
        }
    }
    
    // Check for compensation patterns
    const nose = keypoints[0];
    const shoulderCenter = leftShoulder && rightShoulder ? 
        { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 } : null;
    
    if (nose && shoulderCenter) {
        const headTilt = Math.abs(nose.x - shoulderCenter.x);
        if (headTilt > 40) {
            indicators.push('Head compensation pattern');
        }
    }
    
    return indicators;
}

// Export/Import functions
function exportSessionData() {
    const sessionData = {
        timestamp: Date.now(),
        duration: document.getElementById('session-time').textContent,
        patient: currentPatientData,
        stats: {
            poseCount: poseCount,
            exerciseCount: exerciseCount,
            avgConfidence: Math.round((totalConfidence / poseCount) * 100)
        },
        assessments: {
            posture: postureAssessments,
            fallRisk: fallRiskHistory,
            scoliosis: scoliosisReadings
        },
        session: exerciseSession
    };
    
    const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    log('Session data exported', 'success');
}

// Error handling and recovery
function handleDetectionError(error) {
    console.error('Detection error:', error);
    
    // Attempt recovery
    if (error.message.includes('model')) {
        log('Model error detected, attempting reload...', 'error');
        setTimeout(() => {
            loadAI();
        }, 2000);
    } else if (error.message.includes('camera')) {
        log('Camera error detected, check camera permissions', 'error');
    } else {
        log('Unknown error: ' + error.message, 'error');
    }
}

// Performance monitoring
function monitorPerformance() {
    const memoryInfo = performance.memory;
    if (memoryInfo) {
        const memoryUsage = Math.round(memoryInfo.usedJSHeapSize / 1024 / 1024);
        console.log(`Memory usage: ${memoryUsage}MB`);
        
        // Warn if memory usage is high
        if (memoryUsage > 200) {
            log('High memory usage detected', 'warning');
        }
    }
}

// Initialize performance monitoring
setInterval(monitorPerformance, 30000); // Check every 30 seconds

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
            case 's':
                event.preventDefault();
                if (isDetecting) {
                    stopDetection();
                } else if (detector) {
                    startDetection();
                }
                break;
            case 'r':
                event.preventDefault();
                resetCounters();
                break;
            case 'e':
                event.preventDefault();
                toggleExerciseMode();
                break;
        }
    }
});

// CSS animation for counter pulse
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); color: #4CAF50; }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(style);

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    log('Medical Pose Detection Suite loaded. Click Start Camera to begin.', 'info');
    
    // Set default patient data
    setPatientData('Anonymous Patient', 0, []);
    
    // Initialize canvas size
    canvas.width = 640;
    canvas.height = 480;
    
    console.log('Application initialized successfully');
});

// Cleanup function
window.addEventListener('beforeunload', () => {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    
    if (sessionTimer) {
        clearInterval(sessionTimer);
    }
});

// To collect samples for training
function captureSample(label, keypoints) {
  const sample = {
    label: label,
    timestamp: Date.now(),
    keypoints: keypoints.map(k => ({
      name: k.name,
      x: k.x,
      y: k.y,
      score: k.score
    }))
  };
  collectedSamples.push(sample);
}
function downloadSamples() {
  const blob = new Blob([collectedSamples.map(s => JSON.stringify(s)).join("\n")],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "samples.jsonl";  // JSON lines format
  a.click();
}
