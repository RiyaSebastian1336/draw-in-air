import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// Global App State
const state = {
  handLandmarker: null,
  webcamStream: null,
  isReady: false,
  strokes: [],
  currentStroke: null,
  activeColor: "#00f0ff",
  thickness: 6,
  glowIntensity: 60,
  currentGesture: "idle",
  previousGesture: "idle",
  gestureStableFrames: 0,
  gestureStartTime: 0,
  isModalOpen: true,
  isGrabbing: false,
  grabStartPos: null,
  grabOffset: { x: 0, y: 0 },
  totalOffset: { x: 0, y: 0 },
  nearestStrokeIdx: -1,
  eraserRadius: 28,
  showCamera: true,
  cameraOpacity: 0.35,
  particles: [],
  smoothPos: { x: 0, y: 0 },
  smoothFactor: 0.35,
  width: 0,
  height: 0,
  audioCtx: null
};

// DOM Elements
const getEl = (id) => document.getElementById(id);
const loadingScreen = getEl("loading-screen");
const appContainer = getEl("app");
const webcamElement = getEl("webcam");
const cameraCanvas = getEl("camera-canvas");
const drawingCanvas = getEl("drawing-canvas");
const uiCanvas = getEl("ui-canvas");

const cameraCtx = cameraCanvas.getContext("2d");
const drawingCtx = drawingCanvas.getContext("2d");
const uiCtx = uiCanvas.getContext("2d");

const gestureHud = getEl("gesture-hud");
const gestureIcon = getEl("gesture-icon");
const gestureLabel = getEl("gesture-label");
const thicknessSlider = getEl("thickness-slider");
const thicknessValue = getEl("thickness-value");
const glowSlider = getEl("glow-slider");
const glowValue = getEl("glow-value");
const cameraModeText = getEl("camera-mode-text");
const cameraModeIndicator = getEl("camera-mode-indicator");
const onboardingModal = getEl("onboarding-modal");
const btnStart = getEl("btn-start");

// Audio synthesizer helper using Web Audio API
function getAudioContext() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return state.audioCtx;
}

function playSound(frequency, duration, type = "sine", volume = 0.06) {
  try {
    const audioCtx = getAudioContext();
    // Resume context if suspended (browser security policies)
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    
    osc.type = type;
    osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
    
    gainNode.gain.setValueAtTime(volume, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    
    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (err) {
    console.warn("Audio feedback block:", err);
  }
}

// Chime presets for various gestures and button clicks
const playDrawStartSound = () => playSound(880, 0.08, "sine", 0.04);
const playDrawEndSound = () => playSound(440, 0.1, "sine", 0.03);
const playEraseContactSound = () => playSound(200, 0.06, "triangle", 0.03);
const playGrabStartSound = () => playSound(660, 0.1, "sine", 0.05);
const playGrabEndSound = () => playSound(330, 0.15, "sine", 0.04);
const playEraseStartSound = () => playSound(1200, 0.05, "sine", 0.03);

// Window resizing
function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  state.width = w;
  state.height = h;
  [cameraCanvas, drawingCanvas, uiCanvas].forEach(c => {
    c.width = w;
    c.height = h;
  });
}
window.addEventListener("resize", () => {
  resizeCanvases();
  redrawDrawingCanvas();
});

// Initialize HandLandmarker vision task
async function initHandTracking() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5
  });
  return true;
}

// Request webcam access and setup stream
async function initWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    }
  });
  webcamElement.srcObject = stream;
  state.webcamStream = stream;
  return new Promise((resolve) => {
    webcamElement.onloadedmetadata = () => {
      webcamElement.play();
      resolve();
    };
  });
}

// Detect and classify hand gesture
function classifyGesture(landmarks) {
  if (!landmarks || landmarks.length === 0) return "none";

  const wrist = landmarks[0];
  const thumbTip = landmarks[4];
  const thumbIp = landmarks[3];
  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];

  // Finger states (extended vs folded)
  const isIndexExtended = indexTip.y < indexPip.y - 0.02;
  const isMiddleFolded = middleTip.y > middlePip.y;
  const isRingFolded = ringTip.y > ringPip.y;
  const isPinkyFolded = pinkyTip.y > pinkyPip.y;

  const isMiddleExtended = middleTip.y < middlePip.y;
  const isRingExtended = ringTip.y < ringPip.y;
  const isPinkyExtended = pinkyTip.y < pinkyPip.y;
  const isThumbExtended = Math.abs(thumbTip.x - thumbIp.x) > 0.03 || thumbTip.y < thumbIp.y;

  // 1. Pinch gesture (Grab & Move): Thumb tip close to index tip, other fingers folded
  const thumbIndexDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  if (thumbIndexDist < 0.06 && !isMiddleExtended && !isRingExtended && !isPinkyExtended) {
    return "pinch";
  }

  // 2. Open palm gesture (Erase): All 4 main fingers extended (making thumb check optional to dramatically improve detection reliability)
  if (isIndexExtended && isMiddleExtended && isRingExtended && isPinkyExtended) {
    return "open_palm";
  }

  // 3. Index finger pointing up (Draw): Index extended, others folded
  if (isIndexExtended && isMiddleFolded && isRingFolded && isPinkyFolded) {
    return "index_finger";
  }

  // 4. Fist gesture (Idle/Rest): All fingers folded
  if (!isIndexExtended && !isMiddleExtended && !isRingExtended && !isPinkyExtended) {
    return "fist";
  }

  return "idle";
}

// Debounce gesture states to prevent flickering
function debounceGesture(newGesture) {
  if (newGesture === state.currentGesture) {
    state.previousGesture = newGesture;
    state.gestureStableFrames = 0;
    return state.currentGesture;
  }

  if (newGesture === state.previousGesture) {
    state.gestureStableFrames++;
  } else {
    state.previousGesture = newGesture;
    state.gestureStableFrames = 1;
  }

  // Adjust frame delay for specific gestures (pinch needs faster response)
  const requiredFrames = (newGesture === "pinch") ? 3 : 4;

  if (state.gestureStableFrames >= requiredFrames) {
    const oldGesture = state.currentGesture;
    state.currentGesture = newGesture;
    state.gestureStableFrames = 0;
    state.gestureStartTime = Date.now();
    
    if (oldGesture !== newGesture) {
      handleGestureTransition(oldGesture, newGesture);
    }
    return newGesture;
  }

  return state.currentGesture;
}

// Handle transition triggers
function handleGestureTransition(oldGesture, newGesture) {
  if (newGesture === "index_finger") {
    playDrawStartSound();
  } else if (newGesture === "open_palm") {
    playEraseStartSound();
  } else if (newGesture === "pinch") {
    playGrabStartSound();
  } else if (oldGesture === "index_finger") {
    playDrawEndSound();
  }

  if (oldGesture === "index_finger" && state.currentStroke) {
    // Save the drawn stroke
    if (state.currentStroke.points.length > 1) {
      state.strokes.push({ ...state.currentStroke });
    }
    state.currentStroke = null;
  }

  if (oldGesture === "pinch") {
    handleReleaseGrab();
  }

  updateHudUI(newGesture);
}

// Update the HUD elements with respective status details
function updateHudUI(gesture) {
  const configs = {
    index_finger: { icon: "☝️", label: "Drawing", cls: "drawing" },
    open_palm: { icon: "✋", label: "Erasing", cls: "erasing" },
    pinch: { icon: "🤏", label: "Grab", cls: "grabbing" },
    fist: { icon: "✊", label: "Idle", cls: "" },
    idle: { icon: "🖐️", label: "Ready", cls: "" },
    none: { icon: "👋", label: "Show hand", cls: "" }
  };

  const currentConfig = configs[gesture] || configs.idle;
  gestureIcon.textContent = currentConfig.icon;
  gestureLabel.textContent = currentConfig.label;
  gestureHud.className = currentConfig.cls;
}

// Coordinate converters (MediaPipe returns 0-1 values, mirror index x coordinates)
function toCanvasCoordinates(pt) {
  return {
    x: (1 - pt.x) * state.width,
    y: pt.y * state.height
  };
}

// Exponential moving average for drawing smoothness
function smoothCoordinates(targetPt) {
  state.smoothPos.x += (targetPt.x - state.smoothPos.x) * state.smoothFactor;
  state.smoothPos.y += (targetPt.y - state.smoothPos.y) * state.smoothFactor;
  return { x: state.smoothPos.x, y: state.smoothPos.y };
}

// DRAW GESTURE IMPLEMENTATION
function handleDrawGesture(landmarks) {
  const indexTip = landmarks[8];
  const screenPos = toCanvasCoordinates(indexTip);
  const smoothed = smoothCoordinates(screenPos);

  // If we just transitioned, snap smooth coordinates to avoid cursor drag
  if (Date.now() - state.gestureStartTime < 300) {
    state.smoothPos = { ...screenPos };
    return;
  }

  if (state.currentStroke) {
    state.currentStroke.points.push({ ...smoothed });
  } else {
    state.currentStroke = {
      points: [smoothed],
      color: state.activeColor,
      thickness: state.thickness,
      glow: state.glowIntensity
    };
    state.smoothPos = { ...screenPos };
  }

  spawnDrawingParticles(smoothed.x, smoothed.y, state.activeColor);
  redrawDrawingCanvas();
}

// Calculate Euclidean distance from point p to line segment ab
function getDistanceToSegment(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const l2 = dx * dx + dy * dy;
  if (l2 === 0) return Math.hypot(p.x - a.x, p.y - a.y);
  
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2;
  t = Math.max(0, Math.min(1, t)); // clamp projection to segment bounds
  
  const closestX = a.x + t * dx;
  const closestY = a.y + t * dy;
  return Math.hypot(p.x - closestX, p.y - closestY);
}

// ERASE GESTURE IMPLEMENTATION
function handleEraseGesture(landmarks) {
  const wrist = landmarks[0];
  const middleBase = landmarks[9];
  
  // Calculate palm center
  const palmCenter = {
    x: (1 - (wrist.x + middleBase.x) / 2) * state.width,
    y: ((wrist.y + middleBase.y) / 2) * state.height
  };
  const r = state.eraserRadius;
  let erasedAny = false;
  const filteredStrokes = [];

  for (const stroke of state.strokes) {
    if (stroke.points.length === 0) continue;
    if (stroke.points.length === 1) {
      const pt = stroke.points[0];
      const dist = Math.hypot(pt.x - palmCenter.x, pt.y - palmCenter.y);
      if (dist >= r) {
        filteredStrokes.push(stroke);
      } else {
        erasedAny = true;
      }
      continue;
    }

    // Process stroke segments and split if any segment passes through the eraser area
    let currentGroup = [];
    for (let i = 0; i < stroke.points.length; i++) {
      const pt = stroke.points[i];
      const distToPt = Math.hypot(pt.x - palmCenter.x, pt.y - palmCenter.y);

      if (distToPt < r) {
        // Point itself is in eraser zone, split
        erasedAny = true;
        if (currentGroup.length >= 2) {
          filteredStrokes.push({
            points: currentGroup,
            color: stroke.color,
            thickness: stroke.thickness,
            glow: stroke.glow
          });
        }
        currentGroup = [];
      } else {
        currentGroup.push(pt);
        
        // Check segment connecting to next point
        if (i < stroke.points.length - 1) {
          const nextPt = stroke.points[i + 1];
          const distToSegment = getDistanceToSegment(palmCenter, pt, nextPt);
          
          if (distToSegment < r) {
            // Line intersects eraser, cut the path here
            erasedAny = true;
            if (currentGroup.length >= 2) {
              filteredStrokes.push({
                points: currentGroup,
                color: stroke.color,
                thickness: stroke.thickness,
                glow: stroke.glow
              });
            }
            currentGroup = [];
          }
        }
      }
    }

    if (currentGroup.length >= 2) {
      filteredStrokes.push({
        points: currentGroup,
        color: stroke.color,
        thickness: stroke.thickness,
        glow: stroke.glow
      });
    }
  }

  state.strokes = filteredStrokes;
  if (erasedAny) {
    playEraseContactSound();
  }

  // Draw circular indicator of eraser on UI canvas
  uiCtx.beginPath();
  uiCtx.arc(palmCenter.x, palmCenter.y, r, 0, Math.PI * 2);
  uiCtx.strokeStyle = "rgba(255, 45, 107, 0.5)";
  uiCtx.lineWidth = 1.5;
  uiCtx.setLineDash([5, 5]);
  uiCtx.stroke();
  uiCtx.setLineDash([]);
  uiCtx.fillStyle = "rgba(255, 45, 107, 0.05)";
  uiCtx.fill();

  redrawDrawingCanvas();
}

// MOVE GESTURE IMPLEMENTATION
function handleMoveGesture(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  
  // Midpoint between index and thumb tip represents the pinch point
  const pinchPos = {
    x: (1 - (thumbTip.x + indexTip.x) / 2) * state.width,
    y: ((thumbTip.y + indexTip.y) / 2) * state.height
  };

  if (!state.isGrabbing) {
    state.isGrabbing = true;
    state.grabStartPos = { ...pinchPos };
    state.nearestStrokeIdx = findNearestStrokeIndex(pinchPos);
  } else {
    const deltaX = pinchPos.x - state.grabStartPos.x;
    const deltaY = pinchPos.y - state.grabStartPos.y;

    if (state.nearestStrokeIdx >= 0 && state.nearestStrokeIdx < state.strokes.length) {
      const stroke = state.strokes[state.nearestStrokeIdx];
      const prevOffsetX = state.grabOffset.x;
      const prevOffsetY = state.grabOffset.y;
      
      const stepX = deltaX - prevOffsetX;
      const stepY = deltaY - prevOffsetY;

      // Shift entire stroke by the grab displacement
      for (let i = 0; i < stroke.points.length; i++) {
        stroke.points[i].x += stepX;
        stroke.points[i].y += stepY;
      }
      
      state.grabOffset = { x: deltaX, y: deltaY };
    }
  }

  // Draw visual feedback for grab
  uiCtx.beginPath();
  uiCtx.arc(pinchPos.x, pinchPos.y, 18, 0, Math.PI * 2);
  uiCtx.strokeStyle = "rgba(255, 215, 0, 0.7)";
  uiCtx.lineWidth = 2;
  uiCtx.stroke();
  uiCtx.fillStyle = "rgba(255, 215, 0, 0.1)";
  uiCtx.fill();

  if (state.nearestStrokeIdx >= 0 && state.nearestStrokeIdx < state.strokes.length) {
    highlightStroke(state.strokes[state.nearestStrokeIdx]);
  }

  redrawDrawingCanvas();
}

function handleReleaseGrab() {
  if (state.isGrabbing && state.nearestStrokeIdx >= 0) {
    playGrabEndSound();
  }
  state.isGrabbing = false;
  state.grabStartPos = null;
  state.grabOffset = { x: 0, y: 0 };
  state.nearestStrokeIdx = -1;
  redrawDrawingCanvas();
}

// Find closest drawing stroke to reposition it
function findNearestStrokeIndex(pinchPos) {
  let minDist = Infinity;
  let idx = -1;
  for (let i = 0; i < state.strokes.length; i++) {
    const stroke = state.strokes[i];
    for (const pt of stroke.points) {
      const dist = Math.hypot(pt.x - pinchPos.x, pt.y - pinchPos.y);
      if (dist < minDist) {
        minDist = dist;
        idx = i;
      }
    }
  }
  return minDist < 80 ? idx : -1;
}

// Highlights selected stroke with dashed borders
function highlightStroke(stroke) {
  if (!stroke || stroke.points.length < 2) return;
  uiCtx.save();
  uiCtx.beginPath();
  uiCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
  for (let i = 1; i < stroke.points.length; i++) {
    uiCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
  }
  uiCtx.strokeStyle = "rgba(255, 215, 0, 0.3)";
  uiCtx.lineWidth = stroke.thickness + 12;
  uiCtx.lineCap = "round";
  uiCtx.lineJoin = "round";
  uiCtx.setLineDash([8, 8]);
  uiCtx.stroke();
  uiCtx.setLineDash([]);
  uiCtx.restore();
}

// RENDER STROKE LOGIC (Glowing effects + cubic curve interpolation)
function renderStroke(ctx, stroke) {
  if (!stroke || stroke.points.length < 2) return;

  const pts = stroke.points;
  const col = stroke.color;
  const width = stroke.thickness;
  const glow = stroke.glow / 100;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Layer 1: Ambient outer glow
  if (glow > 0) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const p1 = pts[i - 1];
      const p2 = pts[i];
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2;
      ctx.quadraticCurveTo(p1.x, p1.y, midX, midY);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    ctx.strokeStyle = col;
    ctx.lineWidth = width * 3;
    ctx.globalAlpha = 0.1 * glow;
    ctx.shadowColor = col;
    ctx.shadowBlur = 35 * glow;
    ctx.stroke();
  }

  // Layer 2: Intended inner neon glow
  if (glow > 0) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const p1 = pts[i - 1];
      const p2 = pts[i];
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2;
      ctx.quadraticCurveTo(p1.x, p1.y, midX, midY);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    ctx.strokeStyle = col;
    ctx.lineWidth = width * 1.6;
    ctx.globalAlpha = 0.35 * glow;
    ctx.shadowBlur = 15 * glow;
    ctx.stroke();
  }

  // Layer 3: Solid white-hot core
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) {
    const p1 = pts[i - 1];
    const p2 = pts[i];
    const midX = (p1.x + p2.x) / 2;
    const midY = (p1.y + p2.y) / 2;
    ctx.quadraticCurveTo(p1.x, p1.y, midX, midY);
  }
  ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  ctx.strokeStyle = getCoreColor(col, 0.5);
  ctx.lineWidth = width;
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 6 * glow;
  ctx.shadowColor = col;
  ctx.stroke();

  ctx.restore();
}

// Blend color with white to create glowing light core
function getCoreColor(hex, blendRatio) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);

  const blendedR = Math.min(255, Math.round(r + (255 - r) * blendRatio));
  const blendedG = Math.min(255, Math.round(g + (255 - g) * blendRatio));
  const blendedB = Math.min(255, Math.round(b + (255 - b) * blendRatio));

  return `rgb(${blendedR}, ${blendedG}, ${blendedB})`;
}

// Draw all current strokes on drawing canvas
function redrawDrawingCanvas() {
  drawingCtx.clearRect(0, 0, state.width, state.height);
  for (const stroke of state.strokes) {
    renderStroke(drawingCtx, stroke);
  }
  if (state.currentStroke && state.currentStroke.points.length > 1) {
    renderStroke(drawingCtx, state.currentStroke);
  }
}

// PARTICLE FX SYSTEM
function spawnDrawingParticles(x, y, color) {
  for (let i = 0; i < 2; i++) {
    state.particles.push({
      x,
      y,
      vx: (Math.random() - 0.5) * 3,
      vy: (Math.random() - 0.5) * 3,
      life: 1,
      decay: 0.02 + Math.random() * 0.03,
      size: 2 + Math.random() * 3,
      color
    });
  }
}

function updateAndDrawParticles(ctx) {
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const p = state.particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.life -= p.decay;
    p.size *= 0.97;

    if (p.life <= 0) {
      state.particles.splice(i, 1);
      continue;
    }

    ctx.save();
    ctx.globalAlpha = p.life * 0.7;
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

// Drawing Hand Skeleton Connectors
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [0, 9], [9, 10], [10, 11], [11, 12], // middle
  [0, 13], [13, 14], [14, 15], [15, 16], // ring
  [0, 17], [17, 18], [18, 19], [19, 20], // pinky
  [5, 9], [9, 13], [13, 17] // palm base links
];

// Draws detailed hand lines & dots
function drawHandSkeleton(ctx, landmarks) {
  if (!landmarks) return;
  ctx.save();
  ctx.globalAlpha = 0.3;
  
  // Render connector bones
  for (const [startIdx, endIdx] of CONNECTIONS) {
    const pt1 = toCanvasCoordinates(landmarks[startIdx]);
    const pt2 = toCanvasCoordinates(landmarks[endIdx]);
    ctx.beginPath();
    ctx.moveTo(pt1.x, pt1.y);
    ctx.lineTo(pt2.x, pt2.y);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Draw node joint circles
  for (let i = 0; i < landmarks.length; i++) {
    const pt = toCanvasCoordinates(landmarks[i]);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
    ctx.fill();
  }

  // Highlight tips
  const tips = [4, 8, 12, 16, 20];
  for (const tipIdx of tips) {
    const pt = toCanvasCoordinates(landmarks[tipIdx]);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.restore();
}

// Drawing brush cursor under index finger
function drawBrushCursor(ctx, landmarks, gesture) {
  if (gesture === "index_finger") {
    const indexTip = toCanvasCoordinates(landmarks[8]);
    ctx.save();
    ctx.beginPath();
    ctx.arc(indexTip.x, indexTip.y, state.thickness / 2 + 6, 0, Math.PI * 2);
    ctx.strokeStyle = state.activeColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.shadowColor = state.activeColor;
    ctx.shadowBlur = 8;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(indexTip.x, indexTip.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = state.activeColor;
    ctx.globalAlpha = 0.9;
    ctx.fill();
    ctx.restore();
  }
}

// MAIN ANIMATION LOOP
let lastFrameTime = -1;
function runMainLoop() {
  if (!state.handLandmarker || !state.isReady) {
    requestAnimationFrame(runMainLoop);
    return;
  }

  const video = webcamElement;
  const timestamp = performance.now();

  // 1. Draw camera feedback (horizontal flip mirroring)
  cameraCtx.clearRect(0, 0, state.width, state.height);
  if (state.showCamera) {
    cameraCtx.save();
    cameraCtx.globalAlpha = state.cameraOpacity;
    cameraCtx.translate(state.width, 0);
    cameraCtx.scale(-1, 1);
    cameraCtx.drawImage(video, 0, 0, state.width, state.height);
    cameraCtx.restore();
  }

  // 2. Clear UI canvas for current frame drawing
  uiCtx.clearRect(0, 0, state.width, state.height);

  // 3. Process video frames & run Hand tracking
  if (video.readyState >= 2 && video.currentTime !== lastFrameTime) {
    lastFrameTime = video.currentTime;
    const trackingResult = state.handLandmarker.detectForVideo(video, timestamp);

    if (trackingResult.landmarks && trackingResult.landmarks.length > 0) {
      const landmarks = trackingResult.landmarks[0];
      const rawGesture = classifyGesture(landmarks);
      const debounced = debounceGesture(rawGesture);

      // Perform gesture movements (if onboarding modal is gone)
      if (!state.isModalOpen) {
        if (debounced === "index_finger") {
          handleDrawGesture(landmarks);
        } else if (debounced === "open_palm") {
          handleEraseGesture(landmarks);
        } else if (debounced === "pinch") {
          handleMoveGesture(landmarks);
        }
      }

      drawHandSkeleton(uiCtx, landmarks);
      drawBrushCursor(uiCtx, landmarks, debounced);
    } else {
      // If hand leaves sight, clean up states
      if (state.currentGesture !== "none") {
        handleGestureTransition(state.currentGesture, "none");
        state.currentGesture = "none";
      }
      
      if (state.currentStroke && state.currentStroke.points.length > 1) {
        state.strokes.push({ ...state.currentStroke });
        state.currentStroke = null;
        redrawDrawingCanvas();
      }
    }
  }

  // 4. Update particles FX
  updateAndDrawParticles(uiCtx);
  
  requestAnimationFrame(runMainLoop);
}

// TOOLBAR & UI BUTTON CLICKS
const colorPalette = getEl("color-palette");
const colorPickerToggle = getEl("color-picker-toggle");
const activeColorPreview = getEl("active-color-preview");

function updateActiveColorUI(col) {
  activeColorPreview.style.setProperty("--swatch-color", col);
  activeColorPreview.dataset.color = col;
}

// Color select
document.querySelectorAll("#color-palette .color-swatch").forEach(swatch => {
  swatch.addEventListener("click", () => {
    document.querySelectorAll("#color-palette .color-swatch").forEach(s => s.classList.remove("active"));
    swatch.classList.add("active");
    state.activeColor = swatch.dataset.color;
    updateActiveColorUI(swatch.dataset.color);
    playSound(1000, 0.05, "sine", 0.03);
    colorPalette.classList.remove("mobile-open");
  });
});

// Mobile color drawer menu
colorPickerToggle.addEventListener("click", () => {
  colorPalette.classList.toggle("mobile-open");
});

document.addEventListener("pointerdown", (e) => {
  if (!colorPalette.contains(e.target) && !colorPickerToggle.contains(e.target)) {
    colorPalette.classList.remove("mobile-open");
  }
});

// Thickness settings
thicknessSlider.addEventListener("input", () => {
  state.thickness = parseInt(thicknessSlider.value);
  thicknessValue.textContent = `${state.thickness}px`;
});

// Glow settings
glowSlider.addEventListener("input", () => {
  state.glowIntensity = parseInt(glowSlider.value);
  glowValue.textContent = `${state.glowIntensity}%`;
});

// Actions: Undo
getEl("btn-undo").addEventListener("click", () => {
  if (state.strokes.length > 0) {
    state.strokes.pop();
    redrawDrawingCanvas();
    playSound(500, 0.08, "sine", 0.03);
  }
});

// Actions: Clear
getEl("btn-clear").addEventListener("click", () => {
  state.strokes = [];
  state.currentStroke = null;
  state.particles = [];
  redrawDrawingCanvas();
  playSound(300, 0.15, "triangle", 0.04);
});

// Actions: Camera Mode Toggles
getEl("btn-camera-toggle").addEventListener("click", () => {
  // Cycle settings: Camera ON -> Camera DIM -> Dark Canvas -> Camera ON
  if (state.showCamera && state.cameraOpacity > 0.2) {
    state.cameraOpacity = 0.15;
    cameraModeText.textContent = "Camera DIM";
    cameraModeIndicator.classList.remove("dark-mode");
  } else if (state.showCamera && state.cameraOpacity <= 0.2) {
    state.showCamera = false;
    state.cameraOpacity = 0;
    cameraModeText.textContent = "Dark Canvas";
    cameraModeIndicator.classList.add("dark-mode");
    getEl("btn-camera-toggle").classList.remove("active");
  } else {
    state.showCamera = true;
    state.cameraOpacity = 0.35;
    cameraModeText.textContent = "Camera ON";
    cameraModeIndicator.classList.remove("dark-mode");
    getEl("btn-camera-toggle").classList.add("active");
  }
  playSound(1200, 0.05, "sine", 0.03);
});

cameraModeIndicator.addEventListener("click", () => {
  getEl("btn-camera-toggle").click();
});

// Actions: PNG save exporter
getEl("btn-save").addEventListener("click", () => {
  // Construct a canvas with solid background to draw onto
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = state.width;
  exportCanvas.height = state.height;
  const exportCtx = exportCanvas.getContext("2d");

  // Solid dark background
  exportCtx.fillStyle = "#07070d";
  exportCtx.fillRect(0, 0, state.width, state.height);
  
  // Render drawing canvas content
  exportCtx.drawImage(drawingCanvas, 0, 0);

  const anchor = document.createElement("a");
  anchor.download = `air-draw-${Date.now()}.png`;
  anchor.href = exportCanvas.toDataURL("image/png");
  anchor.click();
  playSound(800, 0.1, "sine", 0.04);
});

// Onboarding Modal Close
btnStart.addEventListener("click", () => {
  onboardingModal.classList.add("hidden");
  state.isModalOpen = false;
  playSound(800, 0.1, "sine", 0.04);
  updateHudUI("idle");
});

// INITIALIZATION ENTRYPOINT
async function startApp() {
  resizeCanvases();
  try {
    const [trackOk, webcamOk] = await Promise.all([
      initHandTracking(),
      initWebcam()
    ]);
    
    state.isReady = true;

    // Transition loading indicators
    const fillEl = document.querySelector(".loader-bar-fill");
    fillEl.style.animation = "none";
    fillEl.style.width = "100%";
    fillEl.style.transition = "width 0.4s ease";

    setTimeout(() => {
      loadingScreen.classList.add("fade-out");
      appContainer.classList.remove("hidden");
      onboardingModal.classList.remove("hidden");
    }, 600);

    setTimeout(() => {
      loadingScreen.style.display = "none";
    }, 1200);

    runMainLoop();
  } catch (err) {
    console.error("Initialization Failed:", err);
    document.querySelector(".loader-subtitle").textContent = 
      "Error: Camera permission required. Please allow camera and refresh.";
    document.querySelector(".loader-subtitle").style.color = "#ff2d6b";
    document.querySelector(".loader-bar").style.display = "none";
  }
}

// Fire app boot
startApp();
