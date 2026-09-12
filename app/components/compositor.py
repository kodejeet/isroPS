"""Interactive visual compositor component for Lunar Registration Dashboard.

Renders:
- Primary view: Draggable vertical Swipe comparison
- Secondary tabs: Checker, Overlay, Edges, Difference, Side-by-side
- Match overlay toggles: Off, All, Kept, Rejected, Vectors
"""

import base64
import json
from typing import Any

import cv2
import numpy as np


def array_to_base64_png(arr: np.ndarray) -> str:
    """Encode 2D or 3D numpy array to base64 data URI PNG."""
    if arr is None:
        return ""
    img = arr.copy()
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img.dtype != np.uint8:
        p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
        if p99 > p1:
            img = np.clip((img - p1) / (p99 - p1) * 255.0, 0, 255.0).astype(
                np.uint8
            )
        else:
            img = img.astype(np.uint8)

    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_compositor_html(
    moving_arr: np.ndarray,
    reference_arr: np.ndarray,
    registered_arr: np.ndarray,
    tiepoints: dict[str, Any],
    instrument_label: str = "CH2 TMC-2",
    reference_label: str = "LROC NAC",
    canvas_size: int = 560,
) -> str:
    """Generate self-contained HTML/CSS/JS interactive comparison component."""
    moving_b64 = array_to_base64_png(moving_arr)
    reference_b64 = array_to_base64_png(reference_arr)
    registered_b64 = (
        array_to_base64_png(registered_arr)
        if registered_arr is not None
        else moving_b64
    )

    # Use json.dumps to ensure strict lowercase true/false in JavaScript!
    moving_pts = (
        tiepoints.get("moving", []).tolist()
        if hasattr(tiepoints.get("moving", []), "tolist")
        else list(tiepoints.get("moving", []))
    )
    ref_pts = (
        tiepoints.get("ref", []).tolist()
        if hasattr(tiepoints.get("ref", []), "tolist")
        else list(tiepoints.get("ref", []))
    )
    inlier_mask = (
        tiepoints.get("inlier_mask", []).tolist()
        if hasattr(tiepoints.get("inlier_mask", []), "tolist")
        else list(tiepoints.get("inlier_mask", []))
    )
    residuals = (
        tiepoints.get("residuals_px", []).tolist()
        if hasattr(tiepoints.get("residuals_px", []), "tolist")
        else list(tiepoints.get("residuals_px", []))
    )

    json_moving = json.dumps(moving_pts)
    json_ref = json.dumps(ref_pts)
    json_mask = json.dumps([bool(m) for m in inlier_mask])
    json_residuals = json.dumps(
        [round(float(r), 3) for r in residuals] if residuals else []
    )

    n_pts = len(moving_pts)
    n_kept = sum(1 for m in inlier_mask if m)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
  :root {{
    --void:    #05070B;
    --panel:   #121820;
    --panel-2: #0E141B;
    --rule:    #1E2833;
    --signal:  #E8EDF2;
    --muted:   #7A8794;
    --isro:    #FF7A45;
    --nasa:    #5B9CFF;
    --good:    #3FD68C;
    --warn:    #FFC14D;
    --bad:     #FF5C5C;
    --accent:  #5B9CFF;
  }}

  body {{
    background: transparent;
    color: var(--signal);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    padding: 0;
    overflow: hidden;
  }}

  /* Compositor Shell */
  .compositor-shell {{
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    background: var(--void);
    border: 1px solid var(--rule);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
  }}

  /* Top Control Bar */
  .bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 12px;
    background: var(--panel);
    border-bottom: 1px solid var(--rule);
  }}

  .tabs {{
    display: flex;
    gap: 4px;
    background: var(--void);
    padding: 3px;
    border-radius: 6px;
    border: 1px solid var(--rule);
  }}

  .tab-btn {{
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .tab-btn:hover {{
    color: var(--signal);
    background: rgba(255,255,255,0.05);
  }}

  .tab-btn.active {{
    background: #1B2430;
    color: var(--accent);
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
  }}

  .match-toggles {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: var(--void);
    padding: 3px 6px;
    border-radius: 6px;
    border: 1px solid var(--rule);
  }}

  .match-lbl {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-right: 4px;
  }}

  .toggle-btn {{
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }}

  .toggle-btn:hover {{
    color: var(--signal);
  }}

  .toggle-btn.active {{
    background: #1B2430;
    color: var(--accent);
    font-weight: 600;
  }}

  #btnKept.active {{
    color: var(--good);
  }}

  #btnRejected.active {{
    color: var(--bad);
  }}

  .match-sep {{
    display: inline-block;
    width: 1px;
    height: 16px;
    background: var(--rule);
    margin: 0 4px;
  }}

  .vector-btn {{
    background: transparent;
    color: var(--muted);
    border: 1px solid transparent;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: all 0.15s ease;
  }}

  .vector-btn:hover {{
    color: var(--signal);
    background: rgba(255, 255, 255, 0.05);
  }}

  .vector-btn.active {{
    background: #1B2430;
    color: #5B9CFF;
    border: 1px solid rgba(91, 156, 255, 0.5);
    font-weight: 600;
    box-shadow: 0 0 8px rgba(91, 156, 255, 0.25);
  }}

  .vector-btn .vec-icon {{
    font-size: 13px;
    font-weight: 700;
  }}

  /* Content Display Area */
  .stage-area {{
    position: relative;
    width: 100%;
    background: var(--void);
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  /* Stage Canvas Container (Swipe, Checker, Overlay, Edges, Difference) */
  .stage-wrap {{
    position: relative;
    width: 100%;
    max-width: {canvas_size}px;
    margin: 0 auto;
    background: var(--void);
  }}

  .stage {{
    position: relative;
    width: 100%;
    height: {canvas_size}px;
    cursor: ew-resize;
    touch-action: none;
    border-left: 1px solid var(--rule);
    border-right: 1px solid var(--rule);
  }}

  canvas {{
    display: block;
    width: 100%;
    height: 100%;
    image-rendering: pixelated;
  }}

  /* Draggable Divider Handle for Swipe */
  .swipe-divider {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #FFFFFF;
    box-shadow: 0 0 8px rgba(0,0,0,0.9), 0 0 2px rgba(255,255,255,0.9);
    pointer-events: none;
    z-index: 10;
  }}

  .swipe-knob {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 32px;
    height: 32px;
    background: #0E141B;
    border: 2px solid #FFFFFF;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.8);
    pointer-events: auto;
    cursor: ew-resize;
  }}

  .swipe-knob::before, .swipe-knob::after {{
    content: "";
    display: block;
    width: 0;
    height: 0;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
  }}
  .swipe-knob::before {{ border-right: 6px solid #FFFFFF; margin-right: 3px; }}
  .swipe-knob::after {{ border-left: 6px solid #FFFFFF; margin-left: 3px; }}

  /* Mission Source Badges (on single stage) */
  .badge {{
    position: absolute;
    bottom: 10px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0.04em;
    backdrop-filter: blur(4px);
    z-index: 8;
    pointer-events: none;
  }}
  .badge-moving {{
    left: 10px;
    background: rgba(255, 122, 69, 0.9);
    color: #FFFFFF;
  }}
  .badge-ref {{
    right: 10px;
    background: rgba(91, 156, 255, 0.9);
    color: #FFFFFF;
  }}

  /* Side-by-Side 3-Card Grid (1:1 Ratio Square Images) */
  .sbs-wrap {{
    width: 100%;
    padding: 12px 16px;
  }}

  .sbs-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    width: 100%;
  }}

  .sbs-card {{
    background: var(--panel);
    border: 1px solid var(--rule);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    transition: border-color 0.2s ease;
  }}

  .sbs-card:hover {{
    border-color: #2E3E4E;
  }}

  .sbs-card-mapped {{
    border-color: rgba(63, 214, 140, 0.35);
  }}

  .sbs-canvas-box {{
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #000000;
  }}

  .sbs-canvas-box canvas {{
    width: 100%;
    height: 100%;
    display: block;
    image-rendering: pixelated;
  }}

  .sbs-footer {{
    padding: 8px 10px;
    background: var(--panel-2);
    border-top: 1px solid var(--rule);
    display: flex;
    flex-direction: column;
    gap: 3px;
  }}

  .sbs-tag {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 6px;
    border-radius: 4px;
    width: fit-content;
  }}

  .sbs-tag-moving {{
    background: rgba(255, 122, 69, 0.18);
    color: var(--isro);
    border: 1px solid rgba(255, 122, 69, 0.4);
  }}

  .sbs-tag-ref {{
    background: rgba(91, 156, 255, 0.18);
    color: var(--nasa);
    border: 1px solid rgba(91, 156, 255, 0.4);
  }}

  .sbs-tag-reg {{
    background: rgba(63, 214, 140, 0.18);
    color: var(--good);
    border: 1px solid rgba(63, 214, 140, 0.4);
  }}

  .sbs-name {{
    font-size: 12px;
    font-weight: 600;
    color: var(--signal);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
  }}

  .sbs-sub {{
    font-size: 11px;
    color: var(--muted);
    line-height: 1.2;
  }}

  .sbs-sub-green {{
    color: var(--good);
    font-weight: 500;
  }}

  /* Sub-caption */
  .caption-strip {{
    padding: 8px 14px;
    background: #090D12;
    border-top: 1px solid var(--rule);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    color: #A0B0C0;
  }}

  .legend-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
  }}
</style>
</head>
<body>

<div class="compositor-shell">
  <div class="bar">
    <div class="tabs" role="tablist">
      <button class="tab-btn active" data-mode="swipe">Swipe</button>
      <button class="tab-btn" data-mode="checker">Checker</button>
      <button class="tab-btn" data-mode="overlay">Overlay</button>
      <button class="tab-btn" data-mode="edges">Edges</button>
      <button class="tab-btn" data-mode="difference">Difference</button>
      <button class="tab-btn" data-mode="sidebyside">Side-by-side</button>
    </div>

    <div class="match-toggles">
      <span class="match-lbl">Matches:</span>
      <button class="toggle-btn" id="btnOff" data-filter="off">Off</button>
      <button class="toggle-btn" id="btnAll" data-filter="all">All ({n_pts})</button>
      <button class="toggle-btn active" id="btnKept" data-filter="inliers">Kept ({n_kept})</button>
      <button class="toggle-btn" id="btnRejected" data-filter="outliers">Rejected ({n_pts - n_kept})</button>
      <span class="match-sep"></span>
      <button class="vector-btn" id="btnVectors" title="Toggle displacement vectors overlay">
        <span class="vec-icon">↗</span> Vectors
      </button>
    </div>
  </div>

  <div class="stage-area">
    <!-- View 1: Single Stage for Swipe, Checker, Overlay, Edges, Difference -->
    <div class="stage-wrap" id="stageWrap">
      <div class="stage" id="stage">
        <canvas id="viewCanvas" width="{canvas_size}" height="{canvas_size}"></canvas>
        
        <div class="swipe-divider" id="divider" style="left: 50%;">
          <div class="swipe-knob"></div>
        </div>

        <span class="badge badge-moving" id="badgeLeft">{instrument_label}</span>
        <span class="badge badge-ref" id="badgeRight">{reference_label}</span>
      </div>
    </div>

    <!-- View 2: Side-by-Side 3-Card Grid (1:1 Ratio Square Images) -->
    <div class="sbs-wrap" id="sbsWrap" style="display: none;">
      <div class="sbs-grid">
        <!-- 1st Image: Moving -->
        <div class="sbs-card">
          <div class="sbs-canvas-box">
            <canvas id="sbsCanvas1" width="512" height="512"></canvas>
          </div>
          <div class="sbs-footer">
            <div class="sbs-tag sbs-tag-moving">1. Moving</div>
            <div class="sbs-name" title="{instrument_label}">{instrument_label}</div>
            <div class="sbs-sub">Source Crop (No Dots)</div>
          </div>
        </div>

        <!-- 2nd Image: Reference -->
        <div class="sbs-card">
          <div class="sbs-canvas-box">
            <canvas id="sbsCanvas2" width="512" height="512"></canvas>
          </div>
          <div class="sbs-footer">
            <div class="sbs-tag sbs-tag-ref">2. Reference</div>
            <div class="sbs-name" title="{reference_label}">{reference_label}</div>
            <div class="sbs-sub">Reference Crop (No Dots)</div>
          </div>
        </div>

        <!-- 3rd Image: Mapped / Registered -->
        <div class="sbs-card sbs-card-mapped">
          <div class="sbs-canvas-box">
            <canvas id="sbsCanvas3" width="512" height="512"></canvas>
          </div>
          <div class="sbs-footer">
            <div class="sbs-tag sbs-tag-reg">3. Mapped</div>
            <div class="sbs-name" title="Registered Output">Registered Output</div>
            <div class="sbs-sub sbs-sub-green" id="sbsSub3">● Matched Points ({n_kept} inliers)</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="caption-strip">
    <span id="captionText">Drag to compare. Terrain should run continuous across the divider.</span>
    <span id="ptSummary" style="font-size: 12px; color: var(--muted);">
      <span class="legend-dot" style="background: var(--good);"></span>{n_kept} kept
      <span class="legend-dot" style="background: var(--bad); margin-left:8px;"></span>{n_pts - n_kept} rejected
    </span>
  </div>
</div>

<script>
(function() {{
  const size = {canvas_size};
  const canvas = document.getElementById("viewCanvas");
  const ctx = canvas.getContext("2d");
  const stage = document.getElementById("stage");
  const stageWrap = document.getElementById("stageWrap");
  const sbsWrap = document.getElementById("sbsWrap");
  const divider = document.getElementById("divider");
  const captionText = document.getElementById("captionText");
  const badgeLeft = document.getElementById("badgeLeft");
  const badgeRight = document.getElementById("badgeRight");

  const captions = {{
    "swipe": "Drag to compare. Terrain should run continuous across the divider.",
    "checker": "Alternating tiles from each image. Craters should cross tile edges unbroken.",
    "overlay": "Both at once in mission colours, brightness ranges matched. Neutral grey means agreement; saffron or blue fringes mean one image sits off the other.",
    "edges": "Ridges and crater rims from each image. Overlapping outlines mean a tight fit.",
    "difference": "One subtracted from the other after matching their brightness ranges, so what is left is structural. Dark is agreement; bright outlines are edges that did not land on top of each other.",
    "sidebyside": "Moving, reference and registered result side-by-side in real 1:1 ratio. Green dots show verified matched points on mapped output."
  }};

  let currentMode = "swipe";
  let matchFilter = "inliers";
  let showVectors = false;
  let splitPercent = 0.50;
  let isDragging = false;

  // Strict valid JSON arrays for JS
  const movingPts = {json_moving};
  const refPts = {json_ref};
  const inlierMask = {json_mask};
  const residuals = {json_residuals};
  const totalPts = movingPts.length;
  const totalInliers = {n_kept};

  // Load images
  function loadImage(src) {{
    return new Promise((resolve) => {{
      if (!src) {{ resolve(null); return; }}
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => {{
        console.warn("Failed to load image");
        resolve(null);
      }};
      img.src = src;
      if (img.complete && img.naturalWidth > 0) {{
        resolve(img);
      }}
    }});
  }}

  let imgMoving = null;
  let imgRef = null;
  let imgReg = null;

  Promise.all([
    loadImage("{moving_b64}"),
    loadImage("{reference_b64}"),
    loadImage("{registered_b64}")
  ]).then(([m, rf, rg]) => {{
    imgMoving = m;
    imgRef = rf;
    imgReg = rg || m;
    render();
  }});

  // Setup tab listeners
  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.addEventListener("click", (e) => {{
      e.preventDefault();
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;

      if (currentMode === "sidebyside") {{
        stageWrap.style.display = "none";
        sbsWrap.style.display = "block";
        divider.style.display = "none";
        badgeLeft.style.display = "none";
        badgeRight.style.display = "none";
      }} else {{
        sbsWrap.style.display = "none";
        stageWrap.style.display = "block";
        divider.style.display = (currentMode === "swipe") ? "block" : "none";
        stage.style.cursor = (currentMode === "swipe") ? "ew-resize" : "default";
        badgeLeft.style.display = (currentMode === "swipe" || currentMode === "overlay") ? "block" : "none";
        badgeRight.style.display = (currentMode === "swipe" || currentMode === "overlay") ? "block" : "none";
      }}

      captionText.textContent = captions[currentMode] || "";
      render();
    }});
  }});

  // Setup match toggle listeners
  const btnOff = document.getElementById("btnOff");
  const btnAll = document.getElementById("btnAll");
  const btnKept = document.getElementById("btnKept");
  const btnRejected = document.getElementById("btnRejected");
  const btnVectors = document.getElementById("btnVectors");

  const filterBtns = [btnOff, btnAll, btnKept, btnRejected];

  filterBtns.forEach(btn => {{
    if (!btn) return;
    btn.addEventListener("click", (e) => {{
      e.preventDefault();
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      matchFilter = btn.dataset.filter;

      if (matchFilter === "off") {{
        showVectors = false;
        if (btnVectors) btnVectors.classList.remove("active");
      }}
      render();
    }});
  }});

  if (btnVectors) {{
    btnVectors.addEventListener("click", (e) => {{
      e.preventDefault();
      showVectors = !showVectors;
      btnVectors.classList.toggle("active", showVectors);

      // If user toggles vectors ON while filter was off, automatically activate Kept
      if (showVectors && matchFilter === "off") {{
        matchFilter = "inliers";
        filterBtns.forEach(b => b.classList.remove("active"));
        if (btnKept) btnKept.classList.add("active");
      }}

      render();
    }});
  }}

  // Pointer Dragging for Swipe (Smooth on desktop & mobile)
  function updateSplit(clientX) {{
    const rect = stage.getBoundingClientRect();
    let frac = (clientX - rect.left) / rect.width;
    frac = Math.max(0.01, Math.min(0.99, frac));
    splitPercent = frac;
    divider.style.left = (splitPercent * 100) + "%";
    render();
  }}

  stage.addEventListener("pointerdown", (e) => {{
    if (currentMode !== "swipe") return;
    isDragging = true;
    try {{ stage.setPointerCapture(e.pointerId); }} catch(err) {{}}
    updateSplit(e.clientX);
  }});

  stage.addEventListener("pointermove", (e) => {{
    if (isDragging && currentMode === "swipe") {{
      updateSplit(e.clientX);
    }}
  }});

  const stopDrag = (e) => {{
    if (isDragging) {{
      isDragging = false;
      try {{ stage.releasePointerCapture(e.pointerId); }} catch(err) {{}}
    }}
  }};

  stage.addEventListener("pointerup", stopDrag);
  stage.addEventListener("pointercancel", stopDrag);

  function render() {{
    if (!imgMoving || !imgRef) {{
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = "#0E141B";
      ctx.fillRect(0, 0, size, size);
      ctx.fillStyle = "#7A8794";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Rendering terrain crops...", size/2, size/2);
      return;
    }}

    const targetReg = imgReg || imgMoving;

    if (currentMode === "sidebyside") {{
      stageWrap.style.display = "none";
      sbsWrap.style.display = "block";
      renderSideBySide(targetReg);
      return; // Do NOT draw tiepoints on viewCanvas
    }}

    stageWrap.style.display = "block";
    sbsWrap.style.display = "none";
    ctx.clearRect(0, 0, size, size);

    if (currentMode === "swipe") {{
      renderSwipe(targetReg);
    }} else if (currentMode === "checker") {{
      renderChecker(targetReg);
    }} else if (currentMode === "overlay") {{
      renderOverlay(targetReg);
    }} else if (currentMode === "edges") {{
      renderEdges(targetReg);
    }} else if (currentMode === "difference") {{
      renderDifference(targetReg);
    }}

    // Draw Tiepoints overlay on swipe and other features
    if (matchFilter !== "off") {{
      renderTiepoints();
    }}
  }}

  function renderSwipe(targetReg) {{
    const splitX = Math.round(size * splitPercent);

    // Left side: Registered image
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, size);
    ctx.clip();
    ctx.drawImage(targetReg, 0, 0, size, size);
    ctx.restore();

    // Right side: Reference image
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, size - splitX, size);
    ctx.clip();
    ctx.drawImage(imgRef, 0, 0, size, size);
    ctx.restore();
  }}

  function renderChecker(targetReg) {{
    const numCells = 8;
    const cellSz = size / numCells;

    for (let r = 0; r < numCells; r++) {{
      for (let c = 0; c < numCells; c++) {{
        const isReg = (r + c) % 2 === 0;
        const sourceImg = isReg ? targetReg : imgRef;
        const x = c * cellSz;
        const y = r * cellSz;

        ctx.drawImage(
          sourceImg,
          (c * sourceImg.width) / numCells,
          (r * sourceImg.height) / numCells,
          sourceImg.width / numCells,
          sourceImg.height / numCells,
          x, y, cellSz, cellSz
        );
      }}
    }}

    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.lineWidth = 1;
    for (let i = 1; i < numCells; i++) {{
      ctx.beginPath();
      ctx.moveTo(i * cellSz, 0); ctx.lineTo(i * cellSz, size);
      ctx.moveTo(0, i * cellSz); ctx.lineTo(size, i * cellSz);
      ctx.stroke();
    }}
  }}

  function renderOverlay(targetReg) {{
    const cA = document.createElement("canvas");
    cA.width = size; cA.height = size;
    const ctxA = cA.getContext("2d");
    ctxA.drawImage(targetReg, 0, 0, size, size);
    const dataA = ctxA.getImageData(0, 0, size, size).data;

    const cB = document.createElement("canvas");
    cB.width = size; cB.height = size;
    const ctxB = cB.getContext("2d");
    ctxB.drawImage(imgRef, 0, 0, size, size);
    const dataB = ctxB.getImageData(0, 0, size, size).data;

    const outImg = ctx.createImageData(size, size);
    const d = outImg.data;

    for (let i = 0; i < dataA.length; i += 4) {{
      const vA = dataA[i] / 255.0;      // Moving (ISRO Saffron #FF7A45)
      const vB = dataB[i] / 255.0;      // Ref (NASA Blue #5B9CFF)

      d[i]   = Math.min(255, Math.floor((vA * 255 * 0.9) + (vB * 91 * 0.4)));
      d[i+1] = Math.min(255, Math.floor((vA * 122 * 0.8) + (vB * 156 * 0.7)));
      d[i+2] = Math.min(255, Math.floor((vA * 69 * 0.4)  + (vB * 255 * 0.9)));
      d[i+3] = 255;
    }}
    ctx.putImageData(outImg, 0, 0);
  }}

  function renderEdges(targetReg) {{
    const cA = document.createElement("canvas");
    cA.width = size; cA.height = size;
    const ctxA = cA.getContext("2d");
    ctxA.drawImage(targetReg, 0, 0, size, size);
    const dataA = ctxA.getImageData(0, 0, size, size).data;

    const cB = document.createElement("canvas");
    cB.width = size; cB.height = size;
    const ctxB = cB.getContext("2d");
    ctxB.drawImage(imgRef, 0, 0, size, size);
    const dataB = ctxB.getImageData(0, 0, size, size).data;

    const outImg = ctx.createImageData(size, size);
    const d = outImg.data;

    for (let y = 1; y < size - 1; y++) {{
      for (let x = 1; x < size - 1; x++) {{
        const idx = (y * size + x) * 4;
        const gAx = Math.abs(dataA[idx + 4] - dataA[idx - 4]);
        const gAy = Math.abs(dataA[idx + size * 4] - dataA[idx - size * 4]);
        const eA = Math.min(255, (gAx + gAy) * 2.5);

        const gBx = Math.abs(dataB[idx + 4] - dataB[idx - 4]);
        const gBy = Math.abs(dataB[idx + size * 4] - dataB[idx - size * 4]);
        const eB = Math.min(255, (gBx + gBy) * 2.5);

        d[idx]   = Math.min(255, Math.floor(eA * 1.0 + eB * 0.2));
        d[idx+1] = Math.min(255, Math.floor(eA * 0.45 + eB * 0.7));
        d[idx+2] = Math.min(255, Math.floor(eA * 0.15 + eB * 1.0));
        d[idx+3] = 255;
      }}
    }}
    ctx.putImageData(outImg, 0, 0);
  }}

  function renderDifference(targetReg) {{
    const cA = document.createElement("canvas");
    cA.width = size; cA.height = size;
    const ctxA = cA.getContext("2d");
    ctxA.drawImage(targetReg, 0, 0, size, size);
    const dataA = ctxA.getImageData(0, 0, size, size).data;

    const cB = document.createElement("canvas");
    cB.width = size; cB.height = size;
    const ctxB = cB.getContext("2d");
    ctxB.drawImage(imgRef, 0, 0, size, size);
    const dataB = ctxB.getImageData(0, 0, size, size).data;

    const outImg = ctx.createImageData(size, size);
    const d = outImg.data;

    for (let i = 0; i < dataA.length; i += 4) {{
      const diff = Math.abs(dataA[i] - dataB[i]);
      const val = Math.min(255, diff * 2.5);
      d[i]   = val;
      d[i+1] = Math.floor(val * 0.85);
      d[i+2] = Math.floor(val * 0.5);
      d[i+3] = 255;
    }}
    ctx.putImageData(outImg, 0, 0);
  }}

  function drawMatchLine(targetCtx, fromX, fromY, toX, toY, color, width, isInlier) {{
    const w = width || 1.8;

    let startX = fromX;
    let startY = fromY;
    let endX = toX;
    let endY = toY;

    if (isInlier) {{
      const dx = toX - fromX;
      const dy = toY - fromY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 0.001) {{
        const ux = dx / dist;
        const uy = dy / dist;
        // Scale sub-pixel inlier residual into a crisp 8-20px presentation vector
        const displayLen = Math.max(8.0, Math.min(20.0, dist * 24.0));
        startX = toX - ux * displayLen;
        startY = toY - uy * displayLen;
      }} else {{
        startX = toX - 8.0;
        startY = toY - 6.0;
      }}
    }}

    targetCtx.save();

    // Straight vector line between dots (no arrows, clean stroke)
    targetCtx.strokeStyle = color;
    targetCtx.lineWidth = w;
    targetCtx.lineCap = "round";
    targetCtx.beginPath();
    targetCtx.moveTo(startX, startY);
    targetCtx.lineTo(endX, endY);
    targetCtx.stroke();

    // Clean dot marker at source point (no dark black border)
    targetCtx.beginPath();
    targetCtx.arc(startX, startY, 3.0, 0, Math.PI * 2);
    targetCtx.fillStyle = color;
    targetCtx.fill();

    // Clean dot marker at destination point (no dark black border)
    targetCtx.beginPath();
    targetCtx.arc(endX, endY, 3.0, 0, Math.PI * 2);
    targetCtx.fillStyle = color;
    targetCtx.fill();

    targetCtx.restore();
  }}

  function renderSideBySide(targetReg) {{
    const c1 = document.getElementById("sbsCanvas1");
    const c2 = document.getElementById("sbsCanvas2");
    const c3 = document.getElementById("sbsCanvas3");
    if (!c1 || !c2 || !c3) return;

    const ctx1 = c1.getContext("2d");
    const ctx2 = c2.getContext("2d");
    const ctx3 = c3.getContext("2d");

    // 1. Moving Image in real 1:1 ratio (NO dots)
    ctx1.clearRect(0, 0, c1.width, c1.height);
    if (imgMoving) {{
      ctx1.drawImage(imgMoving, 0, 0, c1.width, c1.height);
    }}

    // 2. Reference Image in real 1:1 ratio (NO dots)
    ctx2.clearRect(0, 0, c2.width, c2.height);
    if (imgRef) {{
      ctx2.drawImage(imgRef, 0, 0, c2.width, c2.height);
    }}

    // 3. Mapped / Registered Image in real 1:1 ratio
    ctx3.clearRect(0, 0, c3.width, c3.height);
    if (targetReg) {{
      ctx3.drawImage(targetReg, 0, 0, c3.width, c3.height);
    }}

    // Dynamic subtitle feedback under Image 3
    const sbsSub3 = document.getElementById("sbsSub3");
    if (sbsSub3) {{
      if (matchFilter === "off") {{
        sbsSub3.textContent = "Matches: Off";
        sbsSub3.className = "sbs-sub";
        sbsSub3.style.color = "";
      }} else if (matchFilter === "inliers") {{
        sbsSub3.textContent = showVectors ? `↗ Inlier Vectors (${{totalInliers}} pts)` : `● Matched Points (${{totalInliers}} inliers)`;
        sbsSub3.className = "sbs-sub sbs-sub-green";
        sbsSub3.style.color = "var(--good)";
      }} else if (matchFilter === "outliers") {{
        const nOut = totalPts - totalInliers;
        sbsSub3.textContent = showVectors ? `↗ Rejected Vectors (${{nOut}} pts)` : `● Rejected Outliers (${{nOut}} pts)`;
        sbsSub3.className = "sbs-sub";
        sbsSub3.style.color = "var(--bad)";
      }} else if (matchFilter === "all") {{
        sbsSub3.textContent = showVectors ? `↗ All Vectors (${{totalPts}} pts)` : `● All Matches (${{totalPts}} pts)`;
        sbsSub3.className = "sbs-sub";
        sbsSub3.style.color = "var(--nasa)";
      }}
    }}

    // Render points and/or vectors ONLY on Image 3
    if (matchFilter !== "off") {{
      const imgW = (imgRef && imgRef.naturalWidth) ? imgRef.naturalWidth : 512.0;
      const imgH = (imgRef && imgRef.naturalHeight) ? imgRef.naturalHeight : 512.0;
      const scaleX = c3.width / imgW;
      const scaleY = c3.height / imgH;

      for (let i = 0; i < totalPts; i++) {{
        const isInlier = Boolean(inlierMask[i]);
        if (matchFilter === "inliers" && !isInlier) continue;
        if (matchFilter === "outliers" && isInlier) continue;

        const pSrc = movingPts[i];
        const pRef = refPts[i] || pSrc;
        if (!pRef || pRef.length < 2) continue;

        const sx = (pSrc ? pSrc[0] : pRef[0]) * scaleX;
        const sy = (pSrc ? pSrc[1] : pRef[1]) * scaleY;
        const rx = pRef[0] * scaleX;
        const ry = pRef[1] * scaleY;

        const color = isInlier ? "#3FD68C" : "#FF5C5C";

        if (showVectors) {{
          drawMatchLine(ctx3, sx, sy, rx, ry, color, 1.8, isInlier);
        }} else {{
          // Clean point marker at destination (no black border)
          ctx3.beginPath();
          ctx3.arc(rx, ry, 3.2, 0, Math.PI * 2);
          ctx3.fillStyle = color;
          ctx3.fill();
        }}
      }}
    }}
  }}

  function renderTiepoints() {{
    const scale = size / 512.0;

    for (let i = 0; i < totalPts; i++) {{
      const isInlier = Boolean(inlierMask[i]);
      if (matchFilter === "inliers" && !isInlier) continue;
      if (matchFilter === "outliers" && isInlier) continue;

      const pSrc = movingPts[i];
      const pRef = refPts[i] || pSrc;

      const sx = (pSrc ? pSrc[0] : pRef[0]) * scale;
      const sy = (pSrc ? pSrc[1] : pRef[1]) * scale;
      const rx = (pRef ? pRef[0] : pSrc[0]) * scale;
      const ry = (pRef ? pRef[1] : pSrc[1]) * scale;

      const color = isInlier ? "#3FD68C" : "#FF5C5C";

      // Draw point-to-point match line or dot marker
      if (showVectors) {{
        drawMatchLine(ctx, sx, sy, rx, ry, color, 1.8, isInlier);
      }} else {{
        // Clean point marker at destination (no black border)
        ctx.beginPath();
        ctx.arc(rx, ry, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }}
    }}
  }}

}})();
</script>
</body>
</html>
"""
    return html
