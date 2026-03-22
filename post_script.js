const plot = document.querySelector('.js-plotly-plot');
const plotHost = plot.parentNode;
const layoutWrapper = document.createElement('div');
layoutWrapper.style = "display: flex; flex-direction: column; align-items: stretch; gap: 12px; width: fit-content;";
plotHost.insertBefore(layoutWrapper, plot);

const plotFrame = document.createElement('div');
plotFrame.style = "position: relative; padding-top: 20px;";
layoutWrapper.appendChild(plotFrame);
plotFrame.appendChild(plot);

// 0. UI 테마 설정
const UI_STYLE = {
    bg: '#ffffff',
    primary: '#101010',
    secondary: '#e0e0e0',
    textMain: '#101010',
    textMuted: '#666',
    shadow: '0 4px 12px rgba(0,0,0,0.12)',
    accent: '#1393FD'
};

const CONTROL_HEIGHT = 36;

// 1. Control Bar
const controlBar = document.createElement('div');
controlBar.style = `
    position: absolute;
    left: 20px;
    top: 8px;
    z-index: 1000;
    display: flex; 
    gap: 12px; 
    align-items: flex-end; 
    flex-wrap: nowrap;
    padding: 10px;
    font-family: 'Inter', -apple-system, sans-serif;
`;
plotFrame.appendChild(controlBar);

// 그룹 레이블 생성 함수
const createGroup = (label, content) => {
    const group = document.createElement('div');
    group.style = "display: flex; flex-direction: column; gap: 6px;";
    const labelSpan = document.createElement('span');
    labelSpan.innerText = label;
    labelSpan.style = `font-weight: 800; font-size: 11px; color: ${UI_STYLE.textMuted}; text-transform: uppercase; letter-spacing: 0.05em;`;
    group.appendChild(labelSpan);
    group.appendChild(content);
    return group;
};

// 버튼 공통 스타일 적용 함수
const applyButtonStyle = (btn, isActive, isAccent = false) => {
    btn.style.padding = "0 14px";
    btn.style.height = `${CONTROL_HEIGHT}px`;
    btn.style.border = "none";
    btn.style.borderRadius = "4px";
    btn.style.cursor = "pointer";
    btn.style.fontSize = "13px";
    btn.style.fontWeight = "600";
    btn.style.transition = "all 0.2s ease";
    btn.style.display = "flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";
    btn.style.gap = "6px";
    btn.style.boxShadow = "0 2px 4px rgba(0,0,0,0.05)";
    btn.style.lineHeight = "1";
    btn.style.whiteSpace = "nowrap";

    if (isActive) {
        btn.style.background = isAccent ? UI_STYLE.accent : UI_STYLE.primary;
        btn.style.color = "white";
    } else {
        btn.style.background = "white";
        btn.style.color = UI_STYLE.textMuted;
    }
};

// Text Toggle 아이콘 생성
const createTextToggleIcon = (showText = true) => {
    const icon = document.createElement('span');
    icon.style = "position: relative; display: inline-block; width: 16px; height: 14px;";
    const dot = document.createElement('span');
    dot.style = "position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 4px; height: 4px; border-radius: 50%; background: currentColor;";
    if (showText) {
        const text = document.createElement('span');
        text.innerText = "Aa";
        text.style = "position: absolute; top: -2px; left: 50%; transform: translateX(-50%); font-size: 9px; font-weight: 800; color: currentColor;";
        icon.appendChild(text);
    }
    icon.appendChild(dot);
    return icon;
};

// 2. Text Toggle (Show/Hide)
const textGroupWrapper = document.createElement('div');
textGroupWrapper.style = `display: flex; background: ${UI_STYLE.secondary}; padding: 2px; border-radius: 6px; gap: 2px;`;

const showBtn = document.createElement('button');
const hideBtn = document.createElement('button');

const updateTextUI = (show) => {
    applyButtonStyle(showBtn, show, false);
    applyButtonStyle(hideBtn, !show);
    showBtn.replaceChildren(createTextToggleIcon(true), document.createTextNode("ON"));
    hideBtn.replaceChildren(createTextToggleIcon(false), document.createTextNode("OFF"));
};

showBtn.onclick = () => { updateTextUI(true); Plotly.restyle(plot, {mode: 'markers+text'}); };
hideBtn.onclick = () => { updateTextUI(false); Plotly.restyle(plot, {mode: 'markers'}); };
updateTextUI(true);

textGroupWrapper.appendChild(showBtn);
textGroupWrapper.appendChild(hideBtn);
controlBar.appendChild(createGroup("Text", textGroupWrapper));

// 3. Mode Toggle (Move/Focus)
const modeGroupWrapper = document.createElement('div');
modeGroupWrapper.style = `display: flex; background: ${UI_STYLE.secondary}; padding: 2px; border-radius: 6px; gap: 2px;`;

const moveBtn = document.createElement('button');
const focusBtn = document.createElement('button');
moveBtn.innerHTML = "✢ Move";
focusBtn.innerHTML = "⛶ Focus";

const updateModeUI = (isMove) => {
    applyButtonStyle(moveBtn, isMove);
    applyButtonStyle(focusBtn, !isMove);
};

moveBtn.onclick = () => { updateModeUI(true); Plotly.relayout(plot, {dragmode: 'pan'}); };
focusBtn.onclick = () => { updateModeUI(false); Plotly.relayout(plot, {dragmode: 'zoom'}); };
updateModeUI(true);

modeGroupWrapper.appendChild(moveBtn);
modeGroupWrapper.appendChild(focusBtn);
controlBar.appendChild(createGroup("Mode", modeGroupWrapper));

// 4. Zoom Controller
const zoomGroupWrapper = document.createElement('div');
zoomGroupWrapper.style = `display: flex; align-items: center; background: white; border-radius: 6px; border: 1px solid ${UI_STYLE.secondary}; overflow: hidden;`;

const handleZoom = (factor) => {
    const xr = plot._fullLayout.xaxis.range;
    const yr = plot._fullLayout.yaxis.range;
    const currentXSpan = xr[1] - xr[0];
    const currentYSpan = yr[1] - yr[0];
    const centerX = (xr[0] + xr[1]) / 2;
    const centerY = (yr[0] + yr[1]) / 2;
    const newHalfX = (currentXSpan * factor) / 2;
    const newHalfY = (currentYSpan * factor) / 2;
    Plotly.relayout(plot, {
        'xaxis.range': [centerX - newHalfX, centerX + newHalfX],
        'yaxis.range': [centerY - newHalfY, centerY + newHalfY],
        'xaxis.autorange': false,
        'yaxis.autorange': false
    });
};

const minusBtn = document.createElement('button');
minusBtn.innerText = "-";
const plusBtn = document.createElement('button');
plusBtn.innerText = "+";
[minusBtn, plusBtn].forEach(b => {
    b.style = `height: ${CONTROL_HEIGHT}px; padding: 0 14px; border: none; cursor: pointer; background: white; color: ${UI_STYLE.primary}; font-weight: bold; font-size: 16px; line-height: 1; white-space: nowrap;`;
    b.onmouseover = () => b.style.background = "#f5f5f5";
    b.onmouseout = () => b.style.background = "white";
});

minusBtn.onclick = () => handleZoom(1.25);
plusBtn.onclick = () => handleZoom(0.8);

const zoomIcon = document.createElement('div');
zoomIcon.innerHTML = "🔍";
zoomIcon.style = `height: ${CONTROL_HEIGHT}px; padding: 0 8px; display: flex; align-items: center; justify-content: center; font-size: 12px; border-left: 1px solid #eee; border-right: 1px solid #eee;`;

zoomGroupWrapper.appendChild(minusBtn);
zoomGroupWrapper.appendChild(zoomIcon);
zoomGroupWrapper.appendChild(plusBtn);
controlBar.appendChild(createGroup("Zoom", zoomGroupWrapper));

// 5. Search Style
const searchInput = document.createElement('input');
searchInput.placeholder = "Search Style...";
searchInput.style = `height: ${CONTROL_HEIGHT}px; padding: 0 15px; border: 2px solid ${UI_STYLE.secondary}; border-radius: 6px; width: 200px; outline: none; font-size: 13px; font-weight: 500; box-sizing: border-box; transition: border-color 0.2s;`;
searchInput.onfocus = () => searchInput.style.borderColor = UI_STYLE.primary;
searchInput.onblur = () => searchInput.style.borderColor = UI_STYLE.secondary;

const showSearchMessage = (message, color) => {
    const msgBox = document.createElement('div');
    msgBox.innerText = message;
    Object.assign(msgBox.style, {
        position: 'absolute', top: '65px', left: '0', background: color, color: 'white',
        padding: '6px 12px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold',
        boxShadow: UI_STYLE.shadow, zIndex: '1001', transition: 'opacity 0.5s ease', whiteSpace: 'nowrap'
    });
    searchInput.parentElement.style.position = 'relative';
    searchInput.parentElement.appendChild(msgBox);
    setTimeout(() => { msgBox.style.opacity = '0'; setTimeout(() => msgBox.remove(), 500); }, 2000);
};

const normalizeStyleSearch = (value) => (value || "").toLowerCase().replace(/[-_\s]+/g, '');

searchInput.onkeypress = (e) => {
    if (e.key === 'Enter') {
        const term = normalizeStyleSearch(searchInput.value);
        if (!term) return;
        const matches = search_data.filter(row => normalizeStyleSearch(row.style) == term);
        if (matches.length > 0) {
            const target = matches[0];
            const targetX = Number(target.dim_0);
            const targetY = Number(target.dim_1);
            const allX = search_data.map(d => Number(d.dim_0));
            const allY = search_data.map(d => Number(d.dim_1));
            const globalSpanX = Math.max(...allX) - Math.min(...allX);
            const globalSpanY = Math.max(...allY) - Math.min(...allY);
            const zoomRatio = 0.15;
            const halfSpanX = (globalSpanX * zoomRatio) / 2;
            const halfSpanY = (globalSpanY * zoomRatio) / 2;
            const circleSize = Math.min(globalSpanX, globalSpanY) * 0.01;
            Plotly.relayout(plot, {
                'xaxis.autorange': false, 'yaxis.autorange': false,
                'xaxis.range': [targetX - halfSpanX, targetX + halfSpanX],
                'yaxis.range': [targetY - halfSpanY, targetY + halfSpanY],
                'shapes': [{
                    type: 'circle', xref: 'x', yref: 'y',
                    x0: targetX - circleSize, x1: targetX + circleSize,
                    y0: targetY - circleSize, y1: targetY + circleSize,
                    line: { color: UI_STYLE.primary, width: 2 }
                }]
            });
            searchInput.style.borderColor = "#18D85F";
            setTimeout(() => searchInput.style.borderColor = UI_STYLE.secondary, 1000);
        } else {
            searchInput.style.borderColor = "#FB2E46";
            setTimeout(() => searchInput.style.borderColor = UI_STYLE.secondary, 1000);
            showSearchMessage(`"${searchInput.value}" not found`, '#FB2E46');
        }
    }
};
controlBar.appendChild(createGroup("Search", searchInput));

// 6. Reset View
const resetBtn = document.createElement('button');
resetBtn.innerText = "Reset View";
resetBtn.style = `height: ${CONTROL_HEIGHT}px; padding: 0 16px; cursor: pointer; background:${UI_STYLE.primary}; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 12px; line-height: 1; white-space: nowrap; box-shadow: ${UI_STYLE.shadow}; transition: 0.2s;`;
resetBtn.onmouseover = () => resetBtn.style.background = UI_STYLE.primary;
resetBtn.onmouseout = () => resetBtn.style.background = UI_STYLE.primary;
resetBtn.onclick = () => {
    Plotly.relayout(plot, {'xaxis.autorange': true, 'yaxis.autorange': true, 'shapes': []});
};
controlBar.appendChild(createGroup("", resetBtn));

// 7. Mini-map
const MINI_MAP_WIDTH = 180;
const MINI_MAP_HEIGHT = 120;
const MINI_MAP_INSET = 30;
const MINI_MAP_PADDING = 5;

const miniMapContainer = document.createElement('div');
miniMapContainer.id = 'mini-map-container';
miniMapContainer.style = `
    position: absolute; width: ${MINI_MAP_WIDTH}px; height: ${MINI_MAP_HEIGHT}px;
    background: #F9F9F9; border: 1px solid rgba(16, 16, 16, 0.1); border-radius: 12px;
    z-index: 1000; pointer-events: none; box-shadow: none; padding: ${MINI_MAP_PADDING}px; backdrop-filter: blur(2px);
`;
plotFrame.appendChild(miniMapContainer);

const miniPlot = document.createElement('div');
miniPlot.style = "width: 100%; height: 100%;";
miniMapContainer.appendChild(miniPlot);

const miniData = plot.data.map(trace => ({
    x: trace.x, y: trace.y, mode: 'markers', type: 'scatter',
    marker: { size: 5, color: trace.marker.color, opacity: 0.5},
    hoverinfo: 'none'
}));

const miniLayout = {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    xaxis: { visible: false, fixedrange: true },
    yaxis: { visible: false, fixedrange: true },
    showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
};
Plotly.newPlot(miniPlot, miniData, miniLayout, {staticPlot: true});

const viewfinder = document.createElement('div');
viewfinder.style = `position: absolute; border: 2px solid ${UI_STYLE.accent}; background: rgba(19, 147, 253, 0.1); pointer-events: none; box-sizing: border-box;`;
miniMapContainer.appendChild(viewfinder);

const allX_vals = search_data.map(d => d.dim_0);
const allY_vals = search_data.map(d => d.dim_1);
const xAll = [Math.min(...allX_vals), Math.max(...allX_vals)];
const yAll = [Math.min(...allY_vals), Math.max(...allY_vals)];
const xSpan = xAll[1] - xAll[0];
const ySpan = yAll[1] - yAll[0];

const positionMiniMap = () => {
    const size = plot?._fullLayout?._size;
    if (!size) return;

    const left = size.l + size.w - MINI_MAP_WIDTH - MINI_MAP_INSET;
    const top = size.t + size.h - MINI_MAP_HEIGHT - MINI_MAP_INSET;

    miniMapContainer.style.left = `${Math.max(MINI_MAP_INSET, left)}px`;
    miniMapContainer.style.top = `${Math.max(MINI_MAP_INSET, top)}px`;
};

const updateViewfinder = () => {
    const fullX = plot._fullLayout.xaxis;
    const fullY = plot._fullLayout.yaxis;
    const xRange = fullX.range;
    const yRange = fullY.range;
    const w = MINI_MAP_WIDTH - (MINI_MAP_PADDING * 2);
    const h = MINI_MAP_HEIGHT - (MINI_MAP_PADDING * 2);
    const left = ((xRange[0] - xAll[0]) / xSpan) * w;
    const width = ((xRange[1] - xRange[0]) / xSpan) * w;
    const top = ((yAll[1] - yRange[1]) / ySpan) * h;
    const height = ((yRange[1] - yRange[0]) / ySpan) * h;
    viewfinder.style.left = (MINI_MAP_PADDING + Math.max(0, left)) + 'px';
    viewfinder.style.width = Math.min(w - left, width) + 'px';
    viewfinder.style.top = (MINI_MAP_PADDING + Math.max(0, top)) + 'px';
    viewfinder.style.height = Math.min(h - top, height) + 'px';
};

const syncMiniMapLayout = () => {
    window.requestAnimationFrame(() => {
        positionMiniMap();
        updateViewfinder();
    });
};

plot.on('plotly_afterplot', syncMiniMapLayout);
plot.on('plotly_relayout', syncMiniMapLayout);
window.addEventListener('resize', syncMiniMapLayout);
setTimeout(syncMiniMapLayout, 500);

// 8. Interaction (Discogs & Hover)
const styleTag = document.createElement('style');
styleTag.innerHTML = `.hover-pointer .nsewdrag { cursor: pointer !important; }`;
document.head.appendChild(styleTag);

plot.on('plotly_hover', () => plot.classList.add('hover-pointer'));
plot.on('plotly_unhover', () => plot.classList.remove('hover-pointer'));
plot.on('plotly_click', (data) => {
    const point = data.points[0];
    if (point && point.text) {
        const url = `https://www.discogs.com/search?type=masters&page=1&style_exact=${point.text.replace(/ /g, "+")}&sort=have%2Cdesc`;
        window.open(url, '_blank');
    }
});

// 9. Legend & Global Font
const globalStyle = document.createElement('style');
globalStyle.innerHTML = `
    .legendtext { font-family: 'Inter', sans-serif !important; font-size: 12px !important; font-weight: 600 !important; fill: ${UI_STYLE.textMain} !important; }
    .legend { border: 1px solid ${UI_STYLE.secondary} !important; }
`;
document.head.appendChild(globalStyle);
