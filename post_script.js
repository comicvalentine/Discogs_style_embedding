const plot = document.querySelector('.js-plotly-plot');
const plotHost = plot.parentNode;

// 0. UI Theme
const UI_STYLE = {
    bgPageFrame: '#F6F6F6',
    bgPlotFrame: 'white',
    bgPlot: 'white',
    bgMinimap: '#F9F9F9',
    primary: '#101010',
    secondary: '#E9ECEF',
    textMain: '#1A1A1A',
    textMuted: '#6C757D',
    shadow: '0 8px 24px rgba(0,0,0,0.08)',
    accent: '#1393FD',
    radius: '12px'
};
const CONTROL_HEIGHT = 30;
const COMPACT_LAYOUT_MAX_WIDTH = 760;

const applyButtonStyle = (btn, isActive, isAccent = false) => {
    btn.style = `padding: 0 14px; height: ${CONTROL_HEIGHT}px; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); white-space: nowrap;`;
    if (isActive) {
        btn.style.background = isAccent ? UI_STYLE.accent : UI_STYLE.primary;
        btn.style.color = "white";
    } else {
        btn.style.background = "white";
        btn.style.color = UI_STYLE.textMuted;
    }
};

const createGroup = (label, content) => {
    const group = document.createElement('div');
    group.style = "display: flex; flex-direction: column; gap: 6px;";
    if(label) {
        const labelSpan = document.createElement('span');
        labelSpan.innerText = label;
        labelSpan.style = `font-weight: 800; font-size: 10px; color: ${UI_STYLE.textMuted}; text-transform: uppercase;`;
        group.appendChild(labelSpan);
    }
    group.appendChild(content);
    return group;
};

// 1. Layout
const mainContainer = document.createElement('div');
mainContainer.style = `
    display: grid; 
    grid-template-columns: 1fr minmax(150px, 20%);
    grid-template-rows: auto 1fr;
    gap: 16px; 
    width: 100%; 
    padding: 20px; 
    background: ${UI_STYLE.bgPageFrame}; 
    border-radius: ${UI_STYLE.radius}; 
    font-family: 'Inter', sans-serif; 
    box-sizing: border-box;
    align-items: stretch;
`;
plotHost.insertBefore(mainContainer, plot);

const controlBar = document.createElement('div');
controlBar.style = "grid-column: 1 / -1; display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 4px;";
mainContainer.appendChild(controlBar);

const leftSection = document.createElement('div');
leftSection.style = "display: flex; flex-direction: column; min-width: 0;"; 
mainContainer.appendChild(leftSection);

const sidebar = document.createElement('div');
sidebar.style = `
    display: flex; 
    flex-direction: column; 
    gap: 16px; 
    background: white; 
    padding: 20px; 
    border-radius: ${UI_STYLE.radius}; 
    box-shadow: ${UI_STYLE.shadow};
    height: auto;
    box-sizing: border-box;
`;
const sidebarTitle = document.createElement('h3');
sidebarTitle.innerText = 'Overview';
sidebarTitle.style = `margin:0; font-size:11px; color:${UI_STYLE.textMuted}; text-transform:uppercase; letter-spacing:0.05em;`;
sidebar.appendChild(sidebarTitle);
mainContainer.appendChild(sidebar);

const plotFrame = document.createElement('div');
plotFrame.style = `position: relative; background: white; border-radius: ${UI_STYLE.radius}; box-shadow: ${UI_STYLE.shadow}; overflow: hidden;`; 
leftSection.appendChild(plotFrame);
plotFrame.appendChild(plot);



// 2. Controllers (Text, Mode, Reset)
const icon = {
    textOn: `
    <svg width="22" height="18" viewBox="0 0 22 18">
        <circle cx="5" cy="9" r="3" fill="currentColor"/>
        <text x="11" y="8" font-size="8" fill="currentColor" font-family="sans-serif">S</text>
    </svg>
    `,
    textOff: `
    <svg width="22" height="18" viewBox="0 0 22 18">
        <circle cx="5" cy="9" r="3" fill="currentColor"/>
    </svg>
    `,
    pan: `
    <svg width="10" height="10" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round">
    <path d="M12 2v20"/>
    <path d="M2 12h20"/>
    </svg>
    `,
    focus: `
    <svg width="12" height="12" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2">
    <rect x="4" y="4" width="16" height="16" rx="2"/>
    </svg>
    `
};

// TEXT
const textWrapper = document.createElement('div');
textWrapper.style = `
    display: flex;
    background: ${UI_STYLE.secondary};
    padding: 2px;
    border-radius: 8px;
    gap: 2px;
`;

const showBtn = document.createElement('button');
const hideBtn = document.createElement('button');

const updateTextUI = (show) => {
    applyButtonStyle(showBtn, show);
    applyButtonStyle(hideBtn, !show);

    showBtn.innerHTML = `${icon.textOn} ON`;
    hideBtn.innerHTML = `${icon.textOff} OFF`;

    [showBtn, hideBtn].forEach(btn => {
        const svg = btn.querySelector('svg');
        if (svg) svg.style.opacity = btn === showBtn && show ? "1" : "0.7";
    });
};

showBtn.onclick = () => {
    updateTextUI(true);
    Plotly.restyle(plot, {mode: 'markers+text'});
};

hideBtn.onclick = () => {
    updateTextUI(false);
    Plotly.restyle(plot, {mode: 'markers'});
};

updateTextUI(true);
textWrapper.append(showBtn, hideBtn);
controlBar.appendChild(createGroup("Text", textWrapper));


// MODE
const modeWrapper = document.createElement('div');
modeWrapper.style = `
    display: flex;
    background: ${UI_STYLE.secondary};
    padding: 2px;
    border-radius: 8px;
    gap: 2px;
`;

const moveBtn = document.createElement('button');
const focusBtn = document.createElement('button');

const updateModeUI = (isMove) => {
    applyButtonStyle(moveBtn, isMove);
    applyButtonStyle(focusBtn, !isMove);

    moveBtn.innerHTML = `${icon.pan} Move`;
    focusBtn.innerHTML = `${icon.focus} Focus`;

    [moveBtn, focusBtn].forEach(btn => {
        const svg = btn.querySelector('svg');
        if (svg) svg.style.opacity = btn === moveBtn && isMove ? "1" : "0.7";
    });
};

moveBtn.onclick = () => {
    updateModeUI(true);
    Plotly.relayout(plot, {dragmode: 'pan'});
};

focusBtn.onclick = () => {
    updateModeUI(false);
    Plotly.relayout(plot, {dragmode: 'zoom'});
};

updateModeUI(true);
modeWrapper.append(moveBtn, focusBtn);
controlBar.appendChild(createGroup("Mode", modeWrapper));

const resetWrapper = document.createElement('div');
resetWrapper.style = `display: flex; background: ${UI_STYLE.secondary}; padding: 2px; border-radius: 8px; gap: 2px;`;
const resetBtn = document.createElement('button');
resetBtn.innerText = "Reset View";
applyButtonStyle(resetBtn, true);
resetBtn.onclick = () => {
    const fl = plot._fullLayout;

    // 현재 화면 비율 (pixel 기준)
    const xLen = fl.xaxis._length;
    const yLen = fl.yaxis._length;
    const aspect = xLen / yLen;

    // 1. y: autorange
    Plotly.relayout(plot, {
        'yaxis.autorange': true
    }).then(() => {
        const fl2 = plot._fullLayout;
        const yr = fl2.yaxis.range;

        const ySpan = yr[1] - yr[0];

        // 2. x: scale anchor to y
        const xSpan = ySpan * aspect;

        const xCenter = (fl2.xaxis.range[0] + fl2.xaxis.range[1]) / 2;

        const newX = [
            xCenter - xSpan / 2,
            xCenter + xSpan / 2
        ];

        Plotly.relayout(plot, {
            'xaxis.range': newX,
            'xaxis.autorange': false,
            'shapes': []
        });
    });
};
resetWrapper.append(resetBtn)
controlBar.appendChild(createGroup("\u00A0", resetWrapper));

// 3. Search Style
const searchInput = document.createElement('input');
searchInput.placeholder = "Search style...";
searchInput.style = `height: ${CONTROL_HEIGHT}px; padding: 0 12px; border: 2px solid ${UI_STYLE.secondary}; border-radius: 8px; width: 180px; outline: none; font-size: 13px; transition: 0.2s;`;
const normalizeStyleSearch = (value) => (value || "").toLowerCase().replace(/[-_\s]+/g, '');


const showSearchMessage = (message, color) => {
    const msg = document.createElement('div');
    msg.innerText = message;
    Object.assign(msg.style, {
        position: 'absolute', top: '-35px', left: '0', background: color, color: 'white',
        padding: '4px 10px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold', zIndex: '1000'
    });
    searchInput.parentElement.style.position = 'relative';
    searchInput.parentElement.appendChild(msg);
    setTimeout(() => msg.remove(), 2000);
};

const allX_vals = search_data.map(d => Number(d.dim_0));
const allY_vals = search_data.map(d => Number(d.dim_1));
const xAll = [Math.min(...allX_vals), Math.max(...allX_vals)];
const yAll = [Math.min(...allY_vals), Math.max(...allY_vals)];
const xSpan = xAll[1] - xAll[0];
const ySpan = yAll[1] - yAll[0];

searchInput.onkeypress = (e) => {
    if (e.key === 'Enter') {
        const term = normalizeStyleSearch(searchInput.value);
        const matches = search_data.filter(row => normalizeStyleSearch(row.style) === term);
        if (matches.length > 0) {
            const target = matches[0];
            const tX = Number(target.dim_0), tY = Number(target.dim_1);
            const zoomRatio = 0.12; 
            const hX = xSpan * zoomRatio / 2, hY = ySpan * zoomRatio / 2;
            const fl = plot._fullLayout;
            const xUnit = (hX * 2) / fl.xaxis._length, yUnit = (hY * 2) / fl.yaxis._length;
            const r = 30; // 30px 정원

            Plotly.relayout(plot, {
                'xaxis.range': [tX - hX, tX + hX], 'yaxis.range': [tY - hY, tY + hY],
                'shapes': [{
                    type: 'circle', xref: 'x', yref: 'y', x0: tX - r*xUnit, x1: tX + r*xUnit, y0: tY - r*yUnit, y1: tY + r*yUnit,
                    fillcolor: 'rgba(19, 147, 253, 0.3)', line: {width: 0}, layer: 'below'
                }]
            });
            let count = 0;
            if (window.searchBlinkTimer) clearInterval(window.searchBlinkTimer);
            window.searchBlinkTimer = setInterval(() => {
                count++;
                Plotly.relayout(plot, { 'shapes[0].opacity': count % 2 === 0 ? 1 : 0 });
                if (count >= 8) { clearInterval(window.searchBlinkTimer); setTimeout(() => Plotly.relayout(plot, {shapes: []}), 2000); }
            }, 400);
        } else {
            showSearchMessage("Not found", "#FB2E46");
        }
    }
};
controlBar.appendChild(createGroup("Search", searchInput));

// 4. Zoom
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


// 5. Mini-map
const MINI_MAP_WIDTH_PERCENT = 100;
const miniMapWrapper = document.createElement('div');
miniMapWrapper.style = `width: 100%; background: ${UI_STYLE.bgMinimap}; border-radius: 8px; position: relative; overflow: hidden; border: 1px solid ${UI_STYLE.secondary}; aspect-ratio: 4 / 3;`;
sidebar.appendChild(miniMapWrapper);

const miniPlot = document.createElement('div');
miniPlot.style = "width: 100%; height: 100%;";
miniMapWrapper.appendChild(miniPlot);

const viewfinder = document.createElement('div');
viewfinder.style = `position: absolute; border: 1.5px solid ${UI_STYLE.accent}; background: rgba(19, 147, 253, 0.05); pointer-events: none; z-index: 10;`;
miniMapWrapper.appendChild(viewfinder);

const miniData = plot.data.map(trace => ({
    x: trace.x, y: trace.y, mode: 'markers', type: 'scatter',
    marker: { size: 5, color: trace.marker.color, opacity: 0.4},
    hoverinfo: 'none'
}));

Plotly.newPlot(miniPlot, miniData, {
    margin: { t: 0, b: 0, l: 0, r: 0 },
    xaxis: { visible: false, fixedrange: true, range: xAll, autorange: false },
    yaxis: { visible: false, fixedrange: true, range: yAll, autorange: false },
    showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
}, {staticPlot: true});

const updateMiniMapAndViewfinder = () => {
    const fl = plot._fullLayout;
    if (!fl) return;
    const miniW = miniMapWrapper.clientWidth;
    const miniH = miniMapWrapper.clientHeight;
    if (!miniW || !miniH) return;

    Plotly.relayout(miniPlot, {
        width: miniW,
        height: miniH,
        'xaxis.range': xAll,
        'yaxis.range': yAll,
        'xaxis.autorange': false,
        'yaxis.autorange': false
    });

    const fullX = fl.xaxis;
    const fullY = fl.yaxis;
    
    const left = ((fullX.range[0] - xAll[0]) / xSpan) * miniW;
    const width = ((fullX.range[1] - fullX.range[0]) / xSpan) * miniW;
    const top = ((yAll[1] - fullY.range[1]) / ySpan) * miniH;
    const height = ((fullY.range[1] - fullY.range[0]) / ySpan) * miniH;
    
    viewfinder.style.left = Math.max(0, left) + 'px';
    viewfinder.style.width = Math.min(miniW, width) + 'px';
    viewfinder.style.top = Math.max(0, top) + 'px';
    viewfinder.style.height = Math.min(miniH, height) + 'px';
};

// 6. Legend & Sync
const isCompactLayout = () => (mainContainer.clientWidth || window.innerWidth) <= COMPACT_LAYOUT_MAX_WIDTH;

const applyResponsiveLayout = () => {
    const compact = isCompactLayout();
    const legend = sidebar.querySelector('.custom-legend-container');

    document.documentElement.style.height = compact ? 'auto' : '';
    document.documentElement.style.overflowY = compact ? 'auto' : '';
    document.body.style.height = compact ? 'auto' : '';
    document.body.style.minHeight = compact ? '100%' : '';
    document.body.style.overflowY = compact ? 'auto' : '';

    mainContainer.style.gridTemplateColumns = compact ? '1fr' : '1fr minmax(150px, 20%)';
    mainContainer.style.gridTemplateRows = compact ? 'auto 1fr auto' : 'auto 1fr';
    mainContainer.style.padding = compact ? '12px' : '20px';
    mainContainer.style.paddingBottom = compact ? '32px' : '20px';

    leftSection.style.gridColumn = compact ? '1 / -1' : '';
    sidebar.style.gridColumn = compact ? '1 / -1' : '';
    sidebar.style.flexDirection = compact ? 'row' : 'column';
    sidebar.style.alignItems = compact ? 'flex-start' : '';
    sidebar.style.gap = compact ? '12px' : '16px';
    sidebar.style.padding = compact ? '12px' : '20px';
    sidebarTitle.style.display = compact ? 'none' : '';

    miniMapWrapper.style.width = compact ? '42%' : '100%';
    miniMapWrapper.style.minWidth = compact ? '130px' : '';
    miniMapWrapper.style.maxWidth = compact ? '220px' : '';
    miniMapWrapper.style.flex = compact ? '0 0 42%' : '';

    if (legend) {
        legend.style.display = compact ? 'grid' : 'flex';
        legend.style.gridTemplateColumns = compact ? 'repeat(2, minmax(0, 1fr))' : '';
        legend.style.flex = compact ? '1 1 0' : '';
        legend.style.minWidth = compact ? '0' : '';
        legend.style.maxHeight = compact ? 'none' : '300px';
        legend.style.overflowY = compact ? 'visible' : 'auto';
        legend.style.marginTop = compact ? '0' : '10px';
    }
};

const addExternalLegend = () => {
    const old = sidebar.querySelector('.custom-legend-container'); if (old) old.remove();
    const wrapper = document.createElement('div');
    wrapper.className = 'custom-legend-container';
    wrapper.style = "display: flex; flex-direction: column; gap: 8px; max-height: 300px; overflow-y: auto; margin-top: 10px;";
    plot.data.forEach((trace, index) => {
        const item = document.createElement('div');
        item.style = "display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 12px;";
        item.innerHTML = `<div style="width:10px; height:10px; border-radius:2px; background:${trace.marker.color}"></div><span>${trace.name}</span>`;
        item.onclick = () => {
            const vis = trace.visible === 'legendonly' ? true : 'legendonly';
            Plotly.restyle(plot, { visible: vis }, [index]);
            item.style.opacity = vis === true ? '1' : '0.4';
        };
        wrapper.appendChild(item);
    });
    sidebar.appendChild(wrapper);
    applyResponsiveLayout();
};


const resizePlot = () => {
    applyResponsiveLayout();
    const parent = plotFrame;
    const nextWidth = parent.clientWidth;
    const nextHeight = isCompactLayout()
        ? Math.max(420, window.innerHeight - 320)
        : Math.max(600, window.innerHeight - 250);
    const fl = plot._fullLayout;

    const currentXRange = fl?.xaxis?.range ? [...fl.xaxis.range] : [...xAll];
    const currentYRange = fl?.yaxis?.range ? [...fl.yaxis.range] : [...yAll];
    const centerX = (currentXRange[0] + currentXRange[1]) / 2;
    const centerY = (currentYRange[0] + currentYRange[1]) / 2;
    const xUnitsPerPixel = fl?.xaxis?._length
        ? (currentXRange[1] - currentXRange[0]) / fl.xaxis._length
        : xSpan / Math.max(nextWidth, 1);
    const yUnitsPerPixel = fl?.yaxis?._length
        ? (currentYRange[1] - currentYRange[0]) / fl.yaxis._length
        : ySpan / Math.max(nextHeight, 1);

    return Plotly.relayout(plot, {
        width: nextWidth,
        height: nextHeight,
        showlegend: false
    }).then(() => {
        const resizedLayout = plot._fullLayout;
        if (!resizedLayout?.xaxis?._length || !resizedLayout?.yaxis?._length) return;

        const nextHalfX = (xUnitsPerPixel * resizedLayout.xaxis._length) / 2;
        const nextHalfY = (yUnitsPerPixel * resizedLayout.yaxis._length) / 2;

        return Plotly.relayout(plot, {
            'xaxis.range': [centerX - nextHalfX, centerX + nextHalfX],
            'yaxis.range': [centerY - nextHalfY, centerY + nextHalfY],
            'xaxis.autorange': false,
            'yaxis.autorange': false
        });
    }).then(() => {
        updateMiniMapAndViewfinder();
    });
};

const syncAll = () => resizePlot();

plot.on('plotly_relayout', updateMiniMapAndViewfinder);
window.addEventListener('resize', syncAll);


setTimeout(() => {
    addExternalLegend();
    Plotly.relayout(plot, {
        margin: { t: 20, b: 20, l: 20, r: 20 },
        paper_bgcolor: UI_STYLE.bgPlotFrame,
        plot_bgcolor: UI_STYLE.bgPlotFrame,
        showlegend: false
    }).then(syncAll);
}, 500);

// 7. Interaction
const styleTag = document.createElement('style');
styleTag.innerHTML = `.hover-pointer .nsewdrag { cursor: pointer !important; }`;
document.head.appendChild(styleTag);

plot.on('plotly_hover', () => plot.classList.add('hover-pointer'));
plot.on('plotly_unhover', () => plot.classList.remove('hover-pointer'));
plot.on('plotly_click', (data) => {
    const point = data.points[0];
    if (point && point.text) {
        const url = `https://www.discogs.com/search?type=master&page=1&style_exact=${point.text.replace(/ /g, "+")}&sort=have%2Cdesc`;
        window.open(url, '_blank');
    }
});
