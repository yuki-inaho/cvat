const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..', '..', '..');

function loadEnv(envPath = path.join(repoRoot, '.env')) {
    const env = {};
    if (!fs.existsSync(envPath)) return env;
    const lines = fs.readFileSync(envPath, 'utf8').split('\n');
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const idx = trimmed.indexOf('=');
        if (idx > 0) {
            env[trimmed.slice(0, idx).trim()] = trimmed.slice(idx + 1).trim();
        }
    }
    return env;
}

function timestamp() {
    return new Date().toISOString().replace(/[-T:.Z]/g, '').slice(0, 14);
}

function makeRunDir(prefix) {
    const runDir = path.join(repoRoot, 'temp', 'e2e_sam2', `${prefix}_${timestamp()}`);
    fs.mkdirSync(runDir, { recursive: true });
    return runDir;
}

function authStatePath() {
    return path.join(repoRoot, 'temp', 'e2e_sam2', 'check', 'auth-state.json');
}

function requireAuthState() {
    const statePath = authStatePath();
    if (!fs.existsSync(statePath)) {
        throw new Error(`Auth state not found at ${statePath}. Run 'just e2e-login' first.`);
    }
    return statePath;
}

async function getCanvasHashes(page) {
    return page.evaluate(() => {
        function hashBytes(data) {
            let hash = 2166136261;
            let nonTransparent = 0;
            let nonWhite = 0;
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const a = data[i + 3];
                if (a > 0) nonTransparent += 1;
                if (a > 0 && !(r > 245 && g > 245 && b > 245)) nonWhite += 1;
                hash ^= r; hash = Math.imul(hash, 16777619);
                hash ^= g; hash = Math.imul(hash, 16777619);
                hash ^= b; hash = Math.imul(hash, 16777619);
                hash ^= a; hash = Math.imul(hash, 16777619);
            }
            return { hash: (hash >>> 0).toString(16), nonTransparent, nonWhite };
        }

        return Array.from(document.querySelectorAll('.cvat-canvas-container canvas')).map((canvas, index) => {
            try {
                const ctx = canvas.getContext('2d', { willReadFrequently: true });
                if (!ctx || canvas.width === 0 || canvas.height === 0) {
                    return { index, width: canvas.width, height: canvas.height, readable: false };
                }
                const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
                return {
                    index,
                    width: canvas.width,
                    height: canvas.height,
                    readable: true,
                    ...hashBytes(image.data),
                };
            } catch (error) {
                return {
                    index,
                    width: canvas.width,
                    height: canvas.height,
                    readable: false,
                    error: error instanceof Error ? error.message : String(error),
                };
            }
        });
    });
}

function changedCanvasCount(before, after) {
    const beforeByIndex = new Map(before.map((item) => [item.index, item]));
    let changed = 0;
    for (const item of after) {
        const prev = beforeByIndex.get(item.index);
        if (!prev) continue;
        if (prev.hash !== item.hash || prev.nonWhite !== item.nonWhite || prev.nonTransparent !== item.nonTransparent) {
            changed += 1;
        }
    }
    return changed;
}

async function getInteractionPointStats(page) {
    return page.evaluate(() => {
        const points = Array.from(document.querySelectorAll('svg .cvat_interaction_point'));
        const normalized = points.map((node) => ({
            stroke: (node.getAttribute('stroke') || '').toLowerCase(),
            fill: (node.getAttribute('fill') || '').toLowerCase(),
            cx: Number(node.getAttribute('cx')),
            cy: Number(node.getAttribute('cy')),
        }));
        return {
            total: normalized.length,
            positive: normalized.filter((p) => p.stroke.includes('green')).length,
            negative: normalized.filter((p) => p.stroke.includes('red')).length,
            points: normalized,
        };
    });
}

async function getAnnotationSummary(page, host, jobId) {
    const response = await page.request.get(`${host}/api/jobs/${jobId}/annotations`);
    const body = await response.text();
    let data = null;
    try {
        data = JSON.parse(body);
    } catch (_) {
        // keep null data and raw body for diagnostics
    }
    const shapes = data?.shapes || [];
    const masks = shapes.filter((shape) => shape.type === 'mask');
    const semiAuto = shapes.filter((shape) => shape.source === 'semi-auto');
    return {
        status: response.status(),
        ok: response.ok(),
        shapeCount: shapes.length,
        maskCount: masks.length,
        semiAutoCount: semiAuto.length,
        maskBounds: masks.map((shape) => Array.isArray(shape.points) ? shape.points.slice(-4) : []).filter((bounds) => bounds.length === 4),
        rawBodyLength: body.length,
        rawBodySnippet: body.slice(0, 1000),
    };
}

async function clearJobAnnotations(page, host, jobId, enabled = true) {
    if (!enabled) {
        return { skipped: true, success: true };
    }

    const before = await getAnnotationSummary(page, host, jobId);
    const cookies = await page.context().cookies(host);
    const csrfToken = cookies.find((cookie) => cookie.name === 'csrftoken')?.value || '';
    const response = await page.request.delete(`${host}/api/jobs/${jobId}/annotations`, {
        headers: csrfToken ? { 'X-CSRFToken': csrfToken } : {},
    });
    const body = await response.text();
    const after = await getAnnotationSummary(page, host, jobId);

    return {
        skipped: false,
        success: response.status() === 204 && after.shapeCount === 0,
        status: response.status(),
        csrfHeaderSent: Boolean(csrfToken),
        before,
        after,
        bodySnippet: body.slice(0, 1000),
    };
}

async function finishAndSave(page, runDir, host, jobId, prefix) {
    const result = { clickedDone: false, clickedSave: false, annotationsAfterFinish: null };

    const doneButton = await page.$('.cvat-annotation-header-done-button');
    if (!doneButton) {
        result.error = 'Done button not found';
        return result;
    }
    await doneButton.click();
    result.clickedDone = true;
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(runDir, `${prefix}_after_done.png`), fullPage: true }).catch(() => {});

    const saveButton = await page.$('.cvat-annotation-header-save-button');
    if (saveButton) {
        await saveButton.click();
        result.clickedSave = true;
        await page.waitForTimeout(2500);
    }

    result.annotationsAfterFinish = await getAnnotationSummary(page, host, jobId);
    fs.writeFileSync(
        path.join(runDir, `${prefix}_annotations_after_finish.json`),
        JSON.stringify(result.annotationsAfterFinish, null, 2),
    );

    return result;
}

async function openAiTools(page) {
    const aiToolsButton = await page.$('.cvat-tools-control');
    if (!aiToolsButton) {
        throw new Error('AI Tools button not found');
    }
    await aiToolsButton.click();
    await page.waitForTimeout(1000);
}

async function selectSam2Interactor(page) {
    const currentSelection = await page.$eval(
        '.ant-popover .ant-select-selection-item, .ant-popover-content .ant-select-selection-item',
        (el) => el.textContent || '',
    ).catch(() => '');

    if (currentSelection.includes('2.1')) {
        return { changed: false, currentSelection };
    }

    const popoverSelects = await page.$$('.ant-popover .ant-select, .ant-popover-content .ant-select');
    const interactorSelect = popoverSelects.length >= 2 ? popoverSelects[1] : popoverSelects[0];
    if (!interactorSelect) {
        throw new Error('Interactor select not found');
    }
    await interactorSelect.click();
    await page.waitForTimeout(500);

    const sam2Option = await page.$('.ant-select-item-option:has-text("Segment Anything 2.1")');
    if (!sam2Option) {
        await page.keyboard.press('Escape');
        throw new Error('Segment Anything 2.1 option not found');
    }
    await sam2Option.click();
    await page.waitForTimeout(500);
    await openAiTools(page);
    return { changed: true, currentSelection };
}

async function setStartWithBBox(page, enabled) {
    return page.evaluate((shouldEnable) => {
        const divs = document.querySelectorAll('.cvat-tools-interactor-setups div');
        for (const div of divs) {
            if (div.textContent && div.textContent.includes('Start with a bounding box')) {
                const sw = div.querySelector('.ant-switch');
                if (!sw) return { found: true, toggled: false, error: 'switch element not found' };
                const isChecked = sw.classList.contains('ant-switch-checked');
                if (isChecked !== shouldEnable) {
                    sw.click();
                    return { found: true, toggled: true, before: isChecked, after: shouldEnable };
                }
                return { found: true, toggled: false, before: isChecked, after: isChecked };
            }
        }
        return { found: false, toggled: false, error: 'Start with a bounding box switch not found' };
    }, enabled);
}

async function clickInteract(page) {
    const button = await page.$('.cvat-tools-interact-button');
    if (!button) {
        throw new Error('Interact button not found');
    }
    const isDisabled = await button.evaluate((el) => el.disabled || el.classList.contains('ant-btn-disabled'));
    if (isDisabled) {
        throw new Error('Interact button is disabled');
    }
    await button.click();
    await page.waitForTimeout(1500);
}

module.exports = {
    repoRoot,
    loadEnv,
    timestamp,
    makeRunDir,
    authStatePath,
    requireAuthState,
    getCanvasHashes,
    changedCanvasCount,
    getInteractionPointStats,
    getAnnotationSummary,
    clearJobAnnotations,
    finishAndSave,
    openAiTools,
    selectSam2Interactor,
    setStartWithBBox,
    clickInteract,
};
