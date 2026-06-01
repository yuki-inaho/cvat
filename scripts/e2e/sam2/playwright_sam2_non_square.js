#!/usr/bin/env node
/**
 * SAM2 Non-Square Image E2E test for CVAT.
 *
 * Tests: Use task 182/job 181 (640x360 image) to verify that
 * coordinate scaling between canvas, image natural size, SAM2 input (1024x1024),
 * and mask output does not break for non-square images.
 *
 * Key concern: getModelScale() maps (w,h) to (1024/w, 1024/h).
 * For 640x360: scaleX=1.6, scaleY≈2.844. If the encoder or decoder
 * assumes square input, coordinates will be distorted.
 *
 * Trace: TR-08, TR-10
 *
 * Usage:
 *   NODE_PATH=~/temp/playwright-cli/node_modules node scripts/e2e/sam2/playwright_sam2_non_square.js
 */
const fs = require('fs');
const path = require('path');
const http = require('http');
const {
    repoRoot,
    loadEnv,
    makeRunDir,
    requireAuthState,
    getInteractionPointStats,
    getAnnotationSummary,
    clearJobAnnotations,
    finishAndSave,
    openAiTools,
    selectSam2Interactor,
    setStartWithBBox,
    clickInteract,
} = require('./e2e_utils');

function httpRequest(url, opts, body) {
    return new Promise((resolve, reject) => {
        const mod = url.startsWith('https') ? require('https') : http;
        const req = mod.request(url, opts, (res) => {
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                resolve({
                    status: res.statusCode,
                    headers: res.headers,
                    body: Buffer.concat(chunks).toString(),
                });
            });
        });
        req.on('error', reject);
        if (body) req.write(body);
        req.end();
    });
}

function parseCookies(setCookieHeaders) {
    const cookies = {};
    if (!setCookieHeaders) return cookies;
    const arr = Array.isArray(setCookieHeaders) ? setCookieHeaders : [setCookieHeaders];
    for (const h of arr) {
        const parts = h.split(';')[0].split('=');
        if (parts.length >= 2) cookies[parts[0].trim()] = parts.slice(1).join('=').trim();
    }
    return cookies;
}

async function main() {
    const env = loadEnv(path.join(repoRoot, '.env'));
    const host = env.CVAT_E2E_HOST || 'http://localhost:8080';
    const username = env.CVAT_E2E_USER;
    const password = env.CVAT_E2E_PASSWORD;
    const shouldClearAnnotations = env.CVAT_E2E_CLEAR_ANNOTATIONS !== '0';

    if (!username || !password) {
        console.error('ERROR: CVAT_E2E_USER and CVAT_E2E_PASSWORD must be set in .env');
        process.exit(1);
    }

    const authStatePath = requireAuthState();
    const runDir = makeRunDir('run_nonsquare');

    // Non-square task: 640x360
    const TASK_ID = 182;
    const JOB_ID = 181;
    const FUNC_ID = 'ort-facebookresearch-sam2-hiera-base-plus';
    const JOB_URL = `${host}/tasks/${TASK_ID}/jobs/${JOB_ID}`;
    const LAMBDA_URL = `${host}/api/lambda/functions/${FUNC_ID}`;
    const IMAGE_WIDTH = 640;
    const IMAGE_HEIGHT = 360;

    console.log('=== SAM2 Non-Square Image E2E ===');
    console.log(`Run dir: ${runDir}`);
    console.log(`Job URL: ${JOB_URL}`);
    console.log(`Image: ${IMAGE_WIDTH}x${IMAGE_HEIGHT}`);
    console.log('');

    const results = {
        timestamp: new Date().toISOString(),
        runDir,
        imageSize: { width: IMAGE_WIDTH, height: IMAGE_HEIGHT },
        steps: {},
        overall: 'UNKNOWN',
    };

    // ========================================
    // Phase 1: API-based lambda invoke with non-square image
    // ========================================
    console.log('--- Phase 1: API lambda invoke (non-square) ---');

    // Login
    const loginResp = await httpRequest(`${host}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    }, JSON.stringify({ username, password }));

    if (loginResp.status !== 200) {
        console.error(`Login failed: HTTP ${loginResp.status}`);
        process.exit(1);
    }
    const allCookies = parseCookies(loginResp.headers['set-cookie']);
    const cookieStr = Object.entries(allCookies).map(([k, v]) => `${k}=${v}`).join('; ');
    const csrfToken = allCookies.csrftoken || '';

    // Invoke with BBox covering part of the 640x360 image
    // Use coordinates that would expose scaling issues
    const bboxPayload = {
        task: TASK_ID,
        job: JOB_ID,
        frame: 0,
        pos_points: [],
        neg_points: [],
        obj_bbox: [[100, 50], [400, 250]],
    };
    fs.writeFileSync(path.join(runDir, 'api_bbox_request.json'), JSON.stringify(bboxPayload, null, 2));

    console.log('  Invoking lambda with bbox on 640x360 image...');
    const bboxResp = await httpRequest(LAMBDA_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Cookie': cookieStr,
            'X-CSRFToken': csrfToken,
        },
        timeout: 120000,
    }, JSON.stringify(bboxPayload));

    console.log(`  BBox invoke: HTTP ${bboxResp.status}`);
    fs.writeFileSync(path.join(runDir, 'api_bbox_response.json'), bboxResp.body);

    let bboxSuccess = false;
    if (bboxResp.status === 200) {
        try {
            const data = JSON.parse(bboxResp.body);
            const hasEmbeddings = data.image_embed && data.high_res_feats_0 && data.high_res_feats_1;
            if (hasEmbeddings) {
                console.log('  Embeddings present:');
                console.log(`    image_embed length: ${data.image_embed.length}`);
                console.log(`    high_res_feats_0 length: ${data.high_res_feats_0.length}`);
                console.log(`    high_res_feats_1 length: ${data.high_res_feats_1.length}`);
                bboxSuccess = true;
            }
        } catch (e) {
            console.error(`  Parse error: ${e.message}`);
        }
    } else {
        console.error(`  FAILED: HTTP ${bboxResp.status}`);
    }
    results.steps.apiBbox = { success: bboxSuccess, status: bboxResp.status };

    // Also test with positive point via API
    const pointPayload = {
        task: TASK_ID,
        job: JOB_ID,
        frame: 0,
        pos_points: [[250, 150]], // Center-ish of the 640x360 image
        neg_points: [],
        obj_bbox: [],
    };
    fs.writeFileSync(path.join(runDir, 'api_point_request.json'), JSON.stringify(pointPayload, null, 2));

    console.log('  Invoking lambda with positive point on 640x360 image...');
    const pointResp = await httpRequest(LAMBDA_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Cookie': cookieStr,
            'X-CSRFToken': csrfToken,
        },
        timeout: 120000,
    }, JSON.stringify(pointPayload));

    console.log(`  Point invoke: HTTP ${pointResp.status}`);
    fs.writeFileSync(path.join(runDir, 'api_point_response.json'), pointResp.body);

    let pointSuccess = false;
    if (pointResp.status === 200) {
        try {
            const data = JSON.parse(pointResp.body);
            const hasEmbeddings = data.image_embed && data.high_res_feats_0 && data.high_res_feats_1;
            if (hasEmbeddings) {
                console.log('  Embeddings present for point invoke');
                pointSuccess = true;
            }
        } catch (e) {
            console.error(`  Parse error: ${e.message}`);
        }
    }
    results.steps.apiPoint = { success: pointSuccess, status: pointResp.status };

    // ========================================
    // Phase 2: Playwright UI test with non-square image
    // ========================================
    console.log('');
    console.log('--- Phase 2: Playwright UI (non-square) ---');

    const { chromium } = require('playwright');
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        storageState: authStatePath,
        viewport: { width: 1920, height: 1080 },
    });
    const page = await context.newPage();
    const clearResult = await clearJobAnnotations(page, host, JOB_ID, shouldClearAnnotations);
    console.log(`  Clear annotations: ${JSON.stringify(clearResult)}`);
    results.steps.clearAnnotations = clearResult;

    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({ type: msg.type(), text: msg.text() });
    });

    const lambdaResponses = [];
    const networkLogs = [];
    page.on('response', async (response) => {
        const entry = { url: response.url(), status: response.status(), method: response.request().method() };
        networkLogs.push(entry);
        if (response.url().includes('/api/lambda/')) {
            try {
                const body = await response.text();
                lambdaResponses.push({ ...entry, bodyLength: body.length });
            } catch (_) {}
        }
    });

    let phase2Success = false;
    try {
        // Navigate to job page
        console.log('  Opening job page...');
        await page.goto(JOB_URL, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);
        await page.screenshot({ path: path.join(runDir, 'job_loaded.png'), fullPage: true });
        const annotationsBefore = await getAnnotationSummary(page, host, JOB_ID);
        results.steps.annotationsBefore = annotationsBefore;

        // Verify canvas and image dimensions
        const canvasInfo = await page.evaluate(() => {
            const canvas = document.querySelector('#cvat_canvas_background');
            const container = document.querySelector('.cvat-canvas-container');
            return {
                canvasWidth: canvas?.width,
                canvasHeight: canvas?.height,
                canvasNaturalWidth: canvas?.naturalWidth,
                canvasNaturalHeight: canvas?.naturalHeight,
                containerRect: container?.getBoundingClientRect() ? {
                    x: container.getBoundingClientRect().x,
                    y: container.getBoundingClientRect().y,
                    width: container.getBoundingClientRect().width,
                    height: container.getBoundingClientRect().height,
                } : null,
            };
        });
        console.log(`  Canvas info: ${JSON.stringify(canvasInfo)}`);
        results.steps.canvasInfo = canvasInfo;

        // Coordinate scaling analysis
        const scaleX = 1024 / IMAGE_WIDTH;  // 1.6
        const scaleY = 1024 / IMAGE_HEIGHT; // ~2.844
        console.log(`  SAM2 model scale: scaleX=${scaleX.toFixed(4)}, scaleY=${scaleY.toFixed(4)}`);
        console.log(`  Aspect ratio: ${(IMAGE_WIDTH / IMAGE_HEIGHT).toFixed(4)} (non-square)`);
        console.log(`  Scale ratio: ${(scaleX / scaleY).toFixed(4)} (1.0 = isotropic)`);
        results.steps.scaleAnalysis = {
            scaleX,
            scaleY,
            aspectRatio: IMAGE_WIDTH / IMAGE_HEIGHT,
            scaleRatio: scaleX / scaleY,
            isIsotropic: Math.abs(scaleX - scaleY) < 0.001,
            note: 'Non-isotropic scaling means x and y are stretched differently to 1024x1024',
        };

        // Open AI Tools and start interaction
        console.log('  Opening AI Tools...');
        const aiToolsBtn = await page.$('.cvat-tools-control');
        if (aiToolsBtn) {
            await openAiTools(page);

            // Select SAM 2.1 if not already selected
            const selectionResult = await selectSam2Interactor(page);
            console.log(`  SAM2 selection result: ${JSON.stringify(selectionResult)}`);

            // Toggle "Start with bounding box" OFF via JS
            const switchResult = await setStartWithBBox(page, false);
            console.log(`  Switch toggle: ${JSON.stringify(switchResult)}`);
            await page.waitForTimeout(500);

            // Re-open popover (switch toggle may have closed it) then click Interact
            await openAiTools(page);
            await clickInteract(page);

                    // Click on canvas center
                    const canvasWrapper = await page.$('.cvat-canvas-container');
                    if (canvasWrapper) {
                        const bbox = await canvasWrapper.boundingBox();
                        const clickX = bbox.x + bbox.width / 2;
                        const clickY = bbox.y + bbox.height / 2;
                        console.log(`  Clicking at: (${clickX.toFixed(0)}, ${clickY.toFixed(0)})`);

                        const lambdaBefore = lambdaResponses.length;
                        const pointsBefore = await getInteractionPointStats(page);
                        await page.mouse.click(clickX, clickY, { button: 'left' });

                        // Wait for response
                        for (let i = 0; i < 30; i++) {
                            await page.waitForTimeout(1000);
                            if (lambdaResponses.length > lambdaBefore) break;
                        }
                        await page.waitForTimeout(2000);
                        await page.screenshot({ path: path.join(runDir, 'after_click.png'), fullPage: true });

                        const pointStatsAfterClick = await getInteractionPointStats(page);
                        const afterClick = await page.evaluate(() => {
                            const shapes = document.querySelectorAll('.cvat_canvas_shape, .cvat_canvas_shape_mask');
                            return { shapeCount: shapes.length };
                        });
                        console.log(`  After click: shapes=${afterClick.shapeCount}, positivePoints=${pointStatsAfterClick.positive}`);
                        results.steps.uiInteraction = {
                            success: pointStatsAfterClick.positive > pointsBefore.positive,
                            lambdaReceived: lambdaResponses.length > lambdaBefore,
                            beforePoints: pointsBefore,
                            afterPoints: pointStatsAfterClick,
                            ...afterClick,
                        };
                        phase2Success = results.steps.uiInteraction.success;

                        const finishResult = await finishAndSave(page, runDir, host, JOB_ID, 'nonsquare');
                        results.steps.finishAndSave = finishResult;
                    }
        }
    } catch (err) {
        console.error(`  Playwright error: ${err.message}`);
        await page.screenshot({ path: path.join(runDir, 'error.png'), fullPage: true }).catch(() => {});
        results.steps.uiInteraction = { success: false, error: err.message };
    }

    // Save artifacts
    fs.writeFileSync(path.join(runDir, 'console_logs.json'), JSON.stringify(consoleLogs, null, 2));
    fs.writeFileSync(path.join(runDir, 'network_logs.json'), JSON.stringify(networkLogs, null, 2));
    fs.writeFileSync(path.join(runDir, 'lambda_responses.json'), JSON.stringify(lambdaResponses, null, 2));

    // ========================================
    // Phase 3: Check annotations API for the non-square job
    // ========================================
    console.log('');
    console.log('--- Phase 3: Annotations check ---');
    const annotationSummary = await getAnnotationSummary(page, host, JOB_ID);
    fs.writeFileSync(path.join(runDir, 'annotations.json'), JSON.stringify(annotationSummary, null, 2));
    console.log(`  Annotations: ${annotationSummary.shapeCount} shapes, ${annotationSummary.maskCount} masks`);

    const boundsValid = annotationSummary.maskBounds.every(([left, top, right, bottom]) => (
        left >= 0 && top >= 0 && right <= IMAGE_WIDTH && bottom <= IMAGE_HEIGHT
    ));
    results.steps.annotations = annotationSummary;
    results.steps.maskBounds = {
        success: annotationSummary.maskBounds.length > 0 && boundsValid,
        bounds: annotationSummary.maskBounds,
        boundsValid,
        imageSize: { width: IMAGE_WIDTH, height: IMAGE_HEIGHT },
    };
    results.steps.persistenceCheck = {
        success: annotationSummary.maskCount > (results.steps.annotationsBefore?.maskCount || 0) &&
            annotationSummary.semiAutoCount > (results.steps.annotationsBefore?.semiAutoCount || 0),
        beforeMaskCount: results.steps.annotationsBefore?.maskCount || 0,
        afterMaskCount: annotationSummary.maskCount,
        beforeSemiAutoCount: results.steps.annotationsBefore?.semiAutoCount || 0,
        afterSemiAutoCount: annotationSummary.semiAutoCount,
    };

    await browser.close();

    // Overall determination
    results.overall = (
        bboxSuccess &&
        pointSuccess &&
        phase2Success &&
        results.steps.persistenceCheck.success &&
        results.steps.maskBounds.success
    ) ? 'PASS' : 'FAIL';

    fs.writeFileSync(path.join(runDir, 'non_square_summary.json'), JSON.stringify(results, null, 2));

    // Summary
    console.log('');
    console.log('=== Summary ===');
    console.log(`  Overall: ${results.overall}`);
    for (const [step, data] of Object.entries(results.steps)) {
        console.log(`  ${step}: ${JSON.stringify(data)}`);
    }
    console.log(`  Artifacts: ${runDir}`);

    // Fail report
    if (results.overall !== 'PASS') {
        const failReport = [
            `# Non-Square Image E2E Report`,
            ``,
            `## Image`,
            `${IMAGE_WIDTH}x${IMAGE_HEIGHT} (aspect ratio ${(IMAGE_WIDTH / IMAGE_HEIGHT).toFixed(3)})`,
            ``,
            `## Scale Analysis`,
            `scaleX = 1024/${IMAGE_WIDTH} = ${(1024 / IMAGE_WIDTH).toFixed(4)}`,
            `scaleY = 1024/${IMAGE_HEIGHT} = ${(1024 / IMAGE_HEIGHT).toFixed(4)}`,
            `Scale ratio (X/Y): ${((1024 / IMAGE_WIDTH) / (1024 / IMAGE_HEIGHT)).toFixed(4)}`,
            `Non-isotropic: coordinates are scaled differently in X and Y.`,
            ``,
            `## Results`,
            ...Object.entries(results.steps).map(([k, v]) => `- **${k}**: ${JSON.stringify(v)}`),
            ``,
            `## Key Finding`,
            `The getModelScale function in index.tsx scales independently: scaleX=1024/w, scaleY=1024/h.`,
            `This maps the original image to a 1024x1024 space for the SAM2 encoder.`,
            `The ONNX decoder should output a mask in 1024x1024 space, which then needs to be`,
            `mapped back. If the inverse mapping is incorrect, masks will appear stretched.`,
            ``,
            `## Reference`,
            `If coordinates are broken: check hashJoe commit 2571f82e1 "scale point coordinates to square target size"`,
            ``,
            `## Artifacts`,
            `Run dir: ${runDir}`,
        ].join('\n');
        fs.writeFileSync(path.join(runDir, 'expected_fail_non_square.md'), failReport);
    }

    process.exit(results.overall === 'PASS' ? 0 : 1);
}

main().catch(err => {
    console.error(`Fatal error: ${err.message}`);
    process.exit(1);
});
