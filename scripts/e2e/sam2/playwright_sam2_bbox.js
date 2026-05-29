#!/usr/bin/env node
/**
 * SAM2 BBox baseline E2E test for CVAT.
 *
 * Two-phase test:
 *   Phase 1: API-based lambda invoke (POST /api/lambda/functions/<id>)
 *            Tests Nuclio invoke path with bbox payload.
 *   Phase 2: Playwright-based job page load with network interception.
 *            Opens job page, captures lambda-related network traffic.
 *
 * The SAM2 architecture is encoder-only on the Nuclio side:
 *   - Backend sends image to Nuclio -> gets embeddings (high_res_feats_0/1, image_embed)
 *   - Frontend ONNX decoder generates mask from embeddings + points/bbox
 *   - So the API response contains embeddings, NOT a mask directly.
 *
 * Usage:
 *   NODE_PATH=~/temp/playwright-cli/node_modules node scripts/e2e/sam2/playwright_sam2_bbox.js
 *
 * Env (.env):
 *   CVAT_E2E_USER, CVAT_E2E_PASSWORD, CVAT_E2E_HOST
 */
const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const {
    loadEnv,
    makeRunDir,
    requireAuthState,
    getAnnotationSummary,
    clearJobAnnotations,
    finishAndSave,
    openAiTools,
    selectSam2Interactor,
    setStartWithBBox,
    clickInteract,
} = require('./e2e_utils');

// --- helpers ---

function httpRequest(url, opts, body) {
    return new Promise((resolve, reject) => {
        const mod = url.startsWith('https') ? https : http;
        const req = mod.request(url, opts, (res) => {
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                const raw = Buffer.concat(chunks).toString();
                resolve({
                    status: res.statusCode,
                    headers: res.headers,
                    body: raw,
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
        if (parts.length >= 2) {
            cookies[parts[0].trim()] = parts.slice(1).join('=').trim();
        }
    }
    return cookies;
}

// --- main ---

async function main() {
    const repoRoot = path.resolve(__dirname, '..', '..', '..');
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
    const runDir = makeRunDir('run_bbox');

    const TASK_ID = 181;
    const JOB_ID = 180;
    const FUNC_ID = 'ort-facebookresearch-sam2-hiera-base-plus';
    const JOB_URL = `${host}/tasks/${TASK_ID}/jobs/${JOB_ID}`;
    const LAMBDA_URL = `${host}/api/lambda/functions/${FUNC_ID}`;

    console.log(`=== SAM2 BBox Baseline E2E ===`);
    console.log(`Run dir: ${runDir}`);
    console.log(`Task: ${TASK_ID}, Job: ${JOB_ID}`);
    console.log(`Lambda URL: ${LAMBDA_URL}`);
    console.log('');

    // ========================================
    // Phase 1: API-based lambda invoke
    // ========================================
    console.log('--- Phase 1: API-based lambda invoke ---');

    // Step 1: Login via API
    console.log('Step 1: API login...');
    const loginUrl = `${host}/api/auth/login`;
    const loginBody = JSON.stringify({ username, password });
    const loginResp = await httpRequest(loginUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    }, loginBody);

    if (loginResp.status !== 200) {
        console.error(`Login failed: HTTP ${loginResp.status}`);
        fs.writeFileSync(path.join(runDir, 'login_error.json'), loginResp.body);
        process.exit(1);
    }

    // Parse cookies
    const allCookies = {};
    const loginCookies = parseCookies(loginResp.headers['set-cookie']);
    Object.assign(allCookies, loginCookies);
    console.log(`  Cookies received: ${Object.keys(allCookies).join(', ')}`);

    // Build cookie string
    const cookieStr = Object.entries(allCookies).map(([k, v]) => `${k}=${v}`).join('; ');
    const csrfToken = allCookies.csrftoken || '';

    // Step 2: Invoke lambda with BBox
    console.log('Step 2: Lambda invoke with BBox...');
    // BBox: top-left (100, 100) to bottom-right (412, 412) on 512x512 image
    // pos_points and neg_points are mandatory for interactors
    const invokePayload = {
        task: TASK_ID,
        job: JOB_ID,
        frame: 0,
        pos_points: [],
        neg_points: [],
        obj_bbox: [[100, 100], [412, 412]],
    };

    const invokeBody = JSON.stringify(invokePayload);
    fs.writeFileSync(path.join(runDir, 'lambda_request.json'), JSON.stringify(invokePayload, null, 2));

    let invokeResp;
    try {
        invokeResp = await httpRequest(LAMBDA_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cookie': cookieStr,
                'X-CSRFToken': csrfToken,
            },
            timeout: 120000,
        }, invokeBody);
    } catch (err) {
        console.error(`  Lambda invoke network error: ${err.message}`);
        fs.writeFileSync(path.join(runDir, 'lambda_network_error.txt'), err.message);
        invokeResp = { status: 0, body: err.message, headers: {} };
    }

    console.log(`  Lambda response: HTTP ${invokeResp.status}`);
    fs.writeFileSync(path.join(runDir, 'lambda_response_status.txt'), `${invokeResp.status}`);
    fs.writeFileSync(path.join(runDir, 'lambda_response_body.json'), invokeResp.body);

    let phase1Success = false;
    if (invokeResp.status === 200) {
        console.log('  Lambda invoke SUCCESS (HTTP 200)');
        try {
            const respData = JSON.parse(invokeResp.body);
            const hasEmbeddings = respData.image_embed && respData.high_res_feats_0 && respData.high_res_feats_1;
            if (hasEmbeddings) {
                console.log('  Embeddings present in response:');
                console.log(`    image_embed length: ${respData.image_embed.length}`);
                console.log(`    high_res_feats_0 length: ${respData.high_res_feats_0.length}`);
                console.log(`    high_res_feats_1 length: ${respData.high_res_feats_1.length}`);
                phase1Success = true;
            } else {
                console.log('  WARNING: Response does not contain expected embeddings');
                console.log(`  Response keys: ${Object.keys(respData).join(', ')}`);
            }
        } catch (parseErr) {
            console.log(`  WARNING: Response body is not valid JSON: ${parseErr.message}`);
        }
    } else {
        console.error(`  Lambda invoke FAILED (HTTP ${invokeResp.status})`);
        // Try to extract error details
        try {
            const errData = JSON.parse(invokeResp.body);
            console.error(`  Error: ${JSON.stringify(errData).slice(0, 500)}`);
        } catch (_) {
            console.error(`  Body (first 500 chars): ${invokeResp.body.slice(0, 500)}`);
        }
    }

    // ========================================
    // Phase 2: Playwright job page with network capture
    // ========================================
    console.log('');
    console.log('--- Phase 2: Playwright job page + network capture ---');

    const { chromium } = require('playwright');
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        storageState: authStatePath,
        viewport: { width: 1920, height: 1080 },
    });
    const page = await context.newPage();

    // Collect console
    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({
            type: msg.type(),
            text: msg.text(),
        });
    });

    // Collect network - focus on lambda/api calls
    const networkLogs = [];
    const lambdaResponses = [];
    page.on('response', async (response) => {
        const entry = {
            url: response.url(),
            status: response.status(),
            method: response.request().method(),
        };
        networkLogs.push(entry);

        // Capture lambda and annotation related responses
        if (response.url().includes('/api/lambda/') ||
            response.url().includes('/api/jobs/') && response.url().includes('/annotations')) {
            try {
                const body = await response.text();
                lambdaResponses.push({
                    ...entry,
                    body: body.slice(0, 10000),
                });
            } catch (_) {
                // response might not be available
            }
        }
    });

    let phase2Success = false;
    let uiSummary = null;
    try {
        const clearResult = await clearJobAnnotations(page, host, JOB_ID, shouldClearAnnotations);
        console.log(`  Clear annotations: ${JSON.stringify(clearResult)}`);

        // Navigate to job page
        console.log(`  Opening job: ${JOB_URL}`);
        await page.goto(JOB_URL, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);

        const currentUrl = page.url();
        console.log(`  Current URL: ${currentUrl}`);

        // Take screenshot of initial state
        await page.screenshot({ path: path.join(runDir, 'job_initial.png'), fullPage: true });
        console.log('  Screenshot: job_initial.png');

        // Check canvas exists
        const canvasCount = await page.$$eval('canvas', els => els.length);
        console.log(`  Canvas elements: ${canvasCount}`);

        if (canvasCount > 0) {
            phase2Success = true;
            console.log('  Job page loaded successfully');
        } else {
            console.log('  WARNING: No canvas elements found');
        }

        const annotationsBefore = await getAnnotationSummary(page, host, JOB_ID);
        console.log(`  Annotations before: masks=${annotationsBefore.maskCount}, semi-auto=${annotationsBefore.semiAutoCount}`);

        console.log('  Opening AI Tools and selecting SAM2 bbox mode...');
        await openAiTools(page);
        const selectionResult = await selectSam2Interactor(page);
        console.log(`  SAM2 selection result: ${JSON.stringify(selectionResult)}`);
        const bboxSwitch = await setStartWithBBox(page, true);
        console.log(`  BBox switch result: ${JSON.stringify(bboxSwitch)}`);
        await openAiTools(page);
        await clickInteract(page);
        await page.screenshot({ path: path.join(runDir, 'bbox_interact_mode.png'), fullPage: true });

        const drawTarget = await page.$('#cvat_canvas_background') ||
            await page.$('.cvat-canvas-container svg') ||
            await page.$('.cvat-canvas-container');
        if (!drawTarget) throw new Error('Canvas draw target not found');
        const bbox = await drawTarget.boundingBox();
        const elementInfo = await page.evaluate(([x, y]) => {
            const el = document.elementFromPoint(x, y);
            return {
                tagName: el?.tagName || null,
                id: el?.id || null,
                className: typeof el?.className === 'string' ? el.className : String(el?.className || ''),
            };
        }, [bbox.x + bbox.width / 2, bbox.y + bbox.height / 2]);
        fs.writeFileSync(path.join(runDir, 'bbox_draw_target.json'), JSON.stringify({ bbox, elementInfo }, null, 2));

        const startX = bbox.x + bbox.width * 0.35;
        const startY = bbox.y + bbox.height * 0.30;
        const endX = bbox.x + bbox.width * 0.65;
        const endY = bbox.y + bbox.height * 0.70;
        console.log(`  Dragging bbox: (${startX.toFixed(0)},${startY.toFixed(0)}) -> (${endX.toFixed(0)},${endY.toFixed(0)})`);
        const lambdaCountBeforeBBox = lambdaResponses.length;
        await page.mouse.move(startX, startY);
        await page.mouse.down();
        await page.mouse.move(endX, endY, { steps: 24 });
        await page.mouse.up();

        const stopResult = await page.evaluate(() => {
            const rectNode = document.querySelector('.cvat_interaction_rectangle');
            const rect = rectNode?.instance;
            if (!rect) {
                return { attempted: false, reason: 'active interaction rectangle not found' };
            }

            const result = {
                attempted: true,
                x: typeof rect.x === 'function' ? rect.x() : null,
                y: typeof rect.y === 'function' ? rect.y() : null,
                width: typeof rect.width === 'function' ? rect.width() : null,
                height: typeof rect.height === 'function' ? rect.height() : null,
                drawStopFired: false,
                errors: [],
            };

            try {
                rect.fire('drawstop');
                result.drawStopFired = true;
            } catch (error) {
                result.errors.push(error instanceof Error ? error.message : String(error));
            }

            return result;
        });
        console.log(`  Draw stop result: ${JSON.stringify(stopResult)}`);

        let bboxLambdaReceived = false;
        for (let i = 0; i < 30; i++) {
            await page.waitForTimeout(1000);
            if (lambdaResponses.length > lambdaCountBeforeBBox) {
                bboxLambdaReceived = true;
                break;
            }
        }
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(runDir, 'bbox_after_drag.png'), fullPage: true });

        const promptState = await page.evaluate(() => {
            const rects = Array.from(document.querySelectorAll('svg rect'))
                .filter((rect) => rect.getAttribute('stroke') || rect.classList.length);
            const interactionPoints = document.querySelectorAll('.cvat_interaction_point');
            return {
                svgRectCount: rects.length,
                interactionPointCount: interactionPoints.length,
                interactionRectangleCount: document.querySelectorAll('.cvat_interaction_rectangle').length,
            };
        });
        console.log(`  Prompt state: ${JSON.stringify(promptState)}`);

        const finishResult = await finishAndSave(page, runDir, host, JOB_ID, 'bbox');
        const finalAnnotations = await getAnnotationSummary(page, host, JOB_ID);
        phase2Success = (
            finalAnnotations.maskCount > annotationsBefore.maskCount &&
            finalAnnotations.semiAutoCount > annotationsBefore.semiAutoCount
        );
        uiSummary = {
            clearAnnotations: clearResult,
            drawStop: stopResult,
            bboxLambdaReceived,
            promptState,
            annotationsBefore,
            finishAndSave: finishResult,
            finalAnnotations,
            persistedMask: phase2Success,
        };
        console.log(`  Final annotations: masks=${finalAnnotations.maskCount}, semi-auto=${finalAnnotations.semiAutoCount}`);

    } catch (err) {
        console.error(`  Playwright error: ${err.message}`);
        await page.screenshot({ path: path.join(runDir, 'job_error.png'), fullPage: true }).catch(() => {});
    }

    // Save all artifacts
    fs.writeFileSync(path.join(runDir, 'console_logs.json'), JSON.stringify(consoleLogs, null, 2));
    fs.writeFileSync(path.join(runDir, 'network_logs.json'), JSON.stringify(networkLogs, null, 2));
    fs.writeFileSync(path.join(runDir, 'lambda_responses.json'), JSON.stringify(lambdaResponses, null, 2));

    // ========================================
    // Phase 3: Check annotations API
    // ========================================
    console.log('');
    console.log('--- Phase 3: Check annotations API ---');

    const annotationsUrl = `${host}/api/jobs/${JOB_ID}/annotations`;
    const annotResp = await httpRequest(annotationsUrl, {
        method: 'GET',
        headers: {
            'Cookie': cookieStr,
            'X-CSRFToken': csrfToken,
        },
    });

    console.log(`  Annotations API: HTTP ${annotResp.status}`);
    fs.writeFileSync(path.join(runDir, 'annotations_response.json'), annotResp.body);

    let hasMaskAnnotation = false;
    if (annotResp.status === 200) {
        try {
            const annData = JSON.parse(annotResp.body);
            const shapes = annData.shapes || [];
            const masks = shapes.filter(s => s.type === 'mask');
            const semiAuto = shapes.filter(s => s.source === 'semi-auto');
            console.log(`  Total shapes: ${shapes.length}`);
            console.log(`  Mask shapes: ${masks.length}`);
            console.log(`  Semi-auto shapes: ${semiAuto.length}`);
            if (masks.length > 0) {
                hasMaskAnnotation = true;
                console.log('  Mask annotation FOUND');
            } else {
                console.log('  No mask annotations (expected - API invoke only returns embeddings, not persisted masks)');
            }
        } catch (parseErr) {
            console.log(`  Could not parse annotations: ${parseErr.message}`);
        }
    }
    await browser.close();

    // ========================================
    // Summary
    // ========================================
    console.log('');
    console.log('=== Summary ===');
    const summary = {
        timestamp: new Date().toISOString(),
        runDir,
        taskId: TASK_ID,
        jobId: JOB_ID,
        functionId: FUNC_ID,
        phase1_api_invoke: {
            status: invokeResp.status,
            success: phase1Success,
            description: phase1Success
                ? 'Lambda invoke returned 200 with embeddings'
                : `Lambda invoke returned HTTP ${invokeResp.status}`,
        },
        phase2_playwright: {
            success: phase2Success,
            description: phase2Success
                ? 'BBox UI interaction persisted a semi-auto mask'
                : 'BBox UI interaction did not persist a semi-auto mask',
            details: uiSummary,
        },
        phase3_annotations: {
            status: annotResp.status,
            hasMaskAnnotation,
            description: hasMaskAnnotation
                ? 'Mask annotation found in job'
                : 'No mask annotation (expected for API-only invoke)',
        },
        overall: phase1Success && phase2Success ? 'PASS' : 'FAIL',
    };

    fs.writeFileSync(path.join(runDir, 'bbox_summary.json'), JSON.stringify(summary, null, 2));

    console.log(`  Phase 1 (API invoke): ${summary.phase1_api_invoke.success ? 'PASS' : 'FAIL'} - ${summary.phase1_api_invoke.description}`);
    console.log(`  Phase 2 (Playwright): ${summary.phase2_playwright.success ? 'PASS' : 'FAIL'} - ${summary.phase2_playwright.description}`);
    console.log(`  Phase 3 (Annotations): ${summary.phase3_annotations.description}`);
    console.log(`  Overall: ${summary.overall}`);
    console.log(`  Artifacts: ${runDir}`);

    process.exit(summary.overall === 'PASS' ? 0 : 1);
}

main().catch(err => {
    console.error(`Fatal error: ${err.message}`);
    process.exit(1);
});
