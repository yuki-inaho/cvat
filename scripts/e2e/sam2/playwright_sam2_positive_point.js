#!/usr/bin/env node
/**
 * SAM2 Positive Point fail-first E2E test for CVAT.
 *
 * Tests: select SAM2 interactor, switch to point mode, left-click on
 * the red rectangle area to produce a positive point → mask.
 *
 * Architecture note:
 *   - CVAT UI opens AI Tools popover, user picks interactor & clicks Interact.
 *   - In point mode (draw_points), left-click = positive point, right-click = negative.
 *   - Each click fires canvas `canvas.interact` event with shapes.
 *   - tools-control.tsx `onInteraction` collects pos/neg points, calls lambda.
 *   - Lambda returns embeddings, plugin's ONNX decoder produces mask.
 *   - Mask appears on canvas as overlay.
 *
 * Trace: SG-04, TR-06, TR-09
 *
 * Usage:
 *   NODE_PATH=~/temp/playwright-cli/node_modules node scripts/e2e/sam2/playwright_sam2_positive_point.js
 */
const fs = require('fs');
const path = require('path');
const {
    repoRoot,
    loadEnv,
    makeRunDir,
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
} = require('./e2e_utils');

async function main() {
    const env = loadEnv(path.join(repoRoot, '.env'));
    const host = env.CVAT_E2E_HOST || 'http://localhost:8080';

    const authStatePath = requireAuthState();
    const runDir = makeRunDir('run_positive');

    const TASK_ID = 181;
    const JOB_ID = 180;
    const JOB_URL = `${host}/tasks/${TASK_ID}/jobs/${JOB_ID}`;

    console.log('=== SAM2 Positive Point E2E (fail-first) ===');
    console.log(`Run dir: ${runDir}`);
    console.log(`Job URL: ${JOB_URL}`);
    console.log('');

    const { chromium } = require('playwright');
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        storageState: authStatePath,
        viewport: { width: 1920, height: 1080 },
    });
    const page = await context.newPage();
    const shouldClearAnnotations = env.CVAT_E2E_CLEAR_ANNOTATIONS !== '0';

    // Collect console logs
    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({ type: msg.type(), text: msg.text() });
    });

    // Collect network requests/responses
    const networkLogs = [];
    const lambdaResponses = [];
    page.on('response', async (response) => {
        const entry = {
            url: response.url(),
            status: response.status(),
            method: response.request().method(),
        };
        networkLogs.push(entry);

        if (response.url().includes('/api/lambda/')) {
            try {
                const body = await response.text();
                lambdaResponses.push({ ...entry, bodyLength: body.length, bodySnippet: body.slice(0, 2000) });
            } catch (_) {}
        }
    });

    const results = {
        timestamp: new Date().toISOString(),
        runDir,
        steps: {},
        overall: 'UNKNOWN',
    };

    try {
        // Step 1: Navigate to job page
        console.log('Step 1: Navigate to job page...');
        const clearResult = await clearJobAnnotations(page, host, JOB_ID, shouldClearAnnotations);
        console.log(`  Clear annotations: ${JSON.stringify(clearResult)}`);
        results.steps.clearAnnotations = clearResult;
        await page.goto(JOB_URL, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);
        console.log(`  URL: ${page.url()}`);
        await page.screenshot({ path: path.join(runDir, '01_job_loaded.png'), fullPage: true });

        // Check canvas exists
        const canvasCount = await page.$$eval('canvas', els => els.length);
        console.log(`  Canvas elements: ${canvasCount}`);
        results.steps.jobLoad = { success: canvasCount > 0, canvasCount };
        const annotationsBefore = await getAnnotationSummary(page, host, JOB_ID);
        results.steps.annotationsBefore = annotationsBefore;

        // Step 2: Open AI Tools popover
        console.log('Step 2: Open AI Tools...');
        await openAiTools(page);
        await page.screenshot({ path: path.join(runDir, '02_ai_tools_opened.png'), fullPage: true });
        console.log('  AI Tools popover opened');
        results.steps.aiToolsOpen = { success: true };

        // Step 3: Select SAM2 interactor and configure point mode
        console.log('Step 3: Select SAM2 interactor...');
        const selectionResult = await selectSam2Interactor(page);
        console.log(`  SAM2 selection result: ${JSON.stringify(selectionResult)}`);
        await page.screenshot({ path: path.join(runDir, '03b_interactor_selected.png'), fullPage: true });
        results.steps.interactorSelect = { success: true };

        // Step 4: Ensure "Start with bounding box" is OFF for point mode
        console.log('Step 4: Ensure point mode (not bbox)...');
        const switchToggled = await setStartWithBBox(page, false);
        console.log(`  Switch toggle result: ${JSON.stringify(switchToggled)}`);
        await page.waitForTimeout(500);
        await page.screenshot({ path: path.join(runDir, '04_point_mode_set.png'), fullPage: true });
        results.steps.pointMode = { success: true, ...switchToggled };

        // Step 5: Click "Interact" button to enter interaction mode
        console.log('Step 5: Click Interact button...');
        await openAiTools(page);
        await page.screenshot({ path: path.join(runDir, '05a_popover_reopened.png'), fullPage: true });
        await clickInteract(page);
        await page.screenshot({ path: path.join(runDir, '05_interact_mode.png'), fullPage: true });
        console.log('  Entered interaction mode');
        results.steps.interact = { success: true };

        // Step 6: Left-click on canvas (positive point)
        // The 512x512 image has a red rectangle roughly at center.
        // We need to click on the canvas element at the right coordinates.
        console.log('Step 6: Left-click on canvas (positive point)...');

        // Find the canvas area
        const canvasWrapper = await page.$('.cvat-canvas-container');
        if (!canvasWrapper) {
            console.error('  ERROR: Canvas container not found');
            throw new Error('Canvas container not found');
        }

        const canvasBBox = await canvasWrapper.boundingBox();
        console.log(`  Canvas container bounding box: ${JSON.stringify(canvasBBox)}`);

        // Click at the center of the canvas (which should be near the center of the image)
        // The red rectangle in the 512x512 test image is roughly at center
        const clickX = canvasBBox.x + canvasBBox.width / 2;
        const clickY = canvasBBox.y + canvasBBox.height / 2;
        console.log(`  Clicking at canvas position: (${clickX.toFixed(0)}, ${clickY.toFixed(0)})`);

        // Clear any previous lambda responses for comparison
        const lambdaCountBefore = lambdaResponses.length;
        const canvasBefore = await getCanvasHashes(page);
        const pointsBefore = await getInteractionPointStats(page);

        await page.mouse.click(clickX, clickY, { button: 'left' });
        console.log('  Left click sent');

        // Wait for network response (lambda invoke + ONNX decode)
        console.log('  Waiting for lambda response...');
        let lambdaReceived = false;
        for (let i = 0; i < 30; i++) {
            await page.waitForTimeout(1000);
            if (lambdaResponses.length > lambdaCountBefore) {
                lambdaReceived = true;
                break;
            }
        }

        await page.screenshot({ path: path.join(runDir, '06_after_click.png'), fullPage: true });
        const canvasAfterClick = await getCanvasHashes(page);
        const pointsAfterClick = await getInteractionPointStats(page);
        const canvasChangedAfterClick = changedCanvasCount(canvasBefore, canvasAfterClick);

        if (lambdaReceived) {
            const latestLambda = lambdaResponses[lambdaResponses.length - 1];
            console.log(`  Lambda response received: HTTP ${latestLambda.status}`);
            console.log(`  Lambda URL: ${latestLambda.url}`);
            console.log(`  Response body length: ${latestLambda.bodyLength}`);
            results.steps.positiveClick = {
                success: latestLambda.status === 200,
                lambdaStatus: latestLambda.status,
                lambdaUrl: latestLambda.url,
                bodyLength: latestLambda.bodyLength,
            };
        } else {
            console.log('  WARNING: No lambda response received within 30s');
            console.log('  This may mean:');
            console.log('    - The click did not register as a positive point');
            console.log('    - The embeddings were already cached (no server call needed)');
            console.log('    - The interaction mode was not active');
            results.steps.positiveClick = {
                success: pointsAfterClick.positive > pointsBefore.positive,
                note: 'No lambda response within timeout; prompt registration and saved annotation are authoritative when embeddings are cached.',
                lambdaCountBefore,
                lambdaCountAfter: lambdaResponses.length,
            };
        }
        results.steps.positivePrompt = {
            success: pointsAfterClick.positive > pointsBefore.positive,
            before: pointsBefore,
            after: pointsAfterClick,
        };
        results.steps.canvasChangeAfterPositive = {
            diagnosticOnly: true,
            changedCanvasCount: canvasChangedAfterClick,
            before: canvasBefore,
            after: canvasAfterClick,
        };

        // Step 7: Check for mask overlay on canvas
        console.log('Step 7: Check for mask on canvas...');
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(runDir, '07_mask_check.png'), fullPage: true });

        // The mask is rendered as a canvas overlay, not SVG shapes.
        // Also check for interaction point markers (green dots).
        const objectsInDOM = await page.evaluate(() => {
            const svgShapes = document.querySelectorAll('.cvat_canvas_shape, .cvat_canvas_shape_mask, [data-type="mask"]');
            const annotObjects = document.querySelectorAll('.cvat-objects-sidebar-state-item');
            const interactionPts = document.querySelectorAll('.cvat_interaction_point');
            const canvases = document.querySelectorAll('.cvat-canvas-container canvas');
            return {
                svgShapeCount: svgShapes.length,
                sidebarObjectCount: annotObjects.length,
                interactionPointCount: interactionPts.length,
                canvasCount: canvases.length,
            };
        });
        console.log(`  SVG shapes on canvas: ${objectsInDOM.svgShapeCount}`);
        console.log(`  Sidebar annotation objects: ${objectsInDOM.sidebarObjectCount}`);
        console.log(`  Interaction points (circles): ${objectsInDOM.interactionPointCount}`);
        results.steps.maskCheck = objectsInDOM;

        // Step 8: Check UI state
        console.log('Step 8: Check UI interaction state...');
        const uiState = await page.evaluate(() => {
            const activeControl = document.querySelector('.cvat-active-canvas-control');
            // More thorough interaction point search
            const allCircles = document.querySelectorAll('svg circle');
            const greenCircles = Array.from(allCircles).filter(c => {
                const fill = c.getAttribute('fill') || '';
                const stroke = c.getAttribute('stroke') || '';
                return fill.includes('green') || fill.includes('#0f0') || fill.includes('rgb(0') ||
                       stroke.includes('green') || stroke.includes('#0f0');
            });
            return {
                hasActiveControl: !!activeControl,
                activeControlClass: activeControl?.className || '',
                totalSvgCircles: allCircles.length,
                greenCircles: greenCircles.length,
            };
        });
        console.log(`  Active control: ${uiState.hasActiveControl} (${uiState.activeControlClass})`);
        console.log(`  Total SVG circles: ${uiState.totalSvgCircles}`);
        console.log(`  Green circles (positive points): ${uiState.greenCircles}`);
        results.steps.uiState = uiState;

        // Step 9: Finish the interaction and verify it becomes a persisted semi-auto mask.
        console.log('Step 9: Finish interaction and check annotations...');
        const finishResult = await finishAndSave(page, runDir, host, JOB_ID, 'positive');
        const finalAnnotations = await getAnnotationSummary(page, host, JOB_ID);
        results.steps.finishAndSave = finishResult;
        results.steps.finalAnnotations = finalAnnotations;

        const pointRegistered = pointsAfterClick.positive > pointsBefore.positive;
        const persistedMask = (
            finalAnnotations.maskCount > annotationsBefore.maskCount &&
            finalAnnotations.semiAutoCount > annotationsBefore.semiAutoCount
        );
        results.steps.persistenceCheck = {
            success: persistedMask,
            beforeMaskCount: annotationsBefore.maskCount,
            afterMaskCount: finalAnnotations.maskCount,
            beforeSemiAutoCount: annotationsBefore.semiAutoCount,
            afterSemiAutoCount: finalAnnotations.semiAutoCount,
        };
        results.overall = pointRegistered && persistedMask ? 'PASS' : 'FAIL';

    } catch (err) {
        console.error(`ERROR: ${err.message}`);
        await page.screenshot({ path: path.join(runDir, 'error.png'), fullPage: true }).catch(() => {});
        results.overall = 'FAIL';
        results.error = err.message;
    }

    // Save artifacts
    fs.writeFileSync(path.join(runDir, 'console_logs.json'), JSON.stringify(consoleLogs, null, 2));
    fs.writeFileSync(path.join(runDir, 'network_logs.json'), JSON.stringify(networkLogs, null, 2));
    fs.writeFileSync(path.join(runDir, 'lambda_responses.json'), JSON.stringify(lambdaResponses, null, 2));
    fs.writeFileSync(path.join(runDir, 'positive_point_summary.json'), JSON.stringify(results, null, 2));

    await browser.close();

    // Summary
    console.log('');
    console.log('=== Summary ===');
    console.log(`  Overall: ${results.overall}`);
    for (const [step, data] of Object.entries(results.steps)) {
        const status = data.success !== false ? 'OK' : 'FAIL';
        console.log(`  ${step}: ${status} ${JSON.stringify(data)}`);
    }
    console.log(`  Artifacts: ${runDir}`);

    // If fail-first, save expected failure details
    if (results.overall !== 'PASS') {
        const failReport = [
            `# Expected Fail: Positive Point E2E`,
            ``,
            `## Timestamp`,
            `${new Date().toISOString()}`,
            ``,
            `## Result`,
            `Overall: ${results.overall}`,
            ``,
            `## Steps`,
            ...Object.entries(results.steps).map(([k, v]) => `- **${k}**: ${JSON.stringify(v)}`),
            ``,
            `## Analysis`,
            results.error ? `Error: ${results.error}` : '',
            ``,
            `Lambda responses captured: ${lambdaResponses.length}`,
            lambdaResponses.length > 0 ? `Latest lambda: HTTP ${lambdaResponses[lambdaResponses.length - 1]?.status} ${lambdaResponses[lambdaResponses.length - 1]?.url}` : 'No lambda invocations captured.',
            ``,
            `## Possible Reasons`,
            `- If no lambda call: click may not have been registered as interaction point`,
            `- If lambda called but no mask: ONNX decoder may have failed`,
            `- If lambda cached: embeddings already in LRU cache, only ONNX decode needed (no network call)`,
            ``,
            `## Console Errors`,
            ...consoleLogs.filter(l => l.type === 'error').map(l => `- ${l.text}`).slice(0, 20),
            ``,
            `## Artifacts`,
            `Run dir: ${runDir}`,
        ].join('\n');
        fs.writeFileSync(path.join(runDir, 'expected_fail_positive_point.md'), failReport);
        console.log(`  Fail report: ${path.join(runDir, 'expected_fail_positive_point.md')}`);
    }

    process.exit(results.overall === 'PASS' ? 0 : 1);
}

main().catch(err => {
    console.error(`Fatal error: ${err.message}`);
    process.exit(1);
});
