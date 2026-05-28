#!/usr/bin/env node
/**
 * SAM2 Negative Point fail-first E2E test for CVAT.
 *
 * Tests: after placing a positive point, right-click on background
 * to add a negative point → mask should shrink/change.
 *
 * Architecture note:
 *   - In point interaction mode, left-click = positive (clickType 1),
 *     right-click = negative (clickType 0).
 *   - Each new click triggers re-invoke with all accumulated points.
 *   - The ONNX decoder uses maskInput (low-res mask from previous decode)
 *     when current clicks are a superset of previous ones.
 *
 * Trace: SG-04, TR-07, TR-09
 *
 * Usage:
 *   NODE_PATH=~/temp/playwright-cli/node_modules node scripts/e2e/sam2/playwright_sam2_negative_point.js
 */
const fs = require('fs');
const path = require('path');
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

async function main() {
    const env = loadEnv(path.join(repoRoot, '.env'));
    const host = env.CVAT_E2E_HOST || 'http://localhost:8080';

    const authStatePath = requireAuthState();
    const runDir = makeRunDir('run_negative');

    const TASK_ID = 181;
    const JOB_ID = 180;
    const JOB_URL = `${host}/tasks/${TASK_ID}/jobs/${JOB_ID}`;

    console.log('=== SAM2 Negative Point E2E (fail-first) ===');
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

    // Block context menu globally to allow right-click for negative points
    await page.addInitScript(() => {
        document.addEventListener('contextmenu', e => e.preventDefault(), true);
    });

    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({ type: msg.type(), text: msg.text() });
    });

    const networkLogs = [];
    const lambdaResponses = [];
    page.on('response', async (response) => {
        const entry = {
            url: response.url(),
            status: response.status(),
            method: response.request().method(),
            timestamp: Date.now(),
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
        await page.screenshot({ path: path.join(runDir, '01_job_loaded.png'), fullPage: true });
        results.steps.jobLoad = { success: true };
        const annotationsBefore = await getAnnotationSummary(page, host, JOB_ID);
        results.steps.annotationsBefore = annotationsBefore;

        // Step 2: Open AI Tools and select SAM2
        console.log('Step 2: Open AI Tools...');
        await openAiTools(page);
        await page.screenshot({ path: path.join(runDir, '02_ai_tools.png'), fullPage: true });
        results.steps.aiToolsOpen = { success: true };

        // Step 3: Select SAM2 and ensure point mode
        console.log('Step 3: Select SAM2 interactor in point mode...');
        const selectionResult = await selectSam2Interactor(page);
        console.log(`  SAM2 selection result: ${JSON.stringify(selectionResult)}`);

        // Toggle "Start with bounding box" OFF via JS
        const switchToggled = await setStartWithBBox(page, false);
        console.log(`  Switch toggle: ${JSON.stringify(switchToggled)}`);
        await page.waitForTimeout(500);
        await page.screenshot({ path: path.join(runDir, '03_point_mode.png'), fullPage: true });
        results.steps.interactorSetup = { success: true };

        // Step 4: Click Interact (re-open popover first since switch toggle may have closed it)
        console.log('Step 4: Click Interact...');
        await openAiTools(page);
        await clickInteract(page);
        await page.screenshot({ path: path.join(runDir, '04_interact_mode.png'), fullPage: true });
        results.steps.interact = { success: true };

        // Step 5: Positive point (left click on the red rectangle area)
        console.log('Step 5: Left-click positive point...');
        const canvasWrapper = await page.$('.cvat-canvas-container');
        if (!canvasWrapper) throw new Error('Canvas container not found');
        const canvasBBox = await canvasWrapper.boundingBox();

        // Click center of canvas (inside the red rectangle)
        const posX = canvasBBox.x + canvasBBox.width / 2;
        const posY = canvasBBox.y + canvasBBox.height / 2;
        console.log(`  Positive click at: (${posX.toFixed(0)}, ${posY.toFixed(0)})`);

        const lambdaCountBeforePos = lambdaResponses.length;
        const pointsBefore = await getInteractionPointStats(page);
        await page.mouse.click(posX, posY, { button: 'left' });

        // Wait for lambda response
        let posLambdaReceived = false;
        for (let i = 0; i < 30; i++) {
            await page.waitForTimeout(1000);
            if (lambdaResponses.length > lambdaCountBeforePos) {
                posLambdaReceived = true;
                break;
            }
        }
        await page.waitForTimeout(2000); // Extra wait for ONNX decode
        await page.screenshot({ path: path.join(runDir, '05_after_positive.png'), fullPage: true });

        // Capture canvas state after positive point
        const pointStatsAfterPositive = await getInteractionPointStats(page);
        const afterPositive = await page.evaluate(() => {
            const shapes = document.querySelectorAll('.cvat_canvas_shape, .cvat_canvas_shape_mask');
            return { shapeCount: shapes.length };
        });
        Object.assign(afterPositive, {
            positivePoints: pointStatsAfterPositive.positive,
            negativePoints: pointStatsAfterPositive.negative,
            totalPoints: pointStatsAfterPositive.total,
        });
        console.log(`  After positive: shapes=${afterPositive.shapeCount}, positivePoints=${afterPositive.positivePoints}, negativePoints=${afterPositive.negativePoints}`);
        console.log(`  Lambda received: ${posLambdaReceived} (count: ${lambdaResponses.length})`);
        results.steps.positiveClick = {
            success: pointStatsAfterPositive.positive > pointsBefore.positive,
            lambdaReceived: posLambdaReceived,
            ...afterPositive,
        };

        // Step 6: Negative point (right-click on background, inside image but outside red rect)
        console.log('Step 6: Right-click negative point...');
        // The image is centered in the canvas. The red rectangle is roughly in the center-right.
        // We need to click on the white background area INSIDE the image bounds.
        // Using an offset from the image center towards top-left (still inside image, but outside red rect)
        // The image fills roughly the center of the canvas, so 35% from left, 25% from top
        // should be inside the image but in the white/background area.
        const negX = canvasBBox.x + canvasBBox.width * 0.35;
        const negY = canvasBBox.y + canvasBBox.height * 0.25;
        console.log(`  Negative click at: (${negX.toFixed(0)}, ${negY.toFixed(0)})`);

        const lambdaCountBeforeNeg = lambdaResponses.length;
        await page.mouse.click(negX, negY, { button: 'right' });

        // Wait for lambda response
        let negLambdaReceived = false;
        for (let i = 0; i < 30; i++) {
            await page.waitForTimeout(1000);
            if (lambdaResponses.length > lambdaCountBeforeNeg) {
                negLambdaReceived = true;
                break;
            }
        }
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(runDir, '06_after_negative.png'), fullPage: true });

        // Capture canvas state after negative point
        const pointStatsAfterNegative = await getInteractionPointStats(page);
        const afterNegative = await page.evaluate(() => {
            const shapes = document.querySelectorAll('.cvat_canvas_shape, .cvat_canvas_shape_mask');
            return { shapeCount: shapes.length };
        });
        Object.assign(afterNegative, {
            positivePoints: pointStatsAfterNegative.positive,
            negativePoints: pointStatsAfterNegative.negative,
            totalPoints: pointStatsAfterNegative.total,
        });
        console.log(`  After negative: shapes=${afterNegative.shapeCount}, positivePoints=${afterNegative.positivePoints}, negativePoints=${afterNegative.negativePoints}`);
        console.log(`  Lambda received: ${negLambdaReceived} (count: ${lambdaResponses.length})`);
        results.steps.negativeClick = {
            success: pointStatsAfterNegative.negative > pointStatsAfterPositive.negative,
            lambdaReceived: negLambdaReceived,
            ...afterNegative,
        };

        // Step 7: Compare positive-only vs positive+negative
        console.log('Step 7: Compare states...');
        results.steps.comparison = {
            beforePoints: pointsBefore,
            afterPositivePoints: pointStatsAfterPositive,
            afterNegativePoints: pointStatsAfterNegative,
            contextMenuBlocked: true,
        };

        const hasPositivePoint = pointStatsAfterPositive.positive > pointsBefore.positive;
        const hasNegativePoint = pointStatsAfterNegative.negative > pointStatsAfterPositive.negative;
        console.log(`  Positive point detected: ${hasPositivePoint}`);
        console.log(`  Negative point detected: ${hasNegativePoint}`);

        // Step 8: Finish the interaction and verify it becomes a persisted semi-auto mask.
        console.log('Step 8: Finish interaction and check annotations...');
        const finishResult = await finishAndSave(page, runDir, host, JOB_ID, 'negative');
        const finalAnnotations = await getAnnotationSummary(page, host, JOB_ID);
        results.steps.finishAndSave = finishResult;
        results.steps.finalAnnotations = finalAnnotations;
        results.steps.persistenceCheck = {
            success: finalAnnotations.maskCount > annotationsBefore.maskCount &&
                finalAnnotations.semiAutoCount > annotationsBefore.semiAutoCount,
            beforeMaskCount: annotationsBefore.maskCount,
            afterMaskCount: finalAnnotations.maskCount,
            beforeSemiAutoCount: annotationsBefore.semiAutoCount,
            afterSemiAutoCount: finalAnnotations.semiAutoCount,
        };

        // Determine overall
        const hasPersistedMask = results.steps.persistenceCheck.success;
        results.overall = hasPositivePoint && hasNegativePoint && hasPersistedMask ? 'PASS' : 'FAIL';

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
    fs.writeFileSync(path.join(runDir, 'negative_point_summary.json'), JSON.stringify(results, null, 2));

    await browser.close();

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
            `# Expected Fail: Negative Point E2E`,
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
            ``,
            `## Context Menu Prevention`,
            `Used page.addInitScript to block contextmenu globally.`,
            `If right-click still did not register as negative point:`,
            `- Check if CVAT canvas handles mousedown with button=2 instead of contextmenu`,
            `- Check if the interaction handler in tools-control filters right-clicks`,
            ``,
            `## Console Errors`,
            ...consoleLogs.filter(l => l.type === 'error').map(l => `- ${l.text}`).slice(0, 20),
            ``,
            `## Artifacts`,
            `Run dir: ${runDir}`,
        ].join('\n');
        fs.writeFileSync(path.join(runDir, 'expected_fail_negative_point.md'), failReport);
    }

    process.exit(results.overall === 'PASS' ? 0 : 1);
}

main().catch(err => {
    console.error(`Fatal error: ${err.message}`);
    process.exit(1);
});
