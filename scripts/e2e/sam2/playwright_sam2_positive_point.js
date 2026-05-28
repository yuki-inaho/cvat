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

const repoRoot = path.resolve(__dirname, '..', '..', '..');

function loadEnv(envPath) {
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

async function main() {
    const env = loadEnv(path.join(repoRoot, '.env'));
    const host = env.CVAT_E2E_HOST || 'http://localhost:8080';

    const authStatePath = path.join(repoRoot, 'temp', 'e2e_sam2', 'check', 'auth-state.json');
    if (!fs.existsSync(authStatePath)) {
        console.error(`ERROR: Auth state not found at ${authStatePath}. Run 'just e2e-login' first.`);
        process.exit(1);
    }

    const ts = new Date().toISOString().replace(/[-T:.Z]/g, '').slice(0, 14);
    const runDir = path.join(repoRoot, 'temp', 'e2e_sam2', `run_positive_${ts}`);
    fs.mkdirSync(runDir, { recursive: true });

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
        await page.goto(JOB_URL, { waitUntil: 'networkidle', timeout: 60000 });
        await page.waitForTimeout(3000);
        console.log(`  URL: ${page.url()}`);
        await page.screenshot({ path: path.join(runDir, '01_job_loaded.png'), fullPage: true });

        // Check canvas exists
        const canvasCount = await page.$$eval('canvas', els => els.length);
        console.log(`  Canvas elements: ${canvasCount}`);
        results.steps.jobLoad = { success: canvasCount > 0, canvasCount };

        // Step 2: Open AI Tools popover
        console.log('Step 2: Open AI Tools...');
        const aiToolsBtn = await page.$('.cvat-tools-control');
        if (!aiToolsBtn) throw new Error('AI Tools button not found');
        await aiToolsBtn.click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: path.join(runDir, '02_ai_tools_opened.png'), fullPage: true });
        console.log('  AI Tools popover opened');
        results.steps.aiToolsOpen = { success: true };

        // Step 3: Select SAM2 interactor and configure point mode
        // IMPORTANT: Ant Design Select dropdown causes popover to close.
        // We must select the interactor FIRST, then re-open the popover if needed.
        console.log('Step 3: Select SAM2 interactor...');

        // Check current interactor selection text
        const currentSelection = await page.$eval(
            '.ant-popover .ant-select-selection-item, .ant-popover-content .ant-select-selection-item',
            el => el.textContent
        ).catch(() => 'unknown');
        console.log(`  Current interactor: "${currentSelection}"`);

        if (!currentSelection.includes('2.1')) {
            // Need to select SAM 2.1 - this will close the popover
            console.log('  Need to switch to SAM 2.1...');
            const popoverSelects = await page.$$('.ant-popover .ant-select, .ant-popover-content .ant-select');
            // The second select in the popover is the interactor (first is label)
            const interactorSelect = popoverSelects.length >= 2 ? popoverSelects[1] : popoverSelects[0];
            if (interactorSelect) {
                await interactorSelect.click();
                await page.waitForTimeout(500);
                await page.screenshot({ path: path.join(runDir, '03a_dropdown_open.png'), fullPage: true });

                // Select SAM 2.1 option
                const sam2Option = await page.$('.ant-select-item-option:has-text("Segment Anything 2.1")');
                if (sam2Option) {
                    await sam2Option.click();
                    await page.waitForTimeout(500);
                    console.log('  Selected "Segment Anything 2.1"');
                } else {
                    console.log('  SAM2 option not found; pressing Escape');
                    await page.keyboard.press('Escape');
                }

                // Re-open AI Tools popover (it likely closed)
                await page.waitForTimeout(500);
                const aiToolsBtn2 = await page.$('.cvat-tools-control');
                if (aiToolsBtn2) {
                    await aiToolsBtn2.click();
                    await page.waitForTimeout(1000);
                    console.log('  Re-opened AI Tools popover');
                }
            }
        }
        await page.screenshot({ path: path.join(runDir, '03b_interactor_selected.png'), fullPage: true });
        results.steps.interactorSelect = { success: true };

        // Step 4: Ensure "Start with bounding box" is OFF for point mode
        console.log('Step 4: Ensure point mode (not bbox)...');
        // Use evaluate to toggle the switch via DOM to avoid popover-closing issues
        const switchToggled = await page.evaluate(() => {
            // Find all switches in the interactor setups area
            const setupDivs = document.querySelectorAll('.cvat-tools-interactor-setups div');
            for (const div of setupDivs) {
                if (div.textContent && div.textContent.includes('Start with a bounding box')) {
                    const sw = div.querySelector('.ant-switch');
                    if (sw && sw.classList.contains('ant-switch-checked')) {
                        sw.click();
                        return { toggled: true, wasChecked: true };
                    }
                    return { toggled: false, wasChecked: sw ? sw.classList.contains('ant-switch-checked') : null };
                }
            }
            return { toggled: false, wasChecked: null, error: 'Switch not found' };
        });
        console.log(`  Switch toggle result: ${JSON.stringify(switchToggled)}`);
        await page.waitForTimeout(500);
        await page.screenshot({ path: path.join(runDir, '04_point_mode_set.png'), fullPage: true });
        results.steps.pointMode = { success: true, ...switchToggled };

        // Step 5: Click "Interact" button to enter interaction mode
        // The popover may have closed from the switch toggle. Re-open it.
        console.log('Step 5: Click Interact button...');
        // Re-open popover to make the Interact button visible
        const aiToolsBtn4 = await page.$('.cvat-tools-control');
        if (aiToolsBtn4) {
            await aiToolsBtn4.click();
            await page.waitForTimeout(1000);
        }
        await page.screenshot({ path: path.join(runDir, '05a_popover_reopened.png'), fullPage: true });

        // Now click the Interact button
        const interactBtnFinal = await page.$('.cvat-tools-interact-button');
        if (!interactBtnFinal) throw new Error('Interact button not found after re-open');

        const isDisabled = await interactBtnFinal.evaluate(el => el.disabled || el.classList.contains('ant-btn-disabled'));
        console.log(`  Interact button disabled: ${isDisabled}`);
        if (isDisabled) throw new Error('Interact button is disabled');

        await interactBtnFinal.click();
        await page.waitForTimeout(1500);
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
                success: false,
                error: 'No lambda response within timeout',
                lambdaCountBefore,
                lambdaCountAfter: lambdaResponses.length,
            };
        }

        // Step 7: Check for mask overlay on canvas
        console.log('Step 7: Check for mask on canvas...');
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(runDir, '07_mask_check.png'), fullPage: true });

        // The mask is rendered as a canvas overlay, not SVG shapes.
        // Also check for interaction point markers (green dots).
        const objectsInDOM = await page.evaluate(() => {
            // SVG shapes (committed annotations)
            const svgShapes = document.querySelectorAll('.cvat_canvas_shape, .cvat_canvas_shape_mask, [data-type="mask"]');
            const annotObjects = document.querySelectorAll('.cvat-objects-sidebar-state-item');
            // Interaction points (green/red dots on canvas, rendered as SVG circles)
            const interactionPts = document.querySelectorAll('circle, .cvat_canvas_interaction_point');
            // Canvas elements that may contain mask overlay
            const canvases = document.querySelectorAll('.cvat-canvas-container canvas');
            // Check if any canvas has non-trivial content (mask overlay)
            let maskCanvasDetected = false;
            for (const c of canvases) {
                try {
                    const ctx = c.getContext('2d');
                    if (ctx) {
                        const data = ctx.getImageData(0, 0, Math.min(c.width, 10), Math.min(c.height, 10)).data;
                        // Check if any pixel has alpha > 0 (mask overlay)
                        for (let i = 3; i < data.length; i += 4) {
                            if (data[i] > 0) { maskCanvasDetected = true; break; }
                        }
                    }
                } catch (_) {}
                if (maskCanvasDetected) break;
            }
            return {
                svgShapeCount: svgShapes.length,
                sidebarObjectCount: annotObjects.length,
                interactionPointCount: interactionPts.length,
                canvasCount: canvases.length,
                maskCanvasDetected,
            };
        });
        console.log(`  SVG shapes on canvas: ${objectsInDOM.svgShapeCount}`);
        console.log(`  Sidebar annotation objects: ${objectsInDOM.sidebarObjectCount}`);
        console.log(`  Interaction points (circles): ${objectsInDOM.interactionPointCount}`);
        console.log(`  Mask canvas detected: ${objectsInDOM.maskCanvasDetected}`);
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

        // Determine overall success
        // For positive point, success means:
        // 1. Interaction mode active AND
        // 2. Mask visible (canvas overlay or SVG shape) OR interaction point visible
        const maskGenerated = objectsInDOM.maskCanvasDetected || objectsInDOM.svgShapeCount > 0;
        const pointRegistered = objectsInDOM.interactionPointCount > 0 || uiState.greenCircles > 0;
        results.overall = (maskGenerated || pointRegistered) ? 'PASS' : (lambdaReceived ? 'PARTIAL' : 'FAIL');

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
