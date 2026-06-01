#!/usr/bin/env node
/**
 * Playwright headless job page load check for CVAT.
 *
 * Opens a job URL using saved auth-state, captures:
 *   - Screenshot
 *   - Console logs
 *   - Network request/response log
 *   - Canvas element count
 *
 * Usage:
 *   NODE_PATH=~/temp/playwright-cli/node_modules node scripts/e2e/sam2/playwright_job_check.js [JOB_URL]
 *
 * Default JOB_URL: http://localhost:8080/tasks/181/jobs/180
 */
const fs = require('fs');
const path = require('path');

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
    const repoRoot = path.resolve(__dirname, '..', '..', '..');
    const env = loadEnv(path.join(repoRoot, '.env'));
    const host = env.CVAT_E2E_HOST || 'http://localhost:8080';

    const jobUrl = process.argv[2] || `${host}/tasks/181/jobs/180`;
    const authStatePath = path.join(repoRoot, 'temp', 'e2e_sam2', 'check', 'auth-state.json');

    if (!fs.existsSync(authStatePath)) {
        console.error(`ERROR: Auth state not found at ${authStatePath}. Run 'just e2e-login' first.`);
        process.exit(1);
    }

    const ts = new Date().toISOString().replace(/[-T:.Z]/g, '').slice(0, 14);
    const runDir = path.join(repoRoot, 'temp', 'e2e_sam2', `run_${ts}`);
    fs.mkdirSync(runDir, { recursive: true });

    const { chromium } = require('playwright');

    console.log(`Opening job: ${jobUrl}`);
    console.log(`Auth state: ${authStatePath}`);
    console.log(`Run dir: ${runDir}`);

    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        storageState: authStatePath,
        viewport: { width: 1920, height: 1080 },
    });
    const page = await context.newPage();

    // Collect console messages
    const consoleLogs = [];
    page.on('console', msg => {
        consoleLogs.push({
            type: msg.type(),
            text: msg.text(),
            location: msg.location(),
        });
    });

    // Collect network requests
    const networkLogs = [];
    page.on('response', response => {
        networkLogs.push({
            url: response.url(),
            status: response.status(),
            method: response.request().method(),
        });
    });

    let exitCode = 0;

    try {
        // Navigate to job page
        await page.goto(jobUrl, { waitUntil: 'networkidle', timeout: 60000 });
        console.log(`Page loaded. Current URL: ${page.url()}`);

        // Wait a bit for canvas rendering
        await page.waitForTimeout(3000);

        // Check for canvas elements
        const canvasCount = await page.$$eval('canvas', els => els.length);
        console.log(`Canvas elements found: ${canvasCount}`);

        // Take screenshot
        const screenshotPath = path.join(runDir, 'job_page.png');
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.log(`Screenshot saved: ${screenshotPath}`);

        // Check for critical console errors
        const criticalErrors = consoleLogs.filter(l =>
            l.type === 'error' &&
            !l.text.includes('favicon') &&
            !l.text.includes('DevTools')
        );

        if (criticalErrors.length > 0) {
            console.log(`WARNING: ${criticalErrors.length} console error(s) detected`);
            criticalErrors.forEach(e => console.log(`  [error] ${e.text.slice(0, 200)}`));
        } else {
            console.log('No critical console errors');
        }

        // Check network for failed requests (excluding expected 4xx)
        const failedNetwork = networkLogs.filter(l =>
            l.status >= 500 ||
            (l.status >= 400 && !l.url.includes('favicon') && !l.url.includes('.map'))
        );

        if (failedNetwork.length > 0) {
            console.log(`WARNING: ${failedNetwork.length} failed network request(s)`);
            failedNetwork.forEach(r => console.log(`  [${r.status}] ${r.method} ${r.url.slice(0, 150)}`));
        } else {
            console.log('All network requests succeeded');
        }

        // Summary
        const summary = {
            url: page.url(),
            jobUrl: jobUrl,
            canvasCount: canvasCount,
            consoleErrorCount: criticalErrors.length,
            failedNetworkCount: failedNetwork.length,
            totalNetworkRequests: networkLogs.length,
            timestamp: new Date().toISOString(),
        };

        if (canvasCount < 1) {
            console.error('FAILED: No canvas element found on job page');
            exitCode = 1;
        }

        // Save artifacts
        fs.writeFileSync(
            path.join(runDir, 'console_logs.json'),
            JSON.stringify(consoleLogs, null, 2)
        );
        fs.writeFileSync(
            path.join(runDir, 'network_logs.json'),
            JSON.stringify(networkLogs, null, 2)
        );
        fs.writeFileSync(
            path.join(runDir, 'job_check_summary.json'),
            JSON.stringify(summary, null, 2)
        );

        console.log(`\nArtifacts saved to: ${runDir}`);
        if (exitCode === 0) {
            console.log('OK: Job page loaded successfully');
        }
    } catch (err) {
        // Save failure screenshot
        const failScreenshot = path.join(runDir, 'job_page_failure.png');
        await page.screenshot({ path: failScreenshot, fullPage: true }).catch(() => {});
        console.error(`Job page load failed. Screenshot: ${failScreenshot}`);

        // Save whatever we collected
        fs.writeFileSync(
            path.join(runDir, 'console_logs.json'),
            JSON.stringify(consoleLogs, null, 2)
        );
        fs.writeFileSync(
            path.join(runDir, 'network_logs.json'),
            JSON.stringify(networkLogs, null, 2)
        );

        throw err;
    } finally {
        await browser.close();
    }

    process.exit(exitCode);
}

main().catch(err => {
    console.error(err.message);
    process.exit(1);
});
