#!/usr/bin/env node
/**
 * Playwright headless login to CVAT (two-step login form).
 *
 * Step 1: Fill #credential (username), press Enter to reveal password field.
 * Step 2: Fill #password, click "Next" button to submit.
 *
 * Reads CVAT_E2E_USER, CVAT_E2E_PASSWORD, CVAT_E2E_HOST from .env file.
 * Saves auth-state to temp/e2e_sam2/check/auth-state.json.
 *
 * Usage:
 *   NODE_PATH=/home/inaho-omen/temp/playwright-cli/node_modules node scripts/e2e/sam2/playwright_login.js
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

    const username = env.CVAT_E2E_USER;
    const password = env.CVAT_E2E_PASSWORD;
    const host = env.CVAT_E2E_HOST || 'http://localhost:8080';

    if (!username || !password) {
        console.error('ERROR: CVAT_E2E_USER and CVAT_E2E_PASSWORD must be set in .env');
        process.exit(1);
    }

    const authStatePath = path.join(repoRoot, 'temp', 'e2e_sam2', 'check', 'auth-state.json');
    fs.mkdirSync(path.dirname(authStatePath), { recursive: true });

    const { chromium } = require('playwright');

    console.log(`Logging in to ${host} as ${username} ...`);

    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    const consoleErrors = [];
    page.on('console', msg => {
        if (msg.type() === 'error') {
            consoleErrors.push(msg.text());
        }
    });

    try {
        // Navigate to login page
        await page.goto(`${host}/auth/login`, { waitUntil: 'networkidle', timeout: 30000 });

        // Step 1: Fill credential (username) field
        await page.fill('#credential', username);
        await page.keyboard.press('Enter');

        // Wait for password field to appear
        await page.waitForSelector('#password', { state: 'visible', timeout: 10000 });

        // Step 2: Fill password and click Next
        await page.fill('#password', password);
        await page.click('button.cvat-credentials-action-button');

        // Wait for navigation to tasks page
        await page.waitForURL('**/tasks**', { timeout: 15000 });

        console.log(`Login successful. Current URL: ${page.url()}`);

        // Save storage state (cookies + localStorage)
        await context.storageState({ path: authStatePath });
        console.log(`Auth state saved to: ${authStatePath}`);

        // Verify by checking /api/users/self
        const resp = await page.request.get(`${host}/api/users/self`);
        if (resp.ok()) {
            const user = await resp.json();
            console.log(`Verified: username=${user.username}, is_staff=${user.is_staff}, is_superuser=${user.is_superuser}`);
        } else {
            console.error(`WARNING: /api/users/self returned ${resp.status()}`);
        }
    } catch (err) {
        // Save screenshot and errors on failure
        const ts = new Date().toISOString().replace(/[:.]/g, '').slice(0, 15);
        const runDir = path.join(repoRoot, 'temp', 'e2e_sam2', 'run_' + ts);
        fs.mkdirSync(runDir, { recursive: true });
        const screenshotPath = path.join(runDir, 'login_failure.png');
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.error(`Login failed. Screenshot saved to: ${screenshotPath}`);
        if (consoleErrors.length > 0) {
            const consolePath = path.join(runDir, 'login_console_errors.json');
            fs.writeFileSync(consolePath, JSON.stringify(consoleErrors, null, 2));
            console.error(`Console errors saved to: ${consolePath}`);
        }
        throw err;
    } finally {
        await browser.close();
    }
}

main().catch(err => {
    console.error(err.message);
    process.exit(1);
});
