const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const os = require('os');
const http = require('http');

const HOME = os.homedir();
const DIR = path.join(HOME, 'yolo-editor-app');
const VENV_DIR = path.join(HOME, 'yolo-editor-venv');

let PYTHON_PATH = 'python';
const venvPython = path.join(VENV_DIR, 'Scripts', 'python.exe');
const venvPython2 = path.join(VENV_DIR, 'bin', 'python');
if (fs.existsSync(venvPython)) PYTHON_PATH = venvPython;
else if (fs.existsSync(venvPython2)) PYTHON_PATH = venvPython2;

console.log('==================================================');
console.log('  YOLO Dataset Editor (React) - Setup');
console.log('==================================================\n');
console.log('  Project:  ' + DIR);
console.log('  Python:   ' + PYTHON_PATH + '\n');

if (!fs.existsSync(DIR)) {
    console.log('[1/4] Creating project...');
    fs.mkdirSync(DIR, { recursive: true });
    fs.mkdirSync(path.join(DIR, 'src'), { recursive: true });
    console.log('      Done!\n');
} else {
    console.log('[1/4] Project exists\n');
}

fs.writeFileSync(path.join(DIR, 'package.json'), JSON.stringify({
    name: 'yolo-editor', private: true, version: '1.0.0', type: 'module',
    scripts: { dev: 'vite --open' },
    dependencies: { react: '^18.2.0', 'react-dom': '^18.2.0' },
    devDependencies: { '@vitejs/plugin-react': '^4.2.0', vite: '5.4.11' }
}, null, 2));

// IMPORTANT: proxy uses 127.0.0.1 not localhost (avoids IPv6 ::1 issue)
fs.writeFileSync(path.join(DIR, 'vite.config.js'),
`import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:3777'
    }
  }
})
`);

if (!fs.existsSync(path.join(DIR, 'index.html'))) {
    fs.writeFileSync(path.join(DIR, 'index.html'),
`<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" /><title>YOLO Dataset Editor</title></head>
<body><div id="root"></div><script type="module" src="/src/main.jsx"></script></body>
</html>
`);
}

if (!fs.existsSync(path.join(DIR, 'node_modules'))) {
    console.log('[2/4] Installing npm dependencies...');
    execSync('npm install', { cwd: DIR, stdio: 'inherit' });
    console.log('');
} else {
    console.log('[2/4] npm dependencies installed\n');
}

console.log('[3/4] Configuring editor...');
const jsxSource = path.join(__dirname, 'yolo-dataset-editor.jsx');
if (!fs.existsSync(jsxSource)) { console.error('[ERROR] yolo-dataset-editor.jsx not found!'); process.exit(1); }
fs.copyFileSync(jsxSource, path.join(DIR, 'src', 'YoloEditor.jsx'));

fs.writeFileSync(path.join(DIR, 'src', 'App.jsx'),
`import YoloEditor from './YoloEditor'\nexport default function App() { return <YoloEditor /> }\n`);
fs.writeFileSync(path.join(DIR, 'src', 'main.jsx'),
`import React from 'react'\nimport ReactDOM from 'react-dom/client'\nimport App from './App'\nReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>)\n`);
fs.writeFileSync(path.join(DIR, 'src', 'index.css'), `* { margin: 0; padding: 0; box-sizing: border-box; }\n`);

const pyServerSource = path.join(__dirname, 'yolo_server.py');
if (fs.existsSync(pyServerSource)) {
    fs.copyFileSync(pyServerSource, path.join(DIR, 'yolo_server.py'));
}
console.log('      Ready!\n');

// --- Launch ---
console.log('[4/4] Starting servers...\n');

// Start Python Flask server FIRST
console.log('  Starting Python server...');
const pyServer = spawn(PYTHON_PATH, [path.join(DIR, 'yolo_server.py')], {
    cwd: DIR, stdio: 'inherit',
    env: { ...process.env }
});

// Wait for Python server to be ready before starting Vite
function waitForServer(url, maxRetries, delay) {
    return new Promise((resolve, reject) => {
        let attempts = 0;
        function check() {
            attempts++;
            http.get(url, (res) => {
                resolve();
            }).on('error', () => {
                if (attempts >= maxRetries) {
                    reject(new Error('Python server did not start'));
                } else {
                    setTimeout(check, delay);
                }
            });
        }
        check();
    });
}

waitForServer('http://127.0.0.1:3777/api/health', 30, 1000)
    .then(() => {
        console.log('\n  Python server is ready!\n');
        console.log('==================================================');
        console.log('  Frontend: http://localhost:5173');
        console.log('  Backend:  http://127.0.0.1:3777 (Python+GPU)');
        console.log('  Press Ctrl+C to stop');
        console.log('==================================================\n');

        // Now start Vite
        const isWin = process.platform === 'win32';
        const viteBin = isWin ? path.join(DIR, 'node_modules', '.bin', 'vite.cmd') : path.join(DIR, 'node_modules', '.bin', 'vite');
        const viteServer = spawn(isWin ? viteBin : 'node', isWin ? ['--open'] : [viteBin, '--open'], { cwd: DIR, stdio: 'inherit', shell: isWin });

        process.on('SIGINT', () => { pyServer.kill(); viteServer.kill(); process.exit(0); });
        viteServer.on('exit', () => { pyServer.kill(); process.exit(0); });
    })
    .catch((err) => {
        console.error('\n  [ERROR] ' + err.message);
        console.error('  Check the Python output above for errors.');
        pyServer.kill();
        process.exit(1);
    });

pyServer.on('exit', (code) => {
    if (code) {
        console.error('  [ERROR] Python server crashed (code ' + code + ')');
        process.exit(1);
    }
});
