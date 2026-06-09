# YOLO Dataset Editor

Web-редактор датасетов YOLO: разметка, обучение и экспорт.  
Стек: **Flask** (API) + **React/Vite** (UI) + **Ultralytics YOLO**.

## Запуск из исходников

```bat
run_yolo_editor.bat
```

Откроется http://127.0.0.1:3777/

## Portable-сборка

```powershell
.\build_portable.ps1
```

Результат:

- `dist\YOLO-Dataset-Editor\` — папка с `YOLO-Dataset-Editor.exe`
- `portable_dist\YOLO-Dataset-Editor-Portable.zip` — архив для распространения (~265 МБ)

Требования для сборки: Node.js, venv `%USERPROFILE%\yolo-editor-venv` с ultralytics/flask/opencv.

## Структура

| Файл | Назначение |
|------|------------|
| `yolo_server.py` | Flask API и логика редактора |
| `yolo-dataset-editor.jsx` | UI-компонент (копируется в `web/src/`) |
| `yolo_editor_frozen.py` | Точка входа PyInstaller |
| `web/` | Vite + React |
| `build_portable.ps1` | Скрипт сборки portable |
