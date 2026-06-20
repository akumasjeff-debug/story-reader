@echo off
echo 使用 Python 3.12 安裝套件中...
echo.

echo [1/2] 安裝基本套件...
py -3.12 -m pip install flask edge-tts deep-translator Pillow pyopenssl

echo.
echo [2/2] 選擇 OCR 引擎：
echo   (A) Gemini Flash ^(快速、輕量，需 API Key^) - 建議先選這個
echo   (B) EasyOCR ^(離線，首次下載 ~2GB^)
echo.
set /p choice=輸入 A 或 B：

if /i "%choice%"=="A" (
    py -3.12 -m pip install google-generativeai
    echo.
    echo 請編輯 config.env，填入你的 Gemini API Key。
    echo 取得免費 Key：https://aistudio.google.com/app/apikey
)

if /i "%choice%"=="B" (
    py -3.12 -m pip install easyocr numpy
    echo.
    echo EasyOCR 安裝完成。首次辨識會自動下載模型（約 2GB）。
)

echo.
echo 安裝完成！執行 run.bat 啟動程式。
pause
