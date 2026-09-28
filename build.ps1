# build.ps1 — Скрипт сборки и запуска проекта neiro

Write-Host "Очистка старой сборки..." -ForegroundColor Cyan
if (Test-Path "build") {
    Remove-Item -Recurse -Force build
}

Write-Host "Конфигурация CMake (Ninja + UCRT64 GCC)..." -ForegroundColor Cyan
cmake -B build -S . -G Ninja -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка конфигурации CMake!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Сборка проекта..." -ForegroundColor Cyan
cmake --build build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Ошибка компиляции!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Запуск приложения..." -ForegroundColor Green
& ".\build\neiro.exe"