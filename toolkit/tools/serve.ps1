# Tiny static file server for the AI Toolkit.
# Browsers refuse to read local .json files from a double-clicked HTML page,
# so the toolkit must be served over http://localhost. No dependencies needed.
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\serve.ps1 [-Port 8321] [-NoBrowser]

param(
    [int]$Port = 8321,
    [switch]$NoBrowser
)

$root = Split-Path -Parent $PSScriptRoot   # the toolkit folder (parent of tools\)
$entry = "index.html"

$mime = @{
    ".html" = "text/html; charset=utf-8"
    ".js"   = "text/javascript; charset=utf-8"
    ".json" = "application/json; charset=utf-8"
    ".css"  = "text/css; charset=utf-8"
    ".txt"  = "text/plain; charset=utf-8"
    ".png"  = "image/png"
    ".webp" = "image/webp"
    ".svg"  = "image/svg+xml"
    ".ico"  = "image/x-icon"
    ".woff" = "font/woff"
    ".woff2" = "font/woff2"
}

$listener = New-Object System.Net.HttpListener
$prefix = "http://localhost:$Port/"
$listener.Prefixes.Add($prefix)
try {
    $listener.Start()
} catch {
    Write-Host "Could not start on port $Port ($($_.Exception.Message)). Is the toolkit already running?" -ForegroundColor Yellow
    Write-Host "If it is, just open: $prefix" -ForegroundColor Yellow
    if (-not $NoBrowser) { Start-Process ($prefix + [uri]::EscapeDataString($entry)) }
    exit 1
}

$url = $prefix + [uri]::EscapeDataString($entry)
Write-Host ""
Write-Host "  AI Training Toolkit is running." -ForegroundColor Green
Write-Host "  Serving: $root"
Write-Host "  Open:    $url"
Write-Host ""
Write-Host "  Keep this window open while using the toolkit. Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) { Start-Process $url }

while ($listener.IsListening) {
    try { $ctx = $listener.GetContext() } catch { break }
    $res = $ctx.Response
    try {
        $relPath = [uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath).TrimStart("/")
        if ($relPath -eq "") { $relPath = $entry }
        $full = [IO.Path]::GetFullPath((Join-Path $root $relPath))
        # Refuse anything that escapes the toolkit folder
        if (-not $full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path $full -PathType Leaf)) {
            $res.StatusCode = 404
            $body = [Text.Encoding]::UTF8.GetBytes("404 - not found: $relPath")
        } else {
            $ext = [IO.Path]::GetExtension($full).ToLowerInvariant()
            $type = $mime[$ext]; if (-not $type) { $type = "application/octet-stream" }
            $res.ContentType = $type
            $res.Headers.Add("Cache-Control", "no-cache")   # always pick up fresh JSON edits
            $body = [IO.File]::ReadAllBytes($full)
        }
        $res.ContentLength64 = $body.Length
        $res.OutputStream.Write($body, 0, $body.Length)
    } catch {
        # client aborted or file error - ignore and keep serving
    } finally {
        try { $res.Close() } catch {}
    }
}
