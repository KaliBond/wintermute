# Usage: powershell -ExecutionPolicy Bypass -File cowork-snapshot.ps1 [message]
# Commits all current changes with a timestamped label for easy rollback.

param([string]$message = "")

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$label = if ($message) { "$message - $timestamp" } else { "Cowork snapshot - $timestamp" }

git -C $PSScriptRoot add -A
$staged = git -C $PSScriptRoot diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing to commit - working tree clean."
    exit 0
}

git -C $PSScriptRoot commit -m $label
Write-Host ""
Write-Host "Snapshot committed: $label"
Write-Host "To rollback: git reset --hard HEAD~1"
