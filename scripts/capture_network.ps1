<#
Capture every packet while an experiment command runs, to show which traffic leaves the laptop.

Needs an administrator PowerShell, because Windows' packet monitor (pktmon) requires it. Run it
yourself; the script never asks for elevation.

It writes capture.etl, capture.pcapng (open it in Wireshark) and capture.txt to the output folder,
plus connections.csv from watch_connections.ps1, which says which process owned each connection.
Loopback traffic between the experiment and Foundry Local doesn't cross a network adapter, so
anything in the capture really left the device or tried to.

Usage, from the InferX folder in an administrator PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\capture_network.ps1 -Out runs\network_final -Command "<command>"
#>
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$Out = "runs\network"
)
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Run this from an administrator PowerShell: pktmon needs it."
    exit 1
}
New-Item -ItemType Directory -Force $Out | Out-Null
$etl = Join-Path $Out "capture.etl"
$watcher = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
    "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "watch_connections.ps1"), "-Out", (Join-Path $Out "connections.csv"))
pktmon filter remove | Out-Null
pktmon start --capture --pkt-size 256 --file-name $etl | Out-Null
"Capture started at $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $Out "notes.txt")
try {
    Invoke-Expression $Command
}
finally {
    pktmon stop | Out-Null
    Stop-Process -Id $watcher.Id -ErrorAction SilentlyContinue
    "Capture stopped at $(Get-Date -Format o)" | Tee-Object -Append -FilePath (Join-Path $Out "notes.txt")
    pktmon etl2pcap $etl --out (Join-Path $Out "capture.pcapng") | Out-Null
    pktmon etl2txt $etl --out (Join-Path $Out "capture.txt") | Out-Null
}
