<#
Record every network connection the experiment's processes open, without admin rights.

Polls the TCP table every 2 seconds and logs each non-loopback connection owned by Python
(the experiment, mem0) or Foundry Local (the main LLM). Polling can miss a connection shorter
than the interval, so treat this as supporting evidence. capture_network.ps1, which needs admin,
records every packet.

Usage: powershell -File scripts\watch_connections.ps1 -Out runs\connections.csv [-Seconds 3600]
#>
param(
    [string]$Out = "runs\connections.csv",
    [int]$Seconds = 0,
    [string[]]$Names = @("python", "foundry", "Inference.Service.Agent", "Foundry.Local")
)
$seen = @{}
$start = Get-Date
"time,process,pid,local,remote,state" | Out-File -Encoding utf8 $Out
while ($Seconds -eq 0 -or ((Get-Date) - $start).TotalSeconds -lt $Seconds) {
    $processes = @{}
    Get-Process | Where-Object { $name = $_.ProcessName; $Names | Where-Object { $name -like "$_*" } } |
        ForEach-Object { $processes[$_.Id] = $_.ProcessName }
    Get-NetTCPConnection -ErrorAction SilentlyContinue |
        Where-Object { $processes.ContainsKey([int]$_.OwningProcess) -and $_.RemoteAddress -notin @("127.0.0.1", "::1", "0.0.0.0", "::") } |
        ForEach-Object {
            $key = "$($_.OwningProcess)|$($_.RemoteAddress):$($_.RemotePort)"
            if (-not $seen.ContainsKey($key)) {
                $seen[$key] = $true
                "$(Get-Date -Format o),$($processes[[int]$_.OwningProcess]),$($_.OwningProcess),$($_.LocalAddress):$($_.LocalPort),$($_.RemoteAddress):$($_.RemotePort),$($_.State)" |
                    Out-File -Append -Encoding utf8 $Out
            }
        }
    Start-Sleep -Seconds 2
}
