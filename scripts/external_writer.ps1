# Have an outside model write the test timelines described in data/external/brief.md.
#
# Runs opencode in a new empty folder outside the repo (only brief.md is in it), so the writer cannot see
# InferX's simulator or data. One person per call, in one opencode session (--continue), so the writer knows
# who it already wrote in this batch. A person the model fails to write is retried up to 3 times.
# Afterwards copies the files into data/external/<stamp>_<model>/ with the brief and a record of the model.
#
# -Avoid lists people (name, job, place) the writer must not reuse, such as the held-out test people.
#
# Usage: powershell -File scripts/external_writer.ps1 [-Model opencode/deepseek-v4.1-flash-free] [-People 5] [-Avoid "..."]
param(
    [string]$Model = "opencode/deepseek-v4.1-flash-free",
    [int]$People = 5,
    [string]$Avoid = ""
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$opencode = "$env:LOCALAPPDATA\Programs\opencode\opencode.exe"
$stamp = (Get-Date -Format "yyyyMMdd-HHmmss") + "_" + ($Model -replace '^.*/', '' -replace '[^A-Za-z0-9.-]', '-')
$work = Join-Path $HOME "external_writer\$stamp"
New-Item -ItemType Directory -Force $work | Out-Null
# The brief is the only file in the folder; the prompts just point to it (no quotes, so Windows PowerShell
# passes them intact).
Copy-Item (Join-Path $repo "data\external\brief.md") (Join-Path $work "brief.md")
Set-Location $work
$avoidText = ""
if ($Avoid) { $avoidText = " Do not reuse any of these people or anyone very like them: $Avoid." }

for ($i = 1; $i -le $People; $i++) {
    for ($try = 1; $try -le 3 -and -not (Test-Path (Join-Path $work "person_$i.json")); $try++) {
        if ($i -eq 1 -and $try -eq 1) {
            $prompt = "Read brief.md in this folder and follow it. Start now: write person 1 only, to the file " +
                      "person_1.json in this folder.$avoidText Create the file with your write tool and write the " +
                      "whole file; do not stop early or leave placeholders."
            $flags = @()
        } else {
            $prompt = "Now write person $i only, to person_$i.json in this folder, following brief.md. Make them " +
                      "clearly different from the people you already wrote.$avoidText Create the file with your write " +
                      "tool and write the whole file; do not stop early or leave placeholders."
            $flags = @("--continue")
        }
        $log = Join-Path $work "opencode_person_${i}_try$try.log"
        Write-Output "person ${i}: writing with $Model (try $try)"
        # opencode writes progress to stderr, which Windows PowerShell would turn into a terminating error.
        $ErrorActionPreference = "Continue"
        & $opencode run @flags -m $Model $prompt 2>&1 | ForEach-Object { "$_" } | Out-File $log -Encoding utf8
        $ErrorActionPreference = "Stop"
    }
    if (Test-Path (Join-Path $work "person_$i.json")) {
        Write-Output "person ${i}: done ($((Get-Item (Join-Path $work "person_$i.json")).Length) bytes)"
    } else {
        Write-Output "person ${i}: not written after 3 tries"
    }
}

$out = Join-Path $repo "data\external\$stamp"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item (Join-Path $work "person_*.json"), (Join-Path $work "opencode_person_*.log") $out -ErrorAction SilentlyContinue
Copy-Item (Join-Path $repo "data\external\brief.md") $out
@{ model = $Model; opencode = (& $opencode --version); people = $People; avoid = $Avoid; written = (Get-Date -Format o);
   workdir = $work } | ConvertTo-Json | Set-Content (Join-Path $out "writer.json") -Encoding utf8
Write-Output "WRITER_DONE $out"
