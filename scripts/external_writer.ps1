# Have an outside model write the test timelines described in data/external/brief.md.
#
# Runs opencode in a new empty folder outside the repo, so the writer cannot see InferX's simulator or
# data. One person per call, in one opencode session (--continue), so the writer knows who it already wrote.
# Afterwards copies the files into data/external/<run>/ with the brief and a record of the model used.
#
# Usage: powershell -File scripts/external_writer.ps1 [-Model opencode/deepseek-v4.1-flash-free] [-People 5]
param(
    [string]$Model = "opencode/deepseek-v4.1-flash-free",
    [int]$People = 5
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$opencode = "$env:LOCALAPPDATA\Programs\opencode\opencode.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$work = Join-Path $HOME "external_writer\$stamp"
New-Item -ItemType Directory -Force $work | Out-Null
# The brief is the only file in the folder; the prompts just point to it (no quotes, so Windows PowerShell
# passes them intact).
Copy-Item (Join-Path $repo "data\external\brief.md") (Join-Path $work "brief.md")
Set-Location $work

for ($i = 1; $i -le $People; $i++) {
    if ($i -eq 1) {
        $prompt = "Read brief.md in this folder and follow it. Start now: write person 1 only, to the file " +
                  "person_1.json in this folder. Write the whole file; do not stop early or leave placeholders."
        $flags = @()
    } else {
        $prompt = "Now write person $i only, to person_$i.json in this folder, following the same brief. Make them " +
                  "clearly different from the people you already wrote. Write the whole file; do not stop early or " +
                  "leave placeholders."
        $flags = @("--continue")
    }
    $log = Join-Path $work "opencode_person_$i.log"
    Write-Output "person ${i}: writing with $Model"
    # opencode writes progress to stderr, which Windows PowerShell would turn into a terminating error.
    $ErrorActionPreference = "Continue"
    & $opencode run @flags -m $Model $prompt 2>&1 | ForEach-Object { "$_" } | Out-File $log -Encoding utf8
    $ErrorActionPreference = "Stop"
    if (Test-Path (Join-Path $work "person_$i.json")) {
        Write-Output "person ${i}: done ($((Get-Item (Join-Path $work "person_$i.json")).Length) bytes)"
    } else {
        Write-Output "person ${i}: no file written; see $log"
    }
}

$out = Join-Path $repo "data\external\$stamp"
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item (Join-Path $work "person_*.json"), (Join-Path $work "opencode_person_*.log") $out -ErrorAction SilentlyContinue
Copy-Item (Join-Path $repo "data\external\brief.md") $out
@{ model = $Model; opencode = (& $opencode --version); people = $People; written = (Get-Date -Format o); workdir = $work } |
    ConvertTo-Json | Set-Content (Join-Path $out "writer.json") -Encoding utf8
Write-Output "WRITER_DONE $out"
