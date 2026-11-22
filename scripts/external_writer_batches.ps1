# Write the training pool for the mixed controller: 4 batches of 5 people, 2 per writer model.
# DeepSeek is left out on purpose: it wrote the held-out test people (data/external/20261118-195649), so the
# retrained controller is tested on a writer it never trained on.
$repo = Split-Path -Parent $PSScriptRoot
$avoid = "Hannah, freelance UX designer in Manchester; Aarti, chemistry teacher in Jaipur; Desmond, plumbing " +
         "supplies owner in Cape Town; Beatriz, design student in Sao Paulo; Siobhan, midwife in Galway"
foreach ($model in "opencode/nemotron-3.5-super-vl-free", "opencode/longcat-2.5-free",
                   "opencode/nemotron-3.5-super-vl-free", "opencode/longcat-2.5-free") {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "external_writer.ps1") `
        -Model $model -People 5 -Avoid $avoid
}
Write-Output "ALL_BATCHES_DONE"
