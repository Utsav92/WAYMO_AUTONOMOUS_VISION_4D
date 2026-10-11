# Export every downloaded Perception context (staging dir first, then published to data/processed when complete).
param([string]$Root = "D:\WAYMO_DATA\raw", [string]$Staging = "D:\WAYMO_DATA\staging")
$proj = "C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D"
$ctxs = Get-ChildItem "$Root\validation\vehicle_pose" -Filter *.parquet | ForEach-Object { $_.BaseName }
foreach ($c in $ctxs) {
  $done = Test-Path "$proj\data\processed\$c\frames\f0197.npz"
  if ($done) { "skip $c (already processed)"; continue }
  if (-not (Test-Path "$Staging\$c\meta.json")) {
    "export $c"
    & python "$proj\preprocessing\export_frames.py" --root $Root --context $c --split validation --semantic "$proj\models\seg.pt" --out $Staging
  }
  if (Test-Path "$Staging\$c\meta.json") {
    $dst = "$proj\data\processed\$c"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item "$Staging\$c\*" $dst -Recurse -Force        # depth_err (if any) is kept; frames/cam/meta replaced
    "published $c"
  }
}
