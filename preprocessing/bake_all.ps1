$root = "D:\WAYMO_DATA\raw"; $proj = "C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D"
Get-ChildItem "$root\validation\vehicle_pose" -Filter *.parquet | ForEach-Object {
  $c = $_.BaseName
  $pd = "$proj\data\processed\$c"
  if ((Test-Path "$pd\frames\f0197.npz") -and -not (Test-Path "$pd\ground_tex.npz")) {
    "bake $c"; python "$proj\preprocessing\bake_ground.py" --root $root --context $c --split validation
  } else { "skip $c" }
}
