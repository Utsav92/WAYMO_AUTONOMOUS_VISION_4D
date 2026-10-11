# Sends REAL key events (W held, then Shift+W) to the TouchDesigner output window and reports what TD saw.
param([int]$ProcId = 0, [double]$Hold = 1.2)
Add-Type @'
using System; using System.Text; using System.Runtime.InteropServices; using System.Collections.Generic;
public class WinX {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
  public static List<string> List(uint pid){ var r=new List<string>(); EnumWindows((h,l)=>{ uint p; GetWindowThreadProcessId(h,out p); if(p==pid && IsWindowVisible(h)){ var sb=new StringBuilder(256); GetWindowText(h,sb,256); r.Add(h.ToInt64()+"|"+sb); } return true;}, IntPtr.Zero); return r; }
}
'@
if ($ProcId -eq 0) { $ProcId = (Get-Process TouchDesigner | Where-Object { $_.MainWindowTitle -match 'WAYMO' } | Select-Object -First 1).Id }
$wins = [WinX]::List([uint32]$ProcId)
$wins | ForEach-Object { "window: $_" }
# the perform window has an empty or short title; fall back to the main window
$target = $wins | Where-Object { $_ -notmatch 'TouchDesigner 2025' } | Select-Object -First 1
if (-not $target) { $target = $wins | Select-Object -First 1 }
$h = [IntPtr][int64]($target.Split('|')[0])
[WinX]::ShowWindow($h, 9) | Out-Null
[WinX]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Milliseconds 600
$VK_W = 0x57; $VK_SHIFT = 0x10
[WinX]::keybd_event($VK_W, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Seconds $Hold
[WinX]::keybd_event($VK_W, 0, 2, [UIntPtr]::Zero)
"sent W for $Hold s to: $target"
