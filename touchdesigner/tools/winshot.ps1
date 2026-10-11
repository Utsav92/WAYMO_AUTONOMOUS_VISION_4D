# Capture the TD output window to a PNG:  winshot.ps1 -Out file.png [-DragFrom x,y -DragTo x,y] (window-relative px)
param([string]$Out = "C:\Users\Utsav\WAYMO_AUTONOMOUS_VISION_4D\output\screenshots\window.png", [string]$DragFrom = "", [string]$DragTo = "")
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System; using System.Text; using System.Runtime.InteropServices; using System.Collections.Generic;
public class WinY {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, int dx, int dy, uint d, UIntPtr e);
  public static IntPtr Find(string needle, uint wantPid){ IntPtr found=IntPtr.Zero; EnumWindows((h,l)=>{ uint pid; GetWindowThreadProcessId(h,out pid); if(pid!=wantPid) return true; var sb=new StringBuilder(300); GetWindowText(h,sb,300); if(IsWindowVisible(h) && sb.ToString().EndsWith(needle)){ found=h; return false;} return true;}, IntPtr.Zero); return found; }
}
'@
$tdpid = (Get-Process TouchDesigner | Where-Object { $_.MainWindowTitle -match "WAYMO" } | Select-Object -First 1).Id
$h = [WinY]::Find("perform_window", [uint32]$tdpid)
if ($h -eq [IntPtr]::Zero) { "perform window not found"; exit 1 }
if ($DragFrom -ne "") { [WinY]::SetForegroundWindow($h) | Out-Null; Start-Sleep -Milliseconds 500 }
$cr = New-Object WinY+RECT; [WinY]::GetClientRect($h, [ref]$cr) | Out-Null
$pt = New-Object WinY+POINT; $pt.X = 0; $pt.Y = 0; [WinY]::ClientToScreen($h, [ref]$pt) | Out-Null
$w = $cr.R - $cr.L; $hh = $cr.B - $cr.T
if ($DragFrom -ne "") {
  $a = $DragFrom.Split(","); $b = $DragTo.Split(",")
  [WinY]::SetCursorPos($pt.X + [int]$a[0], $pt.Y + [int]$a[1]) | Out-Null; Start-Sleep -Milliseconds 150
  [WinY]::mouse_event(0x02, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 150     # left down
  $steps = 12
  for ($i = 1; $i -le $steps; $i++) {
    $x = [int]$a[0] + ([int]$b[0] - [int]$a[0]) * $i / $steps; $y = [int]$a[1] + ([int]$b[1] - [int]$a[1]) * $i / $steps
    [WinY]::SetCursorPos($pt.X + [int]$x, $pt.Y + [int]$y) | Out-Null; Start-Sleep -Milliseconds 40
  }
  [WinY]::mouse_event(0x04, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 400     # left up
}
$bmp = New-Object System.Drawing.Bitmap $w, $hh
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
[WinY]::PrintWindow($h, $hdc, 3) | Out-Null          # PW_CLIENTONLY | PW_RENDERFULLCONTENT: only this window, even if occluded
$g.ReleaseHdc($hdc)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
"saved $Out  client=${w}x${hh} origin=($($pt.X),$($pt.Y))"
