"""Build a standalone single-file Windows installer for Papertilling Desktop Explorer.

    python tools/build_windows_installer.py [VERSION]

Uses Inno Setup 6 (`ISCC.exe`) when installed; otherwise compiles a native per-user
Windows GUI installer (`dist/papertilling-<version>-windows-x86_64-setup.exe`) using the
built-in Windows .NET Framework C# compiler (`csc.exe`), requiring no external tools.
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_EXE = ROOT / "rust" / "target" / "release" / "papertilling-app.exe"
ICON_ICO = ROOT / "rust" / "papertilling-app" / "assets" / "icon.ico"
ISS_FILE = ROOT / "rust" / "papertilling-app" / "installer" / "windows" / "papertilling.iss"
LICENSE_FILE = ROOT / "LICENSE"
DIST_DIR = ROOT / "dist"

CS_INSTALLER = r"""
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("Papertilling Setup")]
[assembly: AssemblyProduct("Papertilling")]
[assembly: AssemblyVersion("__VER4__")]

static class SetupProgram {
    const string AppName = "Papertilling";
    const string AppVer = "__VER__";
    const string UninstKey = @"Software\Microsoft\Windows\CurrentVersion\Uninstall\Papertilling";

    [STAThread]
    static int Main(string[] args) {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        string defDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Papertilling");

        foreach (string a in args) {
            if (a.Equals("--uninstall", StringComparison.OrdinalIgnoreCase)) {
                return RunUninstall(Path.GetDirectoryName(Application.ExecutablePath));
            }
            if (a.Equals("/S", StringComparison.OrdinalIgnoreCase) || a.Equals("--silent", StringComparison.OrdinalIgnoreCase)) {
                PerformInstall(defDir, true, false);
                return 0;
            }
        }

        using (Form f = new Form()) {
            f.Text = "Papertilling " + AppVer + " Setup";
            f.ClientSize = new Size(470, 255);
            f.FormBorderStyle = FormBorderStyle.FixedDialog;
            f.MaximizeBox = false;
            f.StartPosition = FormStartPosition.CenterScreen;
            f.Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);

            PictureBox pic = new PictureBox { Bounds = new Rectangle(20, 20, 48, 48), SizeMode = PictureBoxSizeMode.StretchImage };
            if (f.Icon != null) pic.Image = f.Icon.ToBitmap();

            Label title = new Label {
                Text = "Install Papertilling " + AppVer,
                Font = new Font(SystemFonts.MessageBoxFont.FontFamily, 12f, FontStyle.Bold),
                Bounds = new Rectangle(80, 20, 370, 26)
            };
            Label sub = new Label {
                Text = "Desktop explorer for 2D wallpaper lattices, defects, and surfaces.",
                Bounds = new Rectangle(80, 48, 370, 22)
            };
            Label dirLbl = new Label { Text = "Destination folder:", Bounds = new Rectangle(20, 86, 200, 18) };
            TextBox dirBox = new TextBox { Text = defDir, Bounds = new Rectangle(20, 106, 350, 24) };
            Button browse = new Button { Text = "Browse...", Bounds = new Rectangle(376, 104, 74, 26) };
            browse.Click += (s, e) => {
                using (FolderBrowserDialog d = new FolderBrowserDialog { SelectedPath = dirBox.Text }) {
                    if (d.ShowDialog(f) == DialogResult.OK) dirBox.Text = d.SelectedPath;
                }
            };

            CheckBox cbDesk = new CheckBox { Text = "Create a Desktop shortcut", Checked = true, Bounds = new Rectangle(20, 144, 300, 22) };
            CheckBox cbRun = new CheckBox { Text = "Launch Papertilling when setup finishes", Checked = true, Bounds = new Rectangle(20, 170, 340, 22) };

            Button btnInstall = new Button { Text = "Install", Bounds = new Rectangle(284, 210, 80, 28) };
            Button btnCancel = new Button { Text = "Cancel", Bounds = new Rectangle(370, 210, 80, 28) };
            f.AcceptButton = btnInstall;
            f.CancelButton = btnCancel;

            btnCancel.Click += (s, e) => f.Close();
            btnInstall.Click += (s, e) => {
                try {
                    btnInstall.Enabled = false;
                    f.Cursor = Cursors.WaitCursor;
                    PerformInstall(dirBox.Text.Trim(), cbDesk.Checked, cbRun.Checked);
                    f.Cursor = Cursors.Default;
                    MessageBox.Show(f, "Papertilling has been installed successfully.", "Papertilling Setup", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    f.Close();
                } catch (Exception ex) {
                    f.Cursor = Cursors.Default;
                    btnInstall.Enabled = true;
                    MessageBox.Show(f, "Installation failed:\n" + ex.Message, "Papertilling Setup", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            };

            f.Controls.AddRange(new Control[] { pic, title, sub, dirLbl, dirBox, browse, cbDesk, cbRun, btnInstall, btnCancel });
            Application.Run(f);
        }
        return 0;
    }

    static void ExtractGzipResource(string resName, string destPath) {
        Assembly asm = Assembly.GetExecutingAssembly();
        using (Stream raw = asm.GetManifestResourceStream(resName))
        using (GZipStream gz = new GZipStream(raw, CompressionMode.Decompress))
        using (FileStream fs = File.Create(destPath)) {
            gz.CopyTo(fs);
        }
    }

    static void CreateShortcut(string lnkPath, string targetPath, string desc) {
        Type t = Type.GetTypeFromProgID("WScript.Shell");
        dynamic shell = Activator.CreateInstance(t);
        dynamic sc = shell.CreateShortcut(lnkPath);
        sc.TargetPath = targetPath;
        sc.WorkingDirectory = Path.GetDirectoryName(targetPath);
        sc.Description = desc;
        sc.IconLocation = targetPath + ",0";
        sc.Save();
    }

    static void PerformInstall(string installDir, bool desktopShortcut, bool launchAfter) {
        Directory.CreateDirectory(installDir);
        string appExe = Path.Combine(installDir, "papertilling-app.exe");
        string licTxt = Path.Combine(installDir, "LICENSE.txt");
        string uninsExe = Path.Combine(installDir, "uninstall-papertilling.exe");

        ExtractGzipResource("app.exe.gz", appExe);
        ExtractGzipResource("license.txt.gz", licTxt);
        File.Copy(Application.ExecutablePath, uninsExe, true);

        string startMenu = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Papertilling.lnk");
        CreateShortcut(startMenu, appExe, "Explore 2D wallpaper lattices, defects, and surfaces");
        if (desktopShortcut) {
            string desk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Papertilling.lnk");
            CreateShortcut(desk, appExe, "Explore 2D wallpaper lattices, defects, and surfaces");
        }

        using (RegistryKey k = Registry.CurrentUser.CreateSubKey(UninstKey)) {
            k.SetValue("DisplayName", AppName);
            k.SetValue("DisplayVersion", AppVer);
            k.SetValue("Publisher", "antimajorana"); // package author
            k.SetValue("URLInfoAbout", "https://github.com/antimajorana/papertilling");
            k.SetValue("DisplayIcon", appExe);
            k.SetValue("InstallLocation", installDir);
            k.SetValue("UninstallString", "\"" + uninsExe + "\" --uninstall");
            k.SetValue("NoModify", 1, RegistryValueKind.DWord);
            k.SetValue("NoRepair", 1, RegistryValueKind.DWord);
        }

        if (launchAfter) {
            Process.Start(new ProcessStartInfo(appExe) { WorkingDirectory = installDir, UseShellExecute = true });
        }
    }

    static int RunUninstall(string dir) {
        if (MessageBox.Show("Uninstall Papertilling from your computer?", "Uninstall Papertilling",
                MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes) return 0;
        string startMenu = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Papertilling.lnk");
        string desk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Papertilling.lnk");
        if (File.Exists(startMenu)) File.Delete(startMenu);
        if (File.Exists(desk)) File.Delete(desk);
        Registry.CurrentUser.DeleteSubKeyTree(UninstKey, false);
        foreach (string fn in new[] { "papertilling-app.exe", "LICENSE.txt" }) {
            string p = Path.Combine(dir, fn);
            if (File.Exists(p)) try { File.Delete(p); } catch { }
        }
        MessageBox.Show("Papertilling has been uninstalled.", "Uninstall Papertilling", MessageBoxButtons.OK, MessageBoxIcon.Information);
        return 0;
    }
}
"""


def find_iscc() -> Path | None:
    if hit := shutil.which("iscc"):
        return Path(hit)
    for cand in (
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ):
        if cand.is_file():
            return cand
    return None


def main() -> int:
    ver = sys.argv[1] if len(sys.argv) > 1 else "0.2.0"
    if not APP_EXE.is_file():
        raise SystemExit(f"Release binary not found at {APP_EXE}. Build it first with:\n"
                         "  cargo build --release --manifest-path rust/Cargo.toml -p papertilling-app")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_exe = DIST_DIR / f"papertilling-{ver}-windows-x86_64-setup.exe"

    if iscc := find_iscc():
        subprocess.run([str(iscc), f"/DMyAppVersion={ver}", str(ISS_FILE)], check=True)
        print(f"Built Inno Setup installer: {out_exe} ({out_exe.stat().st_size / 1e6:.2f} MB)")
        return 0

    csc = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    if not csc.is_file():
        raise SystemExit("Neither Inno Setup (ISCC.exe) nor .NET Framework csc.exe was found on this machine.")

    nums = [p for p in ver.split("-")[0].split(".") if p.isdigit()]
    ver4 = ".".join((nums + ["0", "0", "0", "0"])[:4])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        app_gz = tmp / "app.exe.gz"
        lic_gz = tmp / "license.txt.gz"
        app_gz.write_bytes(gzip.compress(APP_EXE.read_bytes(), compresslevel=9))
        lic_gz.write_bytes(gzip.compress(LICENSE_FILE.read_bytes(), compresslevel=9))
        cs_file = tmp / "Setup.cs"
        cs_file.write_text(CS_INSTALLER.replace("__VER__", ver).replace("__VER4__", ver4), encoding="utf-8")
        cmd = [
            str(csc), "/nologo", "/optimize+", "/target:winexe",
            f"/out:{out_exe}",
            f"/win32icon:{ICON_ICO}",
            f"/resource:{app_gz},app.exe.gz",
            f"/resource:{lic_gz},license.txt.gz",
            "/reference:System.dll",
            "/reference:System.Core.dll",
            "/reference:System.Drawing.dll",
            "/reference:System.Windows.Forms.dll",
            "/reference:Microsoft.CSharp.dll",
            str(cs_file),
        ]
        subprocess.run(cmd, check=True)
    print(f"Built standalone Windows installer: {out_exe} ({out_exe.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
