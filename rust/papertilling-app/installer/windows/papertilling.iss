; Inno Setup 6 installer script for Escher Desktop Explorer (Windows x86_64)
; Non-admin per-user install by default (PrivilegesRequired=lowest), with dialog override.

#ifndef MyAppVersion
  #define MyAppVersion "0.2.0"
#endif

[Setup]
AppId={{8F64D4B1-7E29-4C8E-9A52-3E5B08C71942}
AppName=Escher
AppVersion={#MyAppVersion}
; package author
AppPublisher=Escher contributors
AppPublisherURL=https://github.com/antimajorana/Escher
AppSupportURL=https://github.com/antimajorana/Escher/issues
AppUpdatesURL=https://github.com/antimajorana/Escher/releases
DefaultDirName={autopf}\Escher
DefaultGroupName=Escher
DisableProgramGroupPage=yes
LicenseFile=..\..\..\..\LICENSE
OutputDir=..\..\..\..\dist
OutputBaseFilename=escher-{#MyAppVersion}-windows-x86_64-setup
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\Escher.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\..\target\release\papertilling-app.exe"; DestDir: "{app}"; DestName: "Escher.exe"; Flags: ignoreversion
Source: "..\..\..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
Type: files; Name: "{autoprograms}\Papertilling.lnk"
Type: files; Name: "{autodesktop}\Papertilling.lnk"

[Icons]
Name: "{autoprograms}\Escher"; Filename: "{app}\Escher.exe"; Comment: "Explore 2D wallpaper lattices, crystalline defects, and surfaces"
Name: "{autodesktop}\Escher"; Filename: "{app}\Escher.exe"; Tasks: desktopicon; Comment: "Explore 2D wallpaper lattices, crystalline defects, and surfaces"

[Run]
Filename: "{app}\Escher.exe"; Description: "{cm:LaunchProgram,Escher}"; Flags: nowait postinstall skipifsilent
