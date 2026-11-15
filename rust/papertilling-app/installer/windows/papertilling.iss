; Inno Setup 6 installer script for Papertilling Desktop Explorer (Windows x86_64)
; Non-admin per-user install by default (PrivilegesRequired=lowest), with dialog override.

#ifndef MyAppVersion
  #define MyAppVersion "0.2.0"
#endif

[Setup]
AppId={{8F64D4B1-7E29-4C8E-9A52-3E5B08C71942}
AppName=Papertilling
AppVersion={#MyAppVersion}
; package author
AppPublisher=Papertilling contributors
AppPublisherURL=https://github.com/antimajorana/papertilling
AppSupportURL=https://github.com/antimajorana/papertilling/issues
AppUpdatesURL=https://github.com/antimajorana/papertilling/releases
DefaultDirName={autopf}\Papertilling
DefaultGroupName=Papertilling
DisableProgramGroupPage=yes
LicenseFile=..\..\..\..\LICENSE
OutputDir=..\..\..\..\dist
OutputBaseFilename=papertilling-{#MyAppVersion}-windows-x86_64-setup
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\papertilling-app.exe
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
Source: "..\..\..\target\release\papertilling-app.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Papertilling"; Filename: "{app}\papertilling-app.exe"; Comment: "Explore 2D wallpaper lattices, crystalline defects, and surfaces"
Name: "{autodesktop}\Papertilling"; Filename: "{app}\papertilling-app.exe"; Tasks: desktopicon; Comment: "Explore 2D wallpaper lattices, crystalline defects, and surfaces"

[Run]
Filename: "{app}\papertilling-app.exe"; Description: "{cm:LaunchProgram,Papertilling}"; Flags: nowait postinstall skipifsilent
