#define MyAppName "Assessment System for Biodiversity Impacts on Ecosystem Service Functions"
#define MyAppVersion "1.0"
#define MyAppPublisher "JeasunLok"
#define MyAppURL "https://github.com/JeasunLok/Nanling_Biogain"
#define MyAppExeName "NanlingBioGain.exe"
#define MyAppSourceDir "..\dist\NanlingBioGain"
#define MyAppIcon "..\assets\Nanling_Biogain.ico"
#define MyAppLicense "..\LICENSE"

[Setup]
AppId={{4A322C61-0D40-48E0-A2C5-DF3A711C1A87}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=Desktop application for assessing biodiversity impacts on ecosystem service functions
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoCopyright=Copyright (c) {#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=NanlingBioGain_v1.0_setup
SetupIconFile={#MyAppIcon}
LicenseFile={#MyAppLicense}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}
ChangesAssociations=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
