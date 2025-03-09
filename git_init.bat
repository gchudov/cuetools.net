git submodule update --init --recursive
git apply --directory=ThirdParty/flac ThirdParty/submodule_flac_CUETools.patch --whitespace=nowarn
powershell -c "Expand-Archive ThirdParty/MAC_SDK/MAC_1086_SDK.zip -DestinationPath ThirdParty/MAC_SDK/"
git apply --directory=ThirdParty/MAC_SDK ThirdParty/ThirdParty_MAC_SDK_CUETools.patch
git apply --directory=ThirdParty/taglib-sharp ThirdParty/submodule_taglib-sharp_CUETools.patch
git apply --directory=ThirdParty/WavPack ThirdParty/submodule_WavPack_CUETools.patch
git apply --directory=ThirdParty/WindowsMediaLib ThirdParty/submodule_WindowsMediaLib_CUETools.patch