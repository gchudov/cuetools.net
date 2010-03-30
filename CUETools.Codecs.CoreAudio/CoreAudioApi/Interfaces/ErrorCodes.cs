using System;
using System.Collections.Generic;
using System.Text;

namespace NAudio.CoreAudioApi.Interfaces
{
    enum AudioClientErrors
    {
        /// <summary>
        /// AUDCLNT_E_NOT_INITIALIZED
        /// </summary>
        NotInitialized = unchecked((int)0x88890001),
        /// <summary>
        /// AUDCLNT_E_UNSUPPORTED_FORMAT
        /// </summary>
        UnsupportedFormat = unchecked((int)0x88890008),
        /// <summary>
        /// AUDCLNT_E_DEVICE_IN_USE
        /// </summary>
        DeviceInUse = unchecked((int)0x8889000A),
        
    }
}
