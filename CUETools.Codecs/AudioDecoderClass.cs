using System;

namespace CUETools.Codecs
{
    /// <summary>
    ///    This class provides an attribute for marking
    ///    classes that provide <see cref="IAudioSource" />.
    /// </summary>
    /// <remarks>
    ///    When plugins with classes that provide <see cref="IAudioSource" /> are
    ///    registered, their <see cref="AudioDecoderClass" /> attributes are read.
    /// </remarks>
    /// <example>
    ///    <code lang="C#">using CUETools.Codecs;
    ///
    ///[AudioDecoderClass("libFLAC", "flac")]
    ///public class MyDecoder : IAudioSource {
    ///	...
    ///}</code>
    /// </example>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public sealed class AudioDecoderClass : Attribute
    {
        public string DecoderName
        {
            get;
            set;
        }

        public string Extension
        {
            get;
            set;
        }

        public int Priority
        {
            get;
            set;
        }

        public AudioDecoderClass(string decoderName, string extension, int priority)
        {
            DecoderName = decoderName;
            Extension = extension;
            Priority = priority;
        }
    }
}
