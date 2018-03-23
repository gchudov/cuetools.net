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
    ///[AudioDecoderClass(typeof(MyDecoderSettings))]
    ///public class MyDecoder : IAudioSource {
    ///	...
    ///}</code>
    /// </example>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public sealed class AudioDecoderClassAttribute : Attribute
    {
        public Type Settings
        {
            get;
            private set;
        }

        public AudioDecoderClassAttribute(Type settings)
        {
            Settings = settings;
        }
    }
}
