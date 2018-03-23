using System;

namespace CUETools.Codecs
{
    /// <summary>
    ///    This class provides an attribute for marking
    ///    classes that provide <see cref="IAudioDest" />.
    /// </summary>
    /// <remarks>
    ///    When plugins with classes that provide <see cref="IAudioDest" /> are
    ///    registered, their <see cref="AudioEncoderClassAttribute" /> attributes are read.
    /// </remarks>
    /// <example>
    ///    <code lang="C#">using CUETools.Codecs;
    ///
    ///[AudioEncoderClass(typeof(MyEncoderSettings))]
    ///public class MyEncoder : IAudioDest {
    ///	...
    ///}</code>
    /// </example>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class AudioEncoderClassAttribute : Attribute
    {
        public Type Settings
        {
            get;
            private set;
        }

        public AudioEncoderClassAttribute(Type settings)
        {
            Settings = settings;
        }
    }
}
