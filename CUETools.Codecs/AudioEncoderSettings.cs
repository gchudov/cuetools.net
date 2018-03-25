using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;

#if NET20
    namespace System.Runtime.CompilerServices
    {
        [AttributeUsage(AttributeTargets.Assembly | AttributeTargets.Class
             | AttributeTargets.Method)]
        public sealed class ExtensionAttribute : Attribute { }
    }
#endif

namespace CUETools.Codecs
{
    public interface IAudioEncoderSettings
    {
        string Name { get; }

        string Extension { get; }

        Type EncoderType { get; }

        bool Lossless { get; }

        int Priority { get; }

        string SupportedModes { get; }

        string DefaultMode { get; }

        string EncoderMode { get; set; }

        AudioPCMConfig PCM { get; set; }

        int BlockSize { get; set; }

        int Padding { get; set; }

        IAudioEncoderSettings Clone();
    }

    public static class IAudioEncoderSettingsExtensions
    {
        public static bool HasBrowsableAttributes(this IAudioEncoderSettings settings)
        {
            bool hasBrowsable = false;
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(settings))
            {
                bool isBrowsable = true;
                foreach (var attribute in property.Attributes)
                {
                    var browsable = attribute as BrowsableAttribute;
                    isBrowsable &= browsable == null || browsable.Browsable;
                }
                hasBrowsable |= isBrowsable;
            }
            return hasBrowsable;
        }

        public static int GetEncoderModeIndex(this IAudioEncoderSettings settings)
        {
            return new List<string>(settings.SupportedModes.Split(' ')).FindIndex(m => m == settings.EncoderMode);
        }

        public static void SetEncoderModeIndex(this IAudioEncoderSettings settings, int value)
        {
            string[] modes = settings.SupportedModes.Split(' ');
            if (modes.Length == 0 && value < 0)
                return;
            if (value < 0 || value >= modes.Length)
                throw new IndexOutOfRangeException();
            settings.EncoderMode = modes[value];
        }

        public static void SetDefaultValuesForMode(this IAudioEncoderSettings settings)
        {
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(settings))
                if (!property.CanResetValue(settings))
                    foreach (var attribute in property.Attributes)
                        if (attribute is DefaultValueForModeAttribute)
                        {
                            var defaultValueForMode = attribute as DefaultValueForModeAttribute;
                            property.SetValue(settings, defaultValueForMode.m_values[settings.GetEncoderModeIndex()]);
                        }
        }

        public static bool HasDefaultValuesForMode(this IAudioEncoderSettings settings, int index)
        {
            bool res = true;
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(settings))
                foreach (var attribute in property.Attributes)
                    if (attribute is DefaultValueForModeAttribute)
                    {
                        var defaultValueForMode = attribute as DefaultValueForModeAttribute;
                        res &= property.GetValue(settings).Equals(defaultValueForMode.m_values[index]);
                    }
            return res;
        }

        public static int GuessEncoderMode(this IAudioEncoderSettings settings)
        {
            // return new List<string>(settings.SupportedModes.Split(' ')).FindIndex(m => settings.HasDefaultValuesForMode(m));
            string[] modes = settings.SupportedModes.Split(' ');
            if (modes == null || modes.Length < 1)
                return -1;
            for (int i = 0; i < modes.Length; i++)
                if (settings.HasDefaultValuesForMode(i))
                    return i;
            return -1;
        }

        public static void Init(this IAudioEncoderSettings settings, AudioPCMConfig pcm = null)
        {
            // Iterate through each property and call ResetValue()
            foreach (PropertyDescriptor property in TypeDescriptor.GetProperties(settings))
                property.ResetValue(settings);
            settings.EncoderMode = settings.DefaultMode;
            settings.PCM = pcm;
        }
    }
}
