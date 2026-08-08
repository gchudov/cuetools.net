using CUETools.Codecs;
using CUETools.Processor;
using CUETools.Processor.Settings;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;

namespace CUETools.TestProcessor
{
    [TestClass]
    public class CUEConfigSecurityTest
    {
        [TestMethod]
        public void LoadRejectsUntrustedJsonTypeNames()
        {
            string testDir = Path.Combine(Path.GetTempPath(), "CUETools.TestProcessor", Guid.NewGuid().ToString("N"));
            string profileDir = Path.Combine(testDir, "CUE Tools");
            Directory.CreateDirectory(profileDir);

            try
            {
                UntrustedEncoderSettings.CreatedCount = 0;

                string typeName = typeof(UntrustedEncoderSettings).AssemblyQualifiedName;
                string json =
                    "{"
                    + "\"encoders\":[{"
                    + "\"$type\":\"" + typeName + "\","
                    + "\"Name\":\"Injected\","
                    + "\"Extension\":\"bad\","
                    + "\"Lossless\":true"
                    + "}],"
                    + "\"decoders\":[]"
                    + "}";
                File.WriteAllText(Path.Combine(profileDir, "settings.txt"), "Advanced=" + json);

                var reader = new SettingsReader("CUE Tools", "settings.txt", Path.Combine(testDir, "CUETools.exe"));
                var config = new CUEConfig();

                config.Load(reader);

                Assert.AreEqual(0, UntrustedEncoderSettings.CreatedCount);
                Assert.IsFalse(config.advanced.encoders.Any(item => item is UntrustedEncoderSettings));
            }
            finally
            {
                if (Directory.Exists(testDir))
                    Directory.Delete(testDir, true);
            }
        }

        [TestMethod]
        public void SaveAndLoadAllowsRegisteredJsonTypeNames()
        {
            string testDir = Path.Combine(Path.GetTempPath(), "CUETools.TestProcessor", Guid.NewGuid().ToString("N"));
            string appPath = Path.Combine(testDir, "CUETools.exe");

            try
            {
                var savedConfig = new CUEConfig();
                var writer = new SettingsWriter("CUE Tools", "settings.txt", appPath);
                savedConfig.Save(writer);
                writer.Close();

                var loadedConfig = new CUEConfig();
                loadedConfig.Load(new SettingsReader("CUE Tools", "settings.txt", appPath));

                Assert.IsTrue(loadedConfig.advanced.encoders.Any(item => item.GetType() == typeof(CUETools.Codecs.CommandLine.EncoderSettings)));
                Assert.IsTrue(loadedConfig.advanced.decoders.Any(item => item.GetType() == typeof(CUETools.Codecs.CommandLine.DecoderSettings)));
            }
            finally
            {
                if (Directory.Exists(testDir))
                    Directory.Delete(testDir, true);
            }
        }

        public class UntrustedEncoderSettings : IAudioEncoderSettings
        {
            public static int CreatedCount;

            public UntrustedEncoderSettings()
            {
                CreatedCount++;
            }

            public string Name { get; set; }
            public string Extension { get; set; }
            public Type EncoderType => typeof(UntrustedEncoderSettings);
            public bool Lossless { get; set; }
            public int Priority => 0;
            public string SupportedModes { get; set; }
            public string DefaultMode => EncoderMode;
            public string EncoderMode { get; set; }
            public AudioPCMConfig PCM { get; set; }
            public int BlockSize { get; set; }
            public int Padding { get; set; }

            public IAudioEncoderSettings Clone()
            {
                return this;
            }
        }
    }
}
