using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using CUETools.Codecs;
using CUETools.Codecs.FLAC;

namespace CUETools.TestCodecs
{
    [TestClass()]
    public class FlacWriterTest
    {
        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        /// <summary>
        ///A test for FlacWriter (string, int, int, int, Stream)
        ///</summary>
        [TestMethod()]
        public void ConstructorTest()
        {
            AudioBuffer buff = WAVReader.ReadAllSamples("test.wav", null);
            FLACWriter target;

            target = new FLACWriter("flacwriter2.flac", new FLACWriterSettings() { PCM = buff.PCM });
            target.Settings.Padding = 1;
            target.Settings.BlockSize = 32;
            //target.Vendor = "CUETools";
            //target.CreationTime = DateTime.Parse("15 Aug 1976");
            target.FinalSampleCount = buff.Length;
            target.Write(buff);
            target.Close();
            CollectionAssert.AreEqual(File.ReadAllBytes("flacwriter1.flac"), File.ReadAllBytes("flacwriter2.flac"), "flacwriter2.flac doesn't match.");
        }

        [TestMethod()]
        public void SeekTest()
        {
            var r = new FLACReader("test.flac", null);
            var buff1 = new AudioBuffer(r, 16536);
            var buff2 = new AudioBuffer(r, 16536);
            Assert.AreEqual(0, r.Position);
            r.Read(buff1, 7777);
            Assert.AreEqual(7777, r.Position);
            r.Position = 0;
            Assert.AreEqual(0, r.Position);
            r.Read(buff2, 7777);
            Assert.AreEqual(7777, r.Position);
            AudioBufferTest.AreEqual(buff1, buff2);
            r.Read(buff1, 7777);
            Assert.AreEqual(7777+7777, r.Position);
            r.Position = 7777;
            Assert.AreEqual(7777, r.Position);
            r.Read(buff2, 7777);
            Assert.AreEqual(7777+7777, r.Position);
            AudioBufferTest.AreEqual(buff1, buff2);
            r.Close();
        }
    }
}
