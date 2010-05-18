using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Codecs;

namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for CDRepairEncodeTest and is intended
    ///to contain all CDRepairEncodeTest Unit Tests
    ///</summary>
	[TestClass()]
	public class CDRepairEncodeTest
	{

		const int finalSampleCount = 44100 * 60 * 10; // 10 minutes long
		//const int stride = finalSampleCount * 2 / 32768;
		const int stride = 10 * 588 * 2;
		// CD has maximum of 360.000 sectors; 
		// If stride equals 10 sectors (10 sectors * 588 samples * 2 words),
		// then maximum sequence length is 36.000 sectors.
		// 36.000 is less than (65535 - npar), so we're ok here.
		// npar == 8 provides error correction for 4 samples out of every 36k,
		// i.e. at best one sample per 9k can be repaired.
		// Parity data per one CD requires 10 * 588 * 4 * npar bytes,
		// which equals 188.160b == 184kb
		// We might consider shorter strides to reduce parity data size,
		// but it probably should still be a multiple of 588 * 2;
		// (or the size of CD CIRC buffer?)

		const int npar = 8;
		static byte[] wav = new byte[finalSampleCount * 4];
		static AccurateRipVerify ar;
		static CDImageLayout toc;

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

		#region Additional test attributes
		// 
		//You can use the following additional attributes as you write your tests:
		//
		//Use ClassInitialize to run code before running the first test in the class
		[ClassInitialize()]
		public static void MyClassInitialize(TestContext testContext)
		{
			toc = new CDImageLayout(1, 1, 1, string.Format("0 {0}", (finalSampleCount / 588).ToString()));
			ar = new AccurateRipVerify(toc, null);
			new Random(2423).NextBytes(wav);
		}

		//Use ClassCleanup to run code after all tests in a class have run
		//[ClassCleanup()]
		//public static void MyClassCleanup()
		//{
		//}
		//
		//Use TestInitialize to run code before running each test
		//[TestInitialize()]
		//public void MyTestInitialize()
		//{
		//}
		//
		//Use TestCleanup to run code after each test has run
		//[TestCleanup()]
		//public void MyTestCleanup()
		//{
		//}
		//
		#endregion


		/// <summary>
		///A test for Write
		///</summary>
		[TestMethod()]
		public void CDRepairEncodeWriteTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode encode = new CDRepairEncode(ar, stride, npar, false, true);
			buff.Prepare(wav, finalSampleCount);
			ar.Init(toc);
			ar.Write(buff);
			ar.Close();
			Assert.AreEqual<byte>(8, encode.Parity[0]);
			Assert.AreEqual<uint>(2278257733, encode.CRC);
		}
	}
}
