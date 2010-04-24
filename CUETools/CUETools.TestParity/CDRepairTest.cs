using System;
using CUETools.Codecs;
using CUETools.AccurateRip;
using CUETools.CDImage;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for CDRepairTest and is intended
    ///to contain all CDRepairTest Unit Tests
    ///</summary>
	[TestClass()]
	public class CDRepairTest
	{


		private TestContext testContextInstance;

		const int finalSampleCount = 44100 * 60 * 10; // 20 minutes long
		//const int stride = finalSampleCount * 2 / 32768;
		const int stride = 10 * 588 * 2;
		const int npar = 8;
		static byte[] wav = new byte[finalSampleCount * 4];
		static byte[] wav2 = new byte[finalSampleCount * 4];
		static byte[] parity;
		static uint crc;
		static CDRepairEncode decode;
		static CDRepairEncode decode2;
		static CDRepairFix fix;
		static CDRepairFix fix2;
		const int offset = 48;
		static AccurateRipVerify ar, ar2, ar3;
		static CDImageLayout toc;

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
			ar2 = new AccurateRipVerify(toc, null);
			ar3 = new AccurateRipVerify(toc, null);

			new Random(2423).NextBytes(wav);
			new Random(2423).NextBytes(wav2);
			Random rnd = new Random(987);
			for (int i = 0; i < stride / 4; i++ )
				wav2[(int)(rnd.NextDouble() * (wav2.Length - 1))] = (byte)(rnd.NextDouble() * 255);

			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode encode = new CDRepairEncode(ar, stride, npar, false, true);
			buff.Prepare(wav, finalSampleCount);
			ar.Init();
			ar.Write(buff);
			ar.Close(); 
			parity = encode.Parity;
			crc = encode.CRC;

			decode = new CDRepairEncode(ar2, stride, npar, true, false);
			buff.Prepare(wav2, finalSampleCount);
			ar2.Init();
			ar2.Write(buff);
			ar2.Close(); 

			int actualOffset;
			bool hasErrors;
			decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors);
			fix = decode.VerifyParity(parity, actualOffset);

			decode2 = new CDRepairEncode(ar3, stride, npar, true, false);
			ar3.Init();
			buff.Prepare(new byte[offset * 4], offset);
			ar3.Write(buff);
			buff.Prepare(wav2, finalSampleCount - offset);
			ar3.Write(buff);
			ar3.Close();
			decode2.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors);
			fix2 = decode2.VerifyParity(parity, actualOffset);
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
		//Use TestCleanup to run code after each test has run
		//[TestCleanup()]
		//public void MyTestCleanup()
		//{
		//}
		//
		#endregion


		/// <summary>
		///A test for CDRepair Constructor
		///</summary>
		[TestMethod()]
		public void CDRepairFixTest()
		{
			Assert.IsTrue(fix.HasErrors);
			Assert.IsTrue(fix.CanRecover);
			Assert.AreEqual(0, fix.ActualOffset, "wrong offset");

			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			buff.Prepare(wav2, finalSampleCount);
			fix.Write(buff);
			fix.Close();

			Assert.AreEqual<uint>(crc, fix.CRC);
		}

		/// <summary>
		///Repair with offset
		///</summary>
		[TestMethod()]
		public void CDRepairFixWithOffsetTest()
		{
			Assert.IsTrue(fix2.HasErrors);
			Assert.IsTrue(fix2.CanRecover);
			Assert.AreEqual(-offset, fix2.ActualOffset, "wrong offset");

			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			buff.Prepare(new byte[offset * 4], offset);
			fix2.Write(buff);
			buff.Prepare(wav2, finalSampleCount - offset);
			fix2.Write(buff);
			fix2.Close();

			Assert.AreEqual<uint>(crc, fix2.CRC);
		}
	}
}
