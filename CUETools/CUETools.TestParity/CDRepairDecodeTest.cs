using System;
using CUETools.CDRepair;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUETools.Codecs;

namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for CDRepairDecodeTest and is intended
    ///to contain all CDRepairDecodeTest Unit Tests
    ///</summary>
	[TestClass()]
	public class CDRepairDecodeTest
	{

		const int finalSampleCount = 44100 * 60 * 10 + 20; // 10 minutes long
		//const int stride = finalSampleCount * 2 / 32768;
		const int stride = 10 * 588 * 2;
		const int npar = 8;
		static byte[] wav = new byte[finalSampleCount * 4];
		static byte[] wav2 = new byte[finalSampleCount * 4];
		static byte[] wav3 = new byte[finalSampleCount * 4];
		static byte[] parity;
		static uint crc;
		const int offset = 48;
		//const int offset = 5 * 588 - 5;
		//const int offset = 2000;

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
			new Random(2423).NextBytes(wav);
			new Random(2423).NextBytes(wav2);
			Random rnd = new Random(987);
			for (int i = 0; i < stride / 4; i++)
				wav2[(int)(rnd.NextDouble() * (wav2.Length - 1))] = (byte)(rnd.NextDouble() * 255);

			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode encode = new CDRepairEncode(finalSampleCount, stride, npar, false, true);
			buff.Prepare(wav, finalSampleCount);
			encode.Write(buff);
			encode.Close();
			parity = encode.Parity;
			crc = encode.CRC;
		}
		//
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
		///Verifying rip that is accurate
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeOriginalTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode decode = new CDRepairEncode(finalSampleCount, stride, npar, true, false);
			buff.Prepare(wav, finalSampleCount);
			decode.Write(buff);
			decode.Close();
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors));
			Assert.IsFalse(hasErrors, "has errors");
			Assert.AreEqual(0, actualOffset, "wrong offset");
		}

		/// <summary>
		///Verifying rip that has errors
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeModifiedTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode decode = new CDRepairEncode(finalSampleCount, stride, npar, true, false);
			buff.Prepare(wav2, finalSampleCount);
			decode.Write(buff);
			decode.Close();
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(0, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
		}

		/// <summary>
		///Verifying rip that has positive offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodePositiveOffsetTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode decode = new CDRepairEncode(finalSampleCount, stride, npar, true, false);
			Array.Copy(wav, offset * 4, wav3, 0, (finalSampleCount - offset) * 4);
			buff.Prepare(wav3, finalSampleCount);
			decode.Write(buff);
			decode.Close();
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors));
			Assert.IsFalse(hasErrors, "has errors");
			Assert.AreEqual(offset, actualOffset, "wrong offset");
		}

		/// <summary>
		///Verifying rip that has negative offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeNegativeOffsetTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode decode = new CDRepairEncode(finalSampleCount, stride, npar, true, false);
			buff.Prepare(new byte[offset * 4], offset);
			decode.Write(buff);
			buff.Prepare(wav, finalSampleCount - offset);
			decode.Write(buff);
			decode.Close();
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors));
			Assert.IsFalse(hasErrors, "has errors");
			Assert.AreEqual(-offset, actualOffset, "wrong offset");
		}

		/// <summary>
		///Verifying rip that has errors and positive offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodePositiveOffsetErrorsTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode decode = new CDRepairEncode(finalSampleCount, stride, npar, true, false);
			Array.Copy(wav2, offset * 4, wav3, 0, (finalSampleCount - offset) * 4);
			buff.Prepare(wav3, finalSampleCount);
			decode.Write(buff);
			decode.Close();
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(offset, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
		}

		/// <summary>
		///Verifying rip that has errors and negative offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeNegativeOffsetErrorsTest()
		{
			AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, 0);
			CDRepairEncode decode = new CDRepairEncode(finalSampleCount, stride, npar, true, false);
			buff.Prepare(new byte[offset * 4], offset);
			decode.Write(buff);
			buff.Prepare(wav2, finalSampleCount - offset);
			decode.Write(buff);
			decode.Close();
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(npar, parity, 0, crc, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(-offset, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
		}
	}
}
