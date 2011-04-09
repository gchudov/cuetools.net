using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUETools.Codecs;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.TestHelpers;

namespace CUETools.TestParity
{
    /// <summary>
    ///This is a test class for CDRepairDecodeTest and is intended
    ///to contain all CDRepairDecodeTest Unit Tests
    ///</summary>
	[TestClass()]
	public class CDRepairDecodeTest
	{
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
		const int stride = 10 * 588 * 2;
		const int npar = 8;
		const int errors = stride / 4;
		const int offset = 48;
		const int seed = 2423;
		//const int offset = 5 * 588 - 5;
		//const int offset = 2000;

		private static TestImageGenerator generator;
		private static CDRepairEncode encode;
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
			generator = new TestImageGenerator("0 9801", seed, 32 * 588, 0);
			encode = generator.CreateCDRepairEncode(stride, npar, false, true);
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
		///A test for Write
		///</summary>
		[TestMethod()]
		public void CDRepairEncodeWriteTest()
		{
			Assert.AreEqual<string>("jvR9QJ1cSWo=", Convert.ToBase64String(encode.Parity, 0, 8));
			Assert.AreEqual<uint>(377539636, encode.CRC);
		}

		/// <summary>
		///Verifying rip that is accurate
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeOriginalTest()
		{
			var decode = generator.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsFalse(hasErrors, "has errors");
			Assert.AreEqual(0, actualOffset, "wrong offset");
		}

		/// <summary>
		///Verifying rip that is accurate with pregap
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeOriginalWithPregapTest()
		{
			var generator2 = new TestImageGenerator("32 9833", seed, 0, 0);
			var decode = generator2.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(-1176, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(encode.Parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
		}

		/// <summary>
		///Verifying rip that has errors
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeModifiedTest()
		{
			var generator2 = new TestImageGenerator("0 9801", seed, 32 * 588, errors);
			var decode = generator2.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(0, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(encode.Parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
			generator2.Write(fix);
			Assert.AreEqual<uint>(encode.CRC, fix.CRC);
		}

		/// <summary>
		///Verifying rip that has positive offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodePositiveOffsetTest()
		{
			var generator2 = new TestImageGenerator("0 9801", seed, 32 * 588 + offset, 0);
			var decode = generator2.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsFalse(hasErrors, "has errors");
			Assert.AreEqual(offset, actualOffset, "wrong offset");
		}

		/// <summary>
		///Verifying rip that has negative offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeNegativeOffsetTest()
		{
			var generator2 = new TestImageGenerator("0 9801", seed, 32 * 588 - offset, 0);
			var decode = generator2.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsFalse(hasErrors, "has errors");
			Assert.AreEqual(-offset, actualOffset, "wrong offset");
		}

		/// <summary>
		///Verifying rip that has errors and positive offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodePositiveOffsetErrorsTest()
		{
			var generator2 = new TestImageGenerator("0 9801", seed, 32 * 588 + offset, errors);
			var decode = generator2.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(offset, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(encode.Parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
			generator2.Write(fix);
			Assert.AreEqual<uint>(encode.CRC, fix.CRC);
		}

		/// <summary>
		///Verifying rip that has errors and negative offset
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeNegativeOffsetErrorsTest()
		{
			var generator2 = new TestImageGenerator("0 999 9801", seed, 32 * 588 - offset, errors);
			var decode = generator2.CreateCDRepairEncode(stride, npar, true, false);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(-offset, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(encode.Parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
			generator2.Write(fix);
			Assert.AreEqual<uint>(encode.CRC, fix.CRC);
		}
	}
}
