using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUETools.Codecs;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.TestHelpers;
using CUETools.Parity;

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
			encode = generator.CreateCDRepairEncode(stride, npar);
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
			Assert.AreEqual<string>("jvR9QJ1cSWpqbyP0I0tBrBkQRjCDTDDQkttZGj14ROvsXyg+AnnxVKxL7gwLZbrQmTw5ZPps1Q3744g94qaOOQ==", 
				Convert.ToBase64String(encode.Parity, 0, 64));
			Assert.AreEqual<uint>(377539636, encode.CRC);
		}

		/// <summary>
		///Verifying rip that is accurate
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeOriginalTest()
		{
			var decode = generator.CreateCDRepairEncode(stride, npar);
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
			var decode = generator2.CreateCDRepairEncode(stride, npar);
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
			var decode = generator2.CreateCDRepairEncode(stride, npar);
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
			var decode = generator2.CreateCDRepairEncode(stride, npar);
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
			var decode = generator2.CreateCDRepairEncode(stride, npar);
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
			var decode = generator2.CreateCDRepairEncode(stride, npar);
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
			var decode = generator2.CreateCDRepairEncode(stride, npar);
			int actualOffset;
			bool hasErrors;
			Assert.IsTrue(decode.FindOffset(encode.NPAR, encode.Parity, 0, encode.CRC, out actualOffset, out hasErrors), "couldn't find offset");
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(-offset, actualOffset, "wrong offset");
			CDRepairFix fix = decode.VerifyParity(encode.Parity, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
			generator2.Write(fix);
			Assert.AreEqual<uint>(encode.CRC, fix.CRC);
		}

		[TestMethod]
		public void GFMiscTest()
		{
			var g16 = Galois16.instance;
			CollectionAssert.AreEqual(new int[] { 5, 33657, 33184, 33657, 5 }, g16.gfconv(new int[] { 1, 2, 3 }, new int[] { 4, 3, 2 }));
			CollectionAssert.AreEqual(new int[] { 5, 6, 1774, 4, 5 }, g16.gfconv(new int[] { 1, 2, 3 }, new int[] { 4, -1, 2 }));

			var g8 = Galois81D.instance;
			Assert.AreEqual(111, g8.gfadd(11, 15));
			Assert.AreEqual(11, g8.gfadd(11, -1));
			Assert.AreEqual(25, g8.gfadd(1, 0));

			var S = new int[8] { -1, -1, -1, -1, -1, -1, -1, -1 };
			var received = new int[] { 2, 4, 1, 3, 5 };
			for (int ii = 0; ii < 8; ii++)
				for (int x = 0; x < 5; x++)
					S[ii] = g8.gfadd(S[ii], g8.gfmul(received[x], g8.gfpow(ii + 1, x)));
			CollectionAssert.AreEqual(S, new int[] { 219, 96, 208, 202, 116, 211, 182, 129 });

			//S[ii] ^= received[x] * a ^ ((ii + 1) * x);
			//S[0] ^= received[0] * a ^ (1 * 0);
			//S[0] ^= received[1] * a ^ (1 * 1);
			//S[0] ^= received[2] * a ^ (1 * 2);

			//S[1] ^= received[0] * a ^ (2 * 0);
			//S[1] ^= received[1] * a ^ (2 * 1);
			//S[1] ^= received[2] * a ^ (2 * 2);

			received = g8.toExp(received);
			for (int ii = 0; ii < 8; ii++)
			{
				S[ii] = 0;
				for (int x = 0; x < 5; x++)
					S[ii] ^= g8.mulExp(received[x], ((ii + 1) * x) % 255);
			}
			S = g8.toLog(S);
			CollectionAssert.AreEqual(S, new int[] { 219, 96, 208, 202, 116, 211, 182, 129 });
		}

		/// <summary>
		///A test for CRC parralelism
		///</summary>
		[TestMethod()]
		public void CDRepairSplitTest()
		{
			var seed = 723722;
			var ar0 = new TestImageGenerator("13 68 99 136", seed, 0, 0).CreateCDRepairEncode(stride, npar);
			var splits = new int[] { 1, 13 * 588 - 1, 13 * 588, 13 * 588 + 1, 30 * 588 - 1, 30 * 588, 30 * 588 + 1, 68 * 588 - 1, 68 * 588, 68 * 588 + 1 };
			foreach (int split in splits)
			{
				var ar1 = new TestImageGenerator("13 68 99 136", seed, 0, 0, 0, split).CreateCDRepairEncode(stride, npar);
				var ar2 = new TestImageGenerator("13 68 99 136", seed, 0, 0, split, (int)ar0.FinalSampleCount).CreateCDRepairEncode(stride, npar);
				ar1.AR.Combine(ar2.AR, split, (int)ar0.FinalSampleCount);
				var offsets = new int[] { 0, -1, 1, -2, 2, -3, 3, -4, 4, -11, 11, -256, 256, -588, 588, 1 - 588 * 5, 588 * 5 - 1 };
				string message = "split = " + CDImageLayout.TimeToString((uint)split / 588) + "." + (split % 588).ToString();
				Assert.AreEqual(ar0.CRC, ar1.CRC, "CRC was not set correctly, " + message);
				CollectionAssert.AreEqual(ar0.Parity, ar1.Parity, "Parity was not set correctly, " + message);
			}
		}

		/// <summary>
		///A test for CRC parralelism speed
		///</summary>
		[TestMethod()]
		public unsafe void CDRepairSplitSpeedTest()
		{
			var seed = 723722;
			var split = 20 * 588;
			var ar1 = new TestImageGenerator("13 68 99 136", seed, 0, 0, 0, split).CreateCDRepairEncode(stride, npar);
			var ar2 = new TestImageGenerator("13 68 99 136", seed, 0, 0, split, (int)ar1.FinalSampleCount).CreateCDRepairEncode(stride, npar);
			for (int i = 0; i < 20; i++)
				ar1.AR.Combine(ar2.AR, split, (int)ar1.FinalSampleCount);
		}

		/// <summary>
		///A test for Syndrome2Parity speed
		///</summary>
		[TestMethod()]
		public unsafe void CDRepairSyndrome2ParitySpeedTest()
		{
			byte[] parityCopy = new byte[encode.Parity.Length];
			for (int t = 0; t < 100; t++)
			{
				fixed (byte* p = encode.Parity, p1 = parityCopy)
				fixed (ushort* syn = encode.Syndrome)
				{
					ushort* p2 = (ushort*)p1;
					for (int i = 0; i < stride; i++)
						Galois16.instance.Syndrome2Parity(syn + i * npar, p2 + i * npar, npar);
				}
			}
			CollectionAssert.AreEqual(encode.Parity, parityCopy);
		}

		[TestMethod]
		public unsafe void CDRepairEncodeSynParTest()
		{
			byte[] parityCopy = new byte[encode.Parity.Length];
			fixed(byte * p = encode.Parity, p1 = parityCopy)
				fixed (ushort *syn = encode.Syndrome)
				{
					ushort* p2 = (ushort*)p1;
					for (int i = 0; i < stride; i++)
						Galois16.instance.Syndrome2Parity(syn + i * npar, p2 + i * npar, npar);
				}
			CollectionAssert.AreEqual(encode.Parity, parityCopy);
		}

		[TestMethod]
		public void CDRepairEncodeSpeedTest()
		{
			var generator = new TestImageGenerator("0 75000", seed, 0, 0);
			var encode = generator.CreateCDRepairEncode(stride, npar);
			Assert.AreEqual<string>("CWgEDNLjSi22nIOyaeyp+12R3UCVWlzIb+nbv8XWXg9YEhkHxYr8xqrr1+hIbFwKNEXnj0esJrKbiW3XGbHsYw==",
				Convert.ToBase64String(encode.Parity, 0, 64));
			var syndrome = encode.Syndrome;
			var bsyn = new byte[64];
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 8; j++)
				{
					bsyn[(i * 8 + j) * 2] = (byte)syndrome[i, j];
					bsyn[(i * 8 + j) * 2 + 1] = (byte)(syndrome[i, j] >> 8);
				}
			Assert.AreEqual<string>("YJPyo4+KY35P+DpljMplMGbMWXmpvhkdDOCKeEo4NDoRPPW7D0cv8hmLb7yZujp0sVg/6AEWKY5QrDKkiYp0Zw==",
				Convert.ToBase64String(bsyn));
		}
	}
}
