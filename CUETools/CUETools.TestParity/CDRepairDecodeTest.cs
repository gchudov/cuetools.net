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
		const int errors = stride / 4;
		const int offset = 48;
		const int seed = 2423;
		//const int offset = 5 * 588 - 5;
		//const int offset = 2000;

		private static TestImageGenerator generator;
		private static CDRepairEncode encode;
        private static string[] encodeSyndrome = new string[33];
        private static string[] encodeParity = new string[33];
        private static string[] encodeParity1 = new string[33];
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
			encode = generator.CreateCDRepairEncode(stride);
            encodeSyndrome[4] = "DP7tAM2tuWBe7kb/A3o5hcS+o59uoT1ckHh9Am+wZxA=";
            encodeSyndrome[8] = "DP7tAM2tuWCBRjyLjt6a+l7uRv8DejmFzRtv3ofeEWzEvqOfbqE9XFOz/6WaYU+lkHh9Am+wZxCw3m1Y7zKctw==";
            encodeSyndrome[16] = "DP7tAM2tuWCBRjyLjt6a+lr7hvwnJWrfZ0MGKOYwFmVe7kb/A3o5hc0bb96H3hFsIcjxCpERbjnJjVvLc5NDJcS+o59uoT1cU7P/pZphT6WaQ4f3L/ImdyD5psk3fWRvkHh9Am+wZxCw3m1Y7zKct8QUsJHnLA6wcmxT/LmmQdE=";
            encodeParity[8] = "jvR9QJ1cSWpqbyP0I0tBrBkQRjCDTDDQkttZGj14ROvsXyg+AnnxVKxL7gwLZbrQmTw5ZPps1Q3744g94qaOOQ==";
            encodeParity[16] = "gwln1GxlYWH/Jn74PreMLv4aFF2glkScSWVFlxMBx94v5D3/3wPx+2guRLquED0s9tOFikPLiSnAv0Xq8aIQ6Q==";
            encodeParity1[8] = "CWgEDNLjSi22nIOyaeyp+12R3UCVWlzIb+nbv8XWXg9YEhkHxYr8xqrr1+hIbFwKNEXnj0esJrKbiW3XGbHsYw==";
            encodeParity1[16] = "BdvaDZCGCVEggrcfscGQWdfSXnCSrOcpD6NfKZGYraK80J2a+v/zkDPWePOQ9k0u0WdWNJ9hQKvPJD0wf2MN+g==";            
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

        [TestMethod()]
        public void CDRepairEncodeSyndromeTest()
        {
            for (int n = 4; n <= AccurateRipVerify.maxNpar; n *= 2)
            {
                Assert.AreEqual<string>(encodeSyndrome[n], ParityToSyndrome.ToBase64String(encode.AR.GetSyndrome(n), 0, 4));
            }
            Assert.AreEqual<uint>(377539636, encode.CRC);
        }

		/// <summary>
		///A test for Write
		///</summary>
		[TestMethod()]
		public void CDRepairEncodeParityTest()
		{
            for (int n = 8; n <= AccurateRipVerify.maxNpar; n *= 2)
            {
                Assert.AreEqual<string>(encodeParity[n], Convert.ToBase64String(encode.AR.GetParity(n), 0, 64));
            }
            Assert.AreEqual<uint>(377539636, encode.CRC);
		}

        /// <summary>
		///Verifying rip that is accurate
		///</summary>
		[TestMethod()]
		public void CDRepairDecodeOriginalTest()
		{
            var decode = generator.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(), encode.CRC, out actualOffset, out hasErrors));
            Assert.IsFalse(hasErrors, "has errors");
            Assert.AreEqual(0, actualOffset, "wrong offset");
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(8), encode.CRC, out actualOffset, out hasErrors));
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
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(), encode.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(-1176, actualOffset, "wrong offset");
            CDRepairFix fix = decode.VerifyParity(encode.AR.GetSyndrome(), actualOffset);
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
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(), encode.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(0, actualOffset, "wrong offset");
            CDRepairFix fix = decode.VerifyParity(encode.AR.GetSyndrome(), actualOffset);
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
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(), encode.CRC, out actualOffset, out hasErrors));
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
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(), encode.CRC, out actualOffset, out hasErrors));
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
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            var syn = encode.AR.GetSyndrome();
            Assert.IsTrue(decode.FindOffset(syn, encode.CRC, out actualOffset, out hasErrors));
            Assert.IsTrue(hasErrors, "doesn't have errors");
            Assert.AreEqual(offset, actualOffset, "wrong offset");
            CDRepairFix fix = decode.VerifyParity(syn, actualOffset);
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
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            Assert.IsTrue(decode.FindOffset(encode.AR.GetSyndrome(), encode.CRC, out actualOffset, out hasErrors), "couldn't find offset");
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(-offset, actualOffset, "wrong offset");
            var fix = decode.VerifyParity(encode.AR.GetSyndrome(), actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
			generator2.Write(fix);
			Assert.AreEqual<uint>(encode.CRC, fix.CRC);

            if (AccurateRipVerify.maxNpar > 8)
            {
                fix = decode.VerifyParity(encode.AR.GetSyndrome(8), actualOffset);
                Assert.IsTrue(fix.HasErrors, "doesn't have errors");
                Assert.IsTrue(fix.CanRecover, "cannot recover");
                generator2.Write(fix);
                Assert.AreEqual<uint>(encode.CRC, fix.CRC);
            }
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
        //[Ignore]
		public void CDRepairSplitTest()
		{
			var seed = 723722;
            var ar0 = new TestImageGenerator("13 68 99 136", seed, 0, 0).CreateCDRepairEncode(stride);
			var splits = new int[] { 1, 13 * 588 - 1, 13 * 588, 13 * 588 + 1, 30 * 588 - 1, 30 * 588, 30 * 588 + 1, 68 * 588 - 1, 68 * 588, 68 * 588 + 1 };
			foreach (int split in splits)
			{
                var ar1 = new TestImageGenerator("13 68 99 136", seed, 0, 0, 0, split).CreateCDRepairEncode(stride);
                var ar2 = new TestImageGenerator("13 68 99 136", seed, 0, 0, split, (int)ar0.FinalSampleCount).CreateCDRepairEncode(stride);
				ar1.AR.Combine(ar2.AR, split, (int)ar0.FinalSampleCount);
                string message = "split = " + CDImageLayout.TimeToString((uint)split / 588) + "." + (split % 588).ToString();
				Assert.AreEqual(ar0.CRC, ar1.CRC, "CRC was not set correctly, " + message);
                CollectionAssert.AreEqual(ar0.AR.GetParity(), ar1.AR.GetParity(), "Parity was not set correctly, " + message);
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
            var ar1 = new TestImageGenerator("13 68 99 136", seed, 0, 0, 0, split).CreateCDRepairEncode(stride);
            var ar2 = new TestImageGenerator("13 68 99 136", seed, 0, 0, split, (int)ar1.FinalSampleCount).CreateCDRepairEncode(stride);
			for (int i = 0; i < 20; i++)
				ar1.AR.Combine(ar2.AR, split, (int)ar1.FinalSampleCount);
		}

		/// <summary>
		///A test for Syndrome2Parity speed
		///</summary>
		[TestMethod()]
		public unsafe void CDRepairSyndrome2ParitySpeedTest()
		{
			byte[] parityCopy = new byte[encode.AR.GetParity().Length];
            var syndrome = encode.AR.GetSyndrome();
			for (int t = 0; t < 100; t++)
                ParityToSyndrome.Syndrome2Parity(syndrome, parityCopy);
			CollectionAssert.AreEqual(encode.AR.GetParity(), parityCopy);
		}

		[TestMethod]
		public unsafe void CDRepairEncodeSynParTest()
		{
            var parityCopy = ParityToSyndrome.Syndrome2Parity(encode.AR.GetSyndrome());
            CollectionAssert.AreEqual(encode.AR.GetParity(), parityCopy);
        }

		[TestMethod]
		public void CDRepairEncodeSpeedTest()
		{
			var generator = new TestImageGenerator("0 75000", seed, 0, 0);
            var encode = generator.CreateCDRepairEncode(stride);
            Assert.AreEqual<string>(encodeParity1[AccurateRipVerify.maxNpar], Convert.ToBase64String(encode.AR.GetParity(), 0, 64), "parity mismatch");
		}

		/// <summary>
		///Verifying rip that has errors
		///</summary>
		[TestMethod()]
		[Ignore]
		public void CDRepairVerifyParitySpeedTest()
		{
			var generator1 = new TestImageGenerator("0 98011", seed, 32 * 588, 0);
            var encode1 = generator1.CreateCDRepairEncode(stride);
			var generator2 = new TestImageGenerator("0 98011", seed, 32 * 588, errors/2);
            var decode = generator2.CreateCDRepairEncode(stride);
			int actualOffset;
			bool hasErrors;
            var syndrome = encode1.AR.GetSyndrome();
            Assert.IsTrue(decode.FindOffset(syndrome, encode1.CRC, out actualOffset, out hasErrors));
			Assert.IsTrue(hasErrors, "doesn't have errors");
			Assert.AreEqual(0, actualOffset, "wrong offset");
			for (int t = 0; t < 1000; t++)
                decode.VerifyParity(syndrome, actualOffset);
            CDRepairFix fix = decode.VerifyParity(syndrome, actualOffset);
			Assert.IsTrue(fix.HasErrors, "doesn't have errors");
			Assert.IsTrue(fix.CanRecover, "cannot recover");
			generator2.Write(fix);
			Assert.AreEqual<uint>(encode1.CRC, fix.CRC);
		}
	}
}
