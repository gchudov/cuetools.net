﻿using CUETools.Codecs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestCodecs
{
    
    
    /// <summary>
    ///This is a test class for Crc32Test and is intended
    ///to contain all Crc32Test Unit Tests
    ///</summary>
	[TestClass()]
	public class Crc32Test
	{

		private byte[] testBytes = new byte[] { 0, 0, 1, 0, 254, 255, 253, 255, 255, 127, 254, 127, 3, 128, 4, 128 };

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
		//[ClassInitialize()]
		//public static void MyClassInitialize(TestContext testContext)
		//{
		//}
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
		///A test for Combine
		///</summary>
		[TestMethod()]
		public void CombineTest()
		{
			int lenAB = testBytes.Length;
			int lenA = 7;
			int lenB = lenAB - lenA;
			uint crcAB = Crc32.ComputeChecksum(0, testBytes, 0, lenAB);
			uint crcA = Crc32.ComputeChecksum(0, testBytes, 0, lenA);
			uint crcB = Crc32.ComputeChecksum(0, testBytes, lenA, lenB);
			Assert.AreEqual<uint>(crcAB, Crc32.Combine(crcA, crcB, lenB), "CRC32 was not combined correctly.");
			Assert.AreEqual<uint>(crcB, Crc32.Combine(crcA, crcAB, lenB), "CRC32 was not subtracted correctly.");
			Assert.AreEqual<uint>(crcA, Crc32.Subtract(crcAB, crcB, lenB), "CRC32 was not subtracted correctly.");
		}

        /// <summary>
        ///A test for Combine
        ///</summary>
        [TestMethod()]
        public void CombineTest16()
        {
            int lenAB = testBytes.Length;
            int lenA = 7;
            int lenB = lenAB - lenA;
            ushort crcAB = Crc16.ComputeChecksum(0, testBytes, 0, lenAB);
            ushort crcA = Crc16.ComputeChecksum(0, testBytes, 0, lenA);
            ushort crcB = Crc16.ComputeChecksum(0, testBytes, lenA, lenB);
            Assert.AreEqual<uint>(crcAB, Crc16.Combine(crcA, crcB, lenB), "CRC16 was not combined correctly.");
            Assert.AreEqual<uint>(crcB, Crc16.Combine(crcA, crcAB, lenB), "CRC16 was not subtracted correctly.");
            Assert.AreEqual<uint>(crcA, Crc16.Subtract(crcAB, crcB, lenB), "CRC16 was not subtracted correctly.");
        }

		/// <summary>
		///A test for ComputeChecksum
		///</summary>
		[TestMethod()]
		public void ComputeChecksumTest()
		{
			uint actual = Crc32.ComputeChecksum(0xffffffff, testBytes, 0, testBytes.Length) ^ 0xffffffff;
			Assert.AreEqual<uint>(2028688632, actual, "CRC32 was not combined correctly.");
		}
	}
}
