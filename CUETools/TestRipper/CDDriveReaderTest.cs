using System;
using System.Text;
using System.Collections.Generic;
using CUETools.Codecs;
using CUETools.Ripper;
using CUETools.Ripper.SCSI;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace TestRipper
{
    
    
    /// <summary>
    ///This is a test class for CDDriveReaderTest and is intended
    ///to contain all CDDriveReaderTest Unit Tests
    ///</summary>
	[TestClass()]
	public class CDDriveReaderTest
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

		const int pass = 16;
		const int Sectors2Read = 10000;
		const int bit_weight = 3;
		const int c2_weight = 1;

		bool markErrors = true;
		int _currentStart = 0, _realErrors = 0;
		byte[] _currentData = new byte[Sectors2Read * 4 * 588];

		static long[,] UserData = new long[Sectors2Read, 4 * 588];
		static long[,] C2Data = new long[Sectors2Read, 4 * 588 / 8];
		static byte[] _realData = new byte[Sectors2Read * 4 * 588];
		static long[] byte2long = new long[256];

		#region Additional test attributes
		// 
		//You can use the following additional attributes as you write your tests:
		//
		//Use ClassInitialize to run code before running the first test in the class
		[ClassInitialize()]
		public static void MyClassInitialize(TestContext testContext)
		{
			for (long i = 0; i < 256; i++)
			{
				long bl = 0;
				for (int b = 0; b < 8; b++)
					bl += ((i >> b) & 1) << (b << 3);
				byte2long[i] = bl;
			}

			Random rnd = new Random(2314);
			rnd.NextBytes(_realData);

			for (int p = 0; p <= pass; p++)
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
					for (int iPar = 0; iPar < 4 * 588; iPar++)
					{
						bool error = rnd.NextDouble() < 0.2;
						byte val = error ? (byte)rnd.Next(255) : _realData[iSector * 4 * 588 + iPar];
						UserData[iSector, iPar] += byte2long[val] * bit_weight;
						if (error && rnd.NextDouble() < 0.5)
						{
							C2Data[iSector, iPar >> 3] += (iPar & 7) * 8;
							UserData[iSector, iPar] += 0x0101010101010101 * (bit_weight / 2) + byte2long[val] * (c2_weight - bit_weight);
						}
					}
		}
		//
		//Use ClassCleanup to run code after all tests in a class have run
		//[ClassCleanup()]
		//public static void MyClassCleanup()
		//{
		//}
		//
		//Use TestInitialize to run code before running each test
		[TestInitialize()]
		public void MyTestInitialize()
		{
		}
		//
		//Use TestCleanup to run code after each test has run
		//[TestCleanup()]
		//public void MyTestCleanup()
		//{
		//}
		//
		#endregion


		/// <summary>
		///A test for CorrectSectors
		///</summary>
		[TestMethod()]
		[DeploymentItem("CUETools.Ripper.SCSI.dll")]
		public void CorrectSectorsTest()
		{
			int _currentErrorsCount = 0;
			int sector = 0;

			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int pos = sector - _currentStart + iSector;
				int avg = (pass + 1) * bit_weight / 2;
				int c2_limit = pass / 3; // 
				int er_limit = avg - pass; // allow 33% minority
				for (int iPar = 0; iPar < 4 * 588; iPar++)
				{
					long val = UserData[pos, iPar];
					byte c2 = (byte)(C2Data[pos, iPar >> 3] >> ((iPar & 7) * 8));
					int bestValue = 0;
					for (int i = 0; i < 8; i++)
					{
						int sum = avg - ((int)(val & 0xff));
						int sig = sum >> 31; // bit value
						if ((sum ^ sig) < er_limit) _currentErrorsCount++;
						bestValue += sig & (1 << i);
						val >>= 8;
					}
					//if (c2 > c2_limit)
						//_currentErrorsCount++;
					_currentData[pos * 4 * 588 + iPar] = (byte)bestValue;
				}
			}
			for (int p = 0; p <= pass; p++)
				for (int iSector = 0; iSector < Sectors2Read; iSector++)
					for (int iPar = 0; iPar < 4 * 588; iPar++)
						if (_realData[iSector * 4 * 588 + iPar] != _currentData[iSector * 4 * 588 + iPar])
							_realErrors++;
			Assert.AreEqual<int>(0, _realErrors, "0 != _realErrors; _currentErrorsCount == " + _currentErrorsCount.ToString());
			//CollectionAssert.AreEqual(_realData, _currentData, "_realData != _currentData");
			Assert.AreEqual<int>(0, _currentErrorsCount, "_currentErrorsCount != 0");
		}
	}
}
