using System;
using System.Text;
using System.IO;
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

		const int max_pass = 64;
		const int Sectors2Read = 2400;

		bool markErrors = true;
		int _currentStart = 0, _realErrors = 0;
		byte[] _currentData = new byte[Sectors2Read * 4 * 588];

		static ulong[,,] UserData = new ulong[Sectors2Read, 2, 4 * 588];
		static byte[,] C2Count = new byte[Sectors2Read, 4 * 588];
		static byte[] _realData = new byte[Sectors2Read * 4 * 588];
		static ulong[] byte2long = new ulong[256];

		#region Additional test attributes
		// 
		//You can use the following additional attributes as you write your tests:
		//
		//Use ClassInitialize to run code before running the first test in the class
		[ClassInitialize()]
		public static void MyClassInitialize(TestContext testContext)
		{
			for (ulong i = 0; i < 256; i++)
			{
				ulong bl = 0;
				for (int b = 0; b < 8; b++)
					bl += ((i >> b) & 1) << (b << 3);
				byte2long[i] = bl;
			}

			//Random rnd = new Random(2314);
			//rnd.NextBytes(_realData);

			byte [] c2data = new byte[Sectors2Read * 296];
			for (int p = 0; p < max_pass; p++)
			{
				//    string nm_d = string.Format("Y:\\Temp\\dbg\\{0:x}-{1:00}.bin", _currentStart, dbg_pass);
				using (FileStream fs = new FileStream(string.Format("Y:\\Temp\\dbg\\960\\960-{0:00}.bin", p), FileMode.Open))
				using (FileStream fs2 = new FileStream(string.Format("Y:\\Temp\\dbg\\960\\960-{0:00}.c2", p), FileMode.Open))
				{
					fs.Read(_realData, 0, Sectors2Read * 4 * 588);
					fs2.Read(c2data, 0, Sectors2Read * 296);
					for (int iSector = 0; iSector < Sectors2Read; iSector++)
					{
						for (int pos = 0; pos < 294; pos++)
						{
							int c2d = c2data[iSector * 296 + pos];
							for (int sample = (pos << 3); sample < (pos << 3) + 8; sample++)
							{
								//int c2 = (c2d >> (7 - (sample & 7))) & 1;
								//int c2 = (c2d >> ((sample & 7))) & 1;
								//int c2 = ((c2d >> ((sample & 7))) | (c2d >> (7 - (sample & 7)))) & 1;
								int c2 = ((-c2d) >> 31) & 1;
								C2Count[iSector, sample] += (byte)c2;
								UserData[iSector, c2, sample] += byte2long[_realData[iSector * 4 * 588 + sample]];
							}
						}
					}
				}
				//for (int iSector = 0; iSector < Sectors2Read; iSector++)
				//    for (int iPar = 0; iPar < 4 * 588; iPar++)
				//    {
				//        bool error = rnd.NextDouble() < 0.2;
				//        byte val = error ? (byte)rnd.Next(255) : _realData[iSector * 4 * 588 + iPar];
				//        UserData[iSector, iPar] += byte2long[val] * bit_weight;
				//        if (error && rnd.NextDouble() < 0.5)
				//        {
				//            C2Data[iSector, iPar >> 3] += (iPar & 7) * 8;
				//            UserData[iSector, iPar] += 0x0101010101010101 * (bit_weight / 2) + byte2long[val] * (c2_weight - bit_weight);
				//        }
				//    }
			}
			using (FileStream fs = new FileStream(string.Format("Y:\\Temp\\dbg\\960\\960.bin", 0), FileMode.Open))
				fs.Read(_realData, 0, Sectors2Read * 4 * 588);
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
			_realErrors = 0;
			const byte c2div = 128;
			const int er_limit = c2div * 3;
			int fErrCnt = 0;
			for (int iSector = 0; iSector < Sectors2Read; iSector++)
			{
				int pos = sector - _currentStart + iSector;
				for (int iPar = 0; iPar < 4 * 588; iPar++)
				{
					ulong val = UserData[pos, 0, iPar];
					ulong val1 = 0;// UserData[pos, 1, iPar];
					byte c2 = C2Count[pos, iPar];
					int ave = (max_pass - c2) * c2div + c2;
					int bestValue = 0;
					bool fError = false;
					for (int i = 0; i < 8; i++)
					{
						int sum = ave - 2 * (int)((val & 0xff) * c2div + (val1 & 0xff));
						int sig = sum >> 31;
						fError |= (sum ^ sig) < er_limit;
						bestValue += sig & (1 << i);
						val >>= 8;
					}
					if (fError)
						fErrCnt++;
					//if (c2 > c2_limit)
						//_currentErrorsCount++;
					_currentData[pos * 4 * 588 + iPar] = (byte)bestValue;
					if (_realData[iSector * 4 * 588 + iPar] != bestValue)
						_realErrors++;
				}
			}
			//Assert.AreEqual<int>(0, fErrCnt);
			Assert.AreEqual<int>(0, _realErrors, "0 != _realErrors; _currentErrorsCount == " + _currentErrorsCount.ToString());
			//CollectionAssert.AreEqual(_realData, _currentData, "_realData != _currentData");
			Assert.AreEqual<int>(0, _currentErrorsCount, "_currentErrorsCount != 0");
		}
	}
}
