using System;
using CUETools.Parity;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for RsDecode16Test and is intended
    ///to contain all RsDecode16Test Unit Tests
    ///</summary>
	[TestClass()]
	public class RsDecode16Test
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
		///A test for decode
		///</summary>
		[TestMethod()]
		public void decodeTest16()
		{
			RsDecode16 target = new RsDecode16(4);
			byte[] data = new byte[1024];
			byte[] parity = new byte[] { 193, 3, 222, 151, 9, 2, 128, 246 };
			byte[] dataplus = new byte[data.Length + parity.Length];
			Random rnd = new Random(2314);
			rnd.NextBytes(data);
			Array.Copy(data, 0, dataplus, 0, data.Length);
			Array.Copy(parity, 0, dataplus, data.Length, parity.Length);

			for (int i = 0; i < 1000; i++)
			{
				int pos1 = (int)(rnd.NextDouble() * 1023);
				int pos2 = (int)(rnd.NextDouble() * 1023);
				if (Math.Abs(pos1 - pos2) < 4)
					pos2 = (pos1 + 4) % 1024;
				dataplus[pos1] = (byte)(rnd.NextDouble() * 255);
				dataplus[pos2] = (byte)(rnd.NextDouble() * 255);
				if (data[pos1] == dataplus[pos1]) dataplus[pos1] ^= 1;
				if (data[pos2] == dataplus[pos2]) dataplus[pos2] ^= 8;
				
				//dataplus[(int)(rnd.NextDouble() * 1023)] = (byte)(rnd.NextDouble() * 255);

				int errors;
				bool ok = target.decode(dataplus, 0, dataplus.Length, false, out errors);
				byte[] fixed_data = new byte[data.Length];
				Array.Copy(dataplus, fixed_data, data.Length);

				Assert.AreEqual(true, ok, "Fail");
				Assert.AreEqual(2, errors, "Errors");
				CollectionAssert.AreEqual(data, fixed_data);
			}
		}
	}
}
