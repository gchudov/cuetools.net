using System;
using System.Text;
using CUETools.Parity;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for RsDecodeTest and is intended
    ///to contain all RsDecodeTest Unit Tests
    ///</summary>
	[TestClass()]
	public class RsDecodeTest
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
		public void decodeTest()
		{
			RsDecode8 target = new RsDecode8(4);
			byte[] data = Encoding.ASCII.GetBytes("my zest dada");
			byte[] parity = new byte[] { 255, 199, 140, 166 };
			byte[] dataplus = new byte[data.Length + 4];
			Array.Copy(data, 0, dataplus, 0, data.Length);
			Array.Copy(parity, 0, dataplus, data.Length, parity.Length);
			int errors;
			bool ok = target.decode(dataplus, dataplus.Length, false, out errors);
			Assert.AreEqual(true, ok, "Fail");
			Assert.AreEqual(2, errors, "Errors");
			Assert.AreEqual("my test data", Encoding.ASCII.GetString(dataplus, 0, data.Length));
		}
	}
}
