using System;
using System.Text;
using CUETools.Parity;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for RsEncode16Test and is intended
    ///to contain all RsEncode16Test Unit Tests
    ///</summary>
	[TestClass()]
	public class RsEncode16Test
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
		///A test for encode
		///</summary>
		[TestMethod()]
		public void encodeTest16()
		{
			RsEncode16 target = new RsEncode16(4);
			byte[] data = new byte[1024];
			byte[] expected = new byte[] { 193, 3, 222, 151, 9, 2, 128, 246 };
			byte[] parity = new byte[8];
			new Random(2314).NextBytes(data);
			target.encode(data, 0, data.Length, parity, 0);
			CollectionAssert.AreEqual(expected, parity, "oops");
		}
	}
}
