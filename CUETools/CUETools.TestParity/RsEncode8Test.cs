using System.Text;
using CUETools.Parity;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestParity
{
    
    
    /// <summary>
    ///This is a test class for RsEncode8Test and is intended
    ///to contain all RsEncode8Test Unit Tests
    ///</summary>
	[TestClass()]
	public class RsEncode8Test
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
		public void encodeTest()
		{
			RsEncode8 target = new RsEncode8(4);
			byte[] data = Encoding.ASCII.GetBytes("my test data");
			byte[] expected = new byte[] { 255, 199, 140, 166 };
			byte[] parity = new byte[4];
			target.encode(data, 0, data.Length, parity, 0);
			CollectionAssert.AreEqual(expected, parity, "oops");
		}
	}
}
