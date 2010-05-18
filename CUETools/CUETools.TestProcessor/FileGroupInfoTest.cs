using CUETools.Processor;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUETools.TestProcessor
{
    
    
    /// <summary>
    ///This is a test class for FileGroupInfoTest and is intended
    ///to contain all FileGroupInfoTest Unit Tests
    ///</summary>
	[TestClass()]
	public class FileGroupInfoTest
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
		///A test for CompareTrackNames
		///</summary>
		[TestMethod()]
		public void CompareTrackNamesTest()
		{
			Assert.AreEqual(0, FileGroupInfo.CompareTrackNames("abc", "abc"));
			Assert.AreEqual(0, FileGroupInfo.CompareTrackNames(" 1.abc", "1.abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("0 abc", "1 abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("0.abc", "1.abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("1 abc", "1.1 abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("1.1 abc", "1.10 abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("1.10 abc", "2.1 abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("2.1.abc", "10.abc"));
			Assert.AreEqual(-1, FileGroupInfo.CompareTrackNames("Track 2a", "Track 10"));
		}
	}
}
