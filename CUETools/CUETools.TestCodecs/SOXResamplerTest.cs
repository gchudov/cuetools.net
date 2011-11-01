using System;
using CUETools.Codecs;
using CUETools.DSP.Resampler;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUETools.TestCodecs
{
    /// <summary>
    ///This is a test class for SOXResamplerTest and is intended
    ///to contain all SOXResamplerTest Unit Tests
    ///</summary>
	[TestClass()]
	public class SOXResamplerTest
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
		///A test for SOXResampler Constructor
		///</summary>
		[TestMethod()]
		public void SOXResamplerConstructorTest()
		{
			AudioPCMConfig inputPCM = new AudioPCMConfig(32, 1, 44100);
			AudioPCMConfig outputPCM = new AudioPCMConfig(32, 1, 48000);
			SOXResamplerConfig cfg;
			cfg.Quality = SOXResamplerQuality.Very;
			cfg.Phase = 50;
			cfg.AllowAliasing = false;
			cfg.Bandwidth = 0;
			SOXResampler resampler = new SOXResampler(inputPCM, outputPCM, cfg);
			AudioBuffer src = new AudioBuffer(inputPCM, 400 * inputPCM.SampleRate / 1000);
			AudioBuffer dst = new AudioBuffer(outputPCM, src.Size * 3);
			int offs = 0;
			double delta = 0;
			for (int i = 0; i < 100; i++)
			{
				src.Prepare(-1);
				for (int j = 0; j < src.Size; j++)
					src.Float[j, 0] = (float)Math.Sin((i * src.Size + j) * Math.PI / 44100);
				src.Length = src.Size;
				resampler.Flow(src, dst);
				for (int j = 0; j < dst.Length; j++)
					delta += dst.Float[j, 0] - Math.Sin((offs + j) * Math.PI / 48000);
				offs += dst.Length;
			}
			Assert.IsTrue(Math.Abs(delta) < 0.00001, "Error too large");
		}
	}
}
