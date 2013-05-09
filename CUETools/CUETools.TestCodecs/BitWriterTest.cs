using CUETools.Codecs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CUETools.TestCodecs
{
    
    
    /// <summary>
    ///This is a test class for BitWriterTest and is intended
    ///to contain all BitWriterTest Unit Tests
    ///</summary>
    [TestClass()]
    public class BitWriterTest
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
        ///A test for writebits
        ///</summary>
        [TestMethod()]
        public void writebitsTest()
        {
            byte[] buf = new byte[0x10000];
            for (int seed = 0; seed < 10; seed++)
            {
                BitWriter target = new BitWriter(buf, 0, buf.Length);
                var rnd = new Random(seed);
                int count = 0;
                do
                {
                    int bits = rnd.Next(0, 64) + 1;
                    ulong val = (1U << (bits - 1));
                    target.writebits(bits, val);
                    count++;
                }
                while (target.Length < buf.Length - 32);
                target.flush();
                rnd = new Random(seed);
                unsafe
                {
                    fixed (byte* ptr = buf)
                    {
                        BitReader reader = new BitReader(ptr, 0, buf.Length);
                        for (int i = 0; i < count; i++)
                        {
                            int bits = rnd.Next(0, 64) + 1;
                            ulong val = (1U << (bits - 1));
                            ulong val1 = reader.readbits64(bits);
                            Assert.AreEqual(val, val1, string.Format("i = {0}, bits = {1}, seed = {2}, pos = {3}", i, bits, seed, reader.Position));
                        }
                    }
                }
            }
        }
    }
}
