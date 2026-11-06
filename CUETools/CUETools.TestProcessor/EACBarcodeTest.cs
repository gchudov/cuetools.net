using System.Collections.Generic;
using CUETools.CTDB;
using CUETools.CTDB.EACPlugin;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUETools.TestProcessor
{
    [TestClass]
    public class EACBarcodeTest
    {
        [TestMethod]
        public void SanitizesBarcodeAndPreservesLeadingZeroes()
        {
            Assert.AreEqual("0012345", BarcodePrompt.Sanitize(" 00-12\r\n345x"));
        }

        [TestMethod]
        public void EmptySanitizedBarcodeIsEmpty()
        {
            Assert.AreEqual("", BarcodePrompt.Sanitize("abc \r\n"));
        }

        [TestMethod]
        public void CancelledPromptReturnsNoBarcode()
        {
            Assert.AreEqual("", BarcodePrompt.GetBarcode(System.Windows.Forms.DialogResult.Cancel, "00123"));
        }

        [TestMethod]
        public void MatchingResultsPreserveOrderAndIgnoreMissingBarcodes()
        {
            var first = new CTDBResponseMeta { barcode = "00123" };
            var missing = new CTDBResponseMeta();
            var match = new CTDBResponseMeta { barcode = "00123" };
            var results = new List<CTDBResponseMeta> { first, missing, match };
            var filtered = BarcodeMetadataFilter.FilterOrOriginal(results, "00123");
            Assert.AreEqual(2, filtered.Count);
            Assert.AreSame(first, filtered[0]);
            Assert.AreSame(match, filtered[1]);
            Assert.AreEqual(3, results.Count);
        }

        [TestMethod]
        public void NoMatchAndNoBarcodeUseTheCompleteOriginalOrder()
        {
            var first = new CTDBResponseMeta { barcode = "123" };
            var second = new CTDBResponseMeta();
            var results = new List<CTDBResponseMeta> { first, second };
            var unmatched = BarcodeMetadataFilter.FilterOrOriginal(results, "999");
            var unspecified = BarcodeMetadataFilter.FilterOrOriginal(results, null);
            Assert.AreEqual(2, unmatched.Count);
            Assert.AreSame(first, unmatched[0]);
            Assert.AreSame(second, unmatched[1]);
            Assert.AreEqual(2, unspecified.Count);
            Assert.AreSame(first, unspecified[0]);
            Assert.AreSame(second, unspecified[1]);
        }

        [TestMethod]
        public void SingleMatchIsAutomaticAndMultipleMatchesRemainSelectable()
        {
            var one = new List<CTDBResponseMeta> { new CTDBResponseMeta { barcode = "001" } };
            var two = new List<CTDBResponseMeta> { one[0], new CTDBResponseMeta { barcode = "001" } };
            Assert.IsTrue(BarcodeMetadataFilter.HasSingleMatch(one, "001"));
            Assert.IsFalse(BarcodeMetadataFilter.HasSingleMatch(two, "001"));
            Assert.IsFalse(BarcodeMetadataFilter.HasSingleMatch(one, null));
        }

        [TestMethod]
        public void PromptOptionPersistsAndDefaultsToDisabled()
        {
            bool original = AudioDataPlugIn.Options.PromptForBarcode;
            try
            {
                AudioDataPlugIn.Options.PromptForBarcode = false;
                Assert.IsFalse(AudioDataPlugIn.Options.PromptForBarcode);
                AudioDataPlugIn.Options.PromptForBarcode = true;
                Assert.IsTrue(AudioDataPlugIn.Options.PromptForBarcode);
            }
            finally
            {
                AudioDataPlugIn.Options.PromptForBarcode = original;
            }
        }
    }
}
