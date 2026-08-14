using CUETools.Codecs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;

namespace CUETools.TestCodecs
{
    [TestClass]
    public class BitReaderBoundsTest
    {
        [TestMethod]
        public unsafe void RiceBlockAcceptsTerminatorAtLogicalEnd()
        {
            // The backing array is intentionally longer than the declared input. Cache
            // lookahead may use zero padding, but the terminator in the second byte is real.
            byte[] backing = new byte[16];
            backing[1] = 0x80;
            int decoded = int.MinValue;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 2);
                reader.read_rice_block(1, 0, &decoded);
            }

            Assert.AreEqual(4, decoded);
        }

        [TestMethod]
        public unsafe void RiceBlockRejectsTerminatorPastLogicalEnd()
        {
            // The sentinel makes an over-read deterministic and memory-safe for the test.
            // It must not terminate a Rice code because it lies outside the declared input.
            byte[] backing = new byte[16];
            backing[2] = 0x80;
            int decoded = int.MinValue;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 2);
                try
                {
                    reader.read_rice_block(1, 0, &decoded);
                    Assert.Fail("The reader accepted a Rice terminator outside the declared input buffer.");
                }
                catch (InvalidDataException)
                {
                }
            }
        }

        [TestMethod]
        public unsafe void RiceBlockFallsBackToCheckedTailAtLogicalEnd()
        {
            // Sixteen declared zero bytes let the optimized Rice loop begin normally.
            // The terminator immediately after them must still be rejected when that
            // loop reaches the logical boundary and continues through the checked tail.
            byte[] backing = new byte[24];
            backing[16] = 0x80;
            int decoded = int.MinValue;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 16);
                try
                {
                    reader.read_rice_block(1, 0, &decoded);
                    Assert.Fail("The reader accepted a Rice terminator after the declared input buffer.");
                }
                catch (InvalidDataException)
                {
                }
            }
        }

        [TestMethod]
        public unsafe void RiceBlockKeepsDecodedPrefixWhenFallingBackToCheckedTail()
        {
            // The first value takes the fast path. The second starts with a whole
            // zero unary byte, which transfers the rest of the block to the checked
            // tail without overwriting or shifting the already decoded prefix.
            byte[] backing = new byte[24];
            backing[0] = 0x80;
            backing[1] = 0x40;
            int* decoded = stackalloc int[2];
            decoded[0] = int.MinValue;
            decoded[1] = int.MinValue;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 21);
                reader.read_rice_block(2, 0, decoded);
            }

            Assert.AreEqual(0, decoded[0]);
            Assert.AreEqual(4, decoded[1]);
        }

        [TestMethod]
        public unsafe void FixedWidthReadRejectsBytesPastLogicalEnd()
        {
            byte[] backing = new byte[16];
            backing[0] = 0xff;
            backing[1] = 0xff;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 1);
                Assert.AreEqual((uint)0xff, reader.readbits(8));
                AssertInvalidData(delegate { reader.readbit(); });
            }
        }

        [TestMethod]
        public unsafe void UnaryReadRejectsTerminatorPastLogicalEnd()
        {
            byte[] backing = new byte[16];
            backing[2] = 0x80;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 2);
                AssertInvalidData(delegate { reader.read_unary(); });
            }
        }

        [TestMethod]
        public unsafe void FlushConsumesDiscardedAlignmentBits()
        {
            byte[] backing = new byte[16];
            backing[0] = 0xff;
            backing[1] = 0xff;

            fixed (byte* input = backing)
            {
                BitReader reader = new BitReader(input, 0, 1);
                Assert.AreEqual((uint)1, reader.readbit());
                reader.flush();
                AssertInvalidData(delegate { reader.readbit(); });
            }
        }

        [TestMethod]
        public void ManagedFlacDecoderRejectsTruncatedFrameAtInputBoundary()
        {
            byte[] source = File.ReadAllBytes(Path.Combine("Data", "test.flac"));
            const int truncatedLength = 216;
            MemoryStream input = new MemoryStream(source, 0, truncatedLength, false, true);
            CUETools.Codecs.Flake.AudioDecoder decoder = null;
            try
            {
                decoder = new CUETools.Codecs.Flake.AudioDecoder(
                    new CUETools.Codecs.Flake.DecoderSettings(),
                    "truncated.flac",
                    input);
                AudioBuffer buffer = new AudioBuffer(decoder, 4096);
                while (decoder.Read(buffer, 4096) != 0)
                {
                }

                Assert.Fail("The managed FLAC decoder accepted a frame cut off inside its Rice payload.");
            }
            catch (InvalidDataException ex)
            {
                StringAssert.Contains(ex.Message, "Rice code exceeds the input buffer");
            }
            finally
            {
                if (decoder != null)
                    decoder.Close();
                else
                    input.Dispose();
            }
        }

        private static void AssertInvalidData(System.Action action)
        {
            try
            {
                action();
                Assert.Fail("The reader accepted bits outside the declared input buffer.");
            }
            catch (InvalidDataException)
            {
            }
        }
    }
}
