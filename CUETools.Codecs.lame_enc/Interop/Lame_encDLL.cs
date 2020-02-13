using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.LAME.Interop
{
    /// <summary>
    /// Lame_enc DLL functions
    /// </summary>
    public class Lame_encDll
    {
        //Error codes
        public const uint BE_ERR_SUCCESSFUL = 0;
        public const uint BE_ERR_INVALID_FORMAT = 1;
        public const uint BE_ERR_INVALID_FORMAT_PARAMETERS = 2;
        public const uint BE_ERR_NO_MORE_HANDLES = 3;
        public const uint BE_ERR_INVALID_HANDLE = 4;

        /// <summary>
        /// This function is the first to call before starting an encoding stream.
        /// </summary>
        /// <param name="pbeConfig">Encoder settings</param>
        /// <param name="dwSamples">Receives the number of samples (not bytes, each sample is a SHORT) to send to each beEncodeChunk() on return.</param>
        /// <param name="dwBufferSize">Receives the minimum number of bytes that must have the output(result) buffer</param>
        /// <param name="phbeStream">Receives the stream handle on return</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        [DllImport("Lame_enc.dll")]
        public static extern uint beInitStream(BE_CONFIG pbeConfig, ref uint dwSamples, ref uint dwBufferSize, ref uint phbeStream);

        /// <summary>
        /// Encodes a chunk of samples. Please note that if you have set the output to 
        /// generate mono MP3 files you must feed beEncodeChunk() with mono samples
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <param name="nSamples">Number of samples to be encoded for this call. 
        /// This should be identical to what is returned by beInitStream(), 
        /// unless you are encoding the last chunk, which might be smaller.</param>
        /// <param name="pInSamples">Array of 16-bit signed samples to be encoded. 
        /// These should be in stereo when encoding a stereo MP3 
        /// and mono when encoding a mono MP3</param>
        /// <param name="pOutput">Buffer where to write the encoded data. 
        /// This buffer should be at least of the minimum size returned by beInitStream().</param>
        /// <param name="pdwOutput">Returns the number of bytes of encoded data written. 
        /// The amount of data written might vary from chunk to chunk</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        [DllImport("Lame_enc.dll")]
        public static extern uint beEncodeChunk(uint hbeStream, uint nSamples, short[] pInSamples, [In, Out] byte[] pOutput, ref uint pdwOutput);

        /// <summary>
        /// Encodes a chunk of samples. Please note that if you have set the output to 
        /// generate mono MP3 files you must feed beEncodeChunk() with mono samples
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <param name="nSamples">Number of samples to be encoded for this call. 
        /// This should be identical to what is returned by beInitStream(), 
        /// unless you are encoding the last chunk, which might be smaller.</param>
        /// <param name="pSamples">Pointer at the 16-bit signed samples to be encoded. 
        /// InPtr is used to pass any type of array without need of make memory copy, 
        /// then gaining in performance. Note that nSamples is not the number of bytes,
        /// but samples (is sample is a SHORT)</param>
        /// <param name="pOutput">Buffer where to write the encoded data. 
        /// This buffer should be at least of the minimum size returned by beInitStream().</param>
        /// <param name="pdwOutput">Returns the number of bytes of encoded data written. 
        /// The amount of data written might vary from chunk to chunk</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        [DllImport("Lame_enc.dll")]
        protected static extern uint beEncodeChunk(uint hbeStream, uint nSamples, IntPtr pSamples, IntPtr pOutput, ref uint pdwOutput);

        /// <summary>
        /// Encodes a chunk of samples. Samples are contained in a byte array
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <param name="buffer">Bytes to encode</param>
        /// <param name="index">Position of the first byte to encode</param>
        /// <param name="nBytes">Number of bytes to encode (not samples, samples are two byte length)</param>
        /// <param name="pOutput">Buffer where to write the encoded data.
        /// This buffer should be at least of the minimum size returned by beInitStream().</param>
        /// <param name="pdwOutput">Returns the number of bytes of encoded data written. 
        /// The amount of data written might vary from chunk to chunk</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        public static unsafe uint EncodeChunk(uint hbeStream, byte* pSamples, uint nBytes, byte* pOutput, ref uint pdwOutput)
        {
            return beEncodeChunk(hbeStream, nBytes / 2/*Samples*/, (IntPtr)pSamples, (IntPtr)pOutput, ref pdwOutput);
        }

        /// <summary>
        /// Encodes a chunk of samples. Samples are contained in a byte array
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <param name="buffer">Bytes to encode</param>
        /// <param name="index">Position of the first byte to encode</param>
        /// <param name="nBytes">Number of bytes to encode (not samples, samples are two byte length)</param>
        /// <param name="pOutput">Buffer where to write the encoded data.
        /// This buffer should be at least of the minimum size returned by beInitStream().</param>
        /// <param name="pdwOutput">Returns the number of bytes of encoded data written. 
        /// The amount of data written might vary from chunk to chunk</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        public static unsafe uint EncodeChunk(uint hbeStream, byte[] Samples, int index, uint nBytes, byte[] Output, ref uint pdwOutput)
        {
            fixed (byte* pSamples = &Samples[index], pOutput = Output)
                return beEncodeChunk(hbeStream, nBytes / 2/*Samples*/, (IntPtr)pSamples, (IntPtr)pOutput, ref pdwOutput);
        }

        /// <summary>
        /// Encodes a chunk of samples. Samples are contained in a byte array
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <param name="buffer">Bytes to encode</param>
        /// <param name="pOutput">Buffer where to write the encoded data.
        /// This buffer should be at least of the minimum size returned by beInitStream().</param>
        /// <param name="pdwOutput">Returns the number of bytes of encoded data written. 
        /// The amount of data written might vary from chunk to chunk</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        public static uint EncodeChunk(uint hbeStream, byte[] buffer, byte[] pOutput, ref uint pdwOutput)
        {
            return EncodeChunk(hbeStream, buffer, 0, (uint)buffer.Length, pOutput, ref pdwOutput);
        }

        /// <summary>
        /// This function should be called after encoding the last chunk in order to flush 
        /// the encoder. It writes any encoded data that still might be left inside the 
        /// encoder to the output buffer. This function should NOT be called unless 
        /// you have encoded all of the chunks in your stream.
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <param name="pOutput">Where to write the encoded data. This buffer should be 
        /// at least of the minimum size returned by beInitStream().</param>
        /// <param name="pdwOutput">Returns number of bytes of encoded data written.</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        [DllImport("Lame_enc.dll")]
        public static extern uint beDeinitStream(uint hbeStream, [In, Out] byte[] pOutput, ref uint pdwOutput);

        /// <summary>
        /// Last function to be called when finished encoding a stream. 
        /// Should unlike beDeinitStream() also be called if the encoding is canceled.
        /// </summary>
        /// <param name="hbeStream">Handle of the stream.</param>
        /// <returns>On success: BE_ERR_SUCCESSFUL</returns>
        [DllImport("Lame_enc.dll")]
        public static extern uint beCloseStream(uint hbeStream);

        /// <summary>
        /// Returns information like version numbers (both of the DLL and encoding engine), 
        /// release date and URL for lame_enc's homepage. 
        /// All this information should be made available to the user of your product 
        /// through a dialog box or something similar.
        /// </summary>
        /// <param name="pbeVersion"Where version number, release date and URL for homepage 
        /// is returned.</param>
        [DllImport("Lame_enc.dll")]
        public static extern void beVersion([Out] BE_VERSION pbeVersion);

        [DllImport("Lame_enc.dll", CharSet = CharSet.Ansi)]
        public static extern void beWriteVBRHeader(string pszMP3FileName);
        [DllImport("Lame_enc.dll")]
        public static extern uint beEncodeChunkFloatS16NI(uint hbeStream, uint nSamples, [In]float[] buffer_l, [In]float[] buffer_r, [In, Out]byte[] pOutput, ref uint pdwOutput);
        [DllImport("Lame_enc.dll")]
        public static extern uint beFlushNoGap(uint hbeStream, [In, Out]byte[] pOutput, ref uint pdwOutput);
        [DllImport("Lame_enc.dll", CharSet = CharSet.Ansi)]
        public static extern uint beWriteInfoTag(uint hbeStream, string lpszFileName);
    }
}
