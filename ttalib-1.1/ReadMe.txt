/////////////////////////////////////////////////////////////////////////////
////////////////// THE LOSSLESS TRUE AUDIO CODEC LIBRARY ////////////////////
/////////////////////////////////////////////////////////////////////////////

This package contains a full-futured CODEC library for realtime encoding and
decoding of True Audio (TTA) files.

The library has a 3 interface classes and 2 functions, which provides a
possibility to work in applications of any complexity.

For maintenance of namespace wholeness, all functions and library classes
are transferred out to the separate namespace, named TTALib.

For a simplicity of work with library namespace it's possible to use a
command:

using namespace TTALib;

/////////////////////////////////////////////////////////////////////////////
/////////////////// Wav2TTA, TTA2Wav & TTATest functions ////////////////////
/////////////////////////////////////////////////////////////////////////////

For a simple use of the library we provide 3 basic functions:

TTAError Wav2TTA ( const char *infile,
                   const char *outfile, 
                   bool CopyID3v2Tag = true,
                   TTACALLBACK TTACallback = NULL,
                   void *uParam = NULL );

The Wav2TTA function is intended for convert a WAV files into the TTA format.

Description of the function parameters:

        infile       - the name of the input WAV file;
        outfile      - the name of the output TTA file;
        CopyID3v2Tag - copy ID3v2 tag if it's present;
	TTACallback  - bool (*TTACALLBACK)(const TTAStat &stat, void *uParam),
                       the callback function, intended for the extension of
                       possibilities of the user program. Can be used to get
                       a statistics of the encoding process. This parameter
                       must be set to NULL if not used. The callback function
                       must return 'true' to continue of the encoding process,
                       or 'false' to interrupt it. In this case the Wav2TTA
                       function will return the TTA_CANCELED error.
	uParam       - users parameter, can be set to NULL;

The TTA2Wav function is intended for convert a TTA files into the Wav format.

TTAError TTA2Wav ( const char *infile,
                   const char *outfile,
                   bool CopyID3v2Tag = true,
                   TTACALLBACK TTACallback = NULL,
                   void *uParam = NULL );

Description of the function parameters:

        infile       - the name of the input TTA file;
        outfile      - the name of the output WAV file;
        CopyID3v2Tag - copy ID3v2 tag if it's present;
	TTACallback  - bool (*TTACALLBACK)(const TTAStat &stat, void *uParam),
                       the callback function, intended for the extension of
                       possibilities of the user program. Can be used to get
                       a statistics of the decoding process. This parameter
                       must be set to NULL if not used. The callback function
                       must return 'true' to continue of the decoding process,
                       or 'false' to interrupt it. In this case the TTA2Wav
                       function will return the TTA_CANCELED error.
	uParam       - users parameter, can be set to NULL;

The TTA2Test function is intended for test a TTA files for errors.

TTAError TTATest ( const char *infile,
		   TTACALLBACK TTACallback = NULL,
		   void *uParam = NULL );

Description of the function parameters:

        infile       - the name of the input TTA file;
	TTACallback  - bool (*TTACALLBACK)(const TTAStat &stat, void *uParam),
                       the callback function, intended for the extension of
                       possibilities of the user program. Can be used to get
                       a statistics of the testing process. This parameter
                       must be set to NULL if not used. The callback function
                       must return 'true' to continue of the testing process,
                       or 'false' to interrupt it. In this case the TTA2Wav
                       function will return the TTA_CANCELED error.
	uParam       - users parameter, can be set to NULL;

All of these functions returns the values of the TTAError type.
The error code can be easily converted into the text string, by calling the
GetErrStr() function, accepting a error code as a parameter.

/////////////////////////////////////////////////////////////////////////////
//////////////// TTAEncoder, TTADecoder & WavFile classes ///////////////////
/////////////////////////////////////////////////////////////////////////////

For using this library in advanced applications the TTAEncoder, TTADecoder
and the WavFile interface classes can be used.

============================================================================
                             TTAEncoder class
============================================================================

The TTAEncoder class is intended for coding PCM data with into the TTA file.

The TTAEncoder class has a 2 constructors:

TTAEncoder( const char *filename, 
            bool append,
            unsigned short AudioFormat, 
            unsigned short NumChannels, 
            unsigned short BitsPerSample,
            unsigned long SampleRate,
            unsigned long DataLength );

TTAEncoder( HANDLE hInFile,
            bool append,
            unsigned short AudioFormat, 
            unsigned short NumChannels, 
            unsigned short BitsPerSample,
            unsigned long SampleRate,
            unsigned long DataLength );

The first of these 2 constructors accepts the name of the input file,
secondary constructor accepts the input file handle. The 'append' parameter
can be used to open input file for writing at the end of the file
(appending); creates the file first if it doesn't exist.

The other parameters contains the information about the stream:

        AudioFormat   - audio format:
                        WAVE_FORMAT_PCM or WAVE_FORMAT_IEEE_FLOAT;
        NumChannels   - number of channels;
        BitsPerSample - count of bits per sample;
        SampleRate    - sample rate;
        DataLength    - overall number of input samples in file.

The CompressBlock function can be used to compress a chunk of the input data:

        bool CompressBlock (long *buffer, long bufLen);

Description of the function parameters:

        buffer - input data buffer;
        bufLen - buffer length (number of input samples).

Returns 'true' on success and 'false' on failure.

The GetStat() function allows to get a compression statistics.

        TTAStat GetStat();

This function returns the TTAStat strucure:

        struct TTAStat
        {
                double ratio;
                unsigned long input_bytes;
                unsigned long output_bytes;
        };

Description of the structure fields:

        ratio - compression ratio;
        input_bytes - count of input bytes processed;
        input_bytes - count of output bytes saved.

============================================================================
                             TTADecoder class
============================================================================

The TTADecoder class is intended for decoding of the TTA audio files.

The TTADecoder class has a 2 constructors:

        TTADecoder (const char *filename);
        TTADecoder (HANDLE hInFile);

The first of these 2 constructors accepts the name of the input file,
secondary constructor accepts the input file handle.

The next set of functions can be used to retrieve the information about the
stream:

        GetAudioFormat()   - returns the audio format:
                             WAVE_FORMAT_PCM or WAVE_FORMAT_IEEE_FLOAT;
        GetNumChannels()   - returns the number of channels;
        GetBitsPerSample() - returns the count of bits per sample;
        GetSampleRate()    - returns the sample rate;
        GetDataLength()    - returns the overall number of input samples
                             in file.

The CompressBlock function can be used to get a chunk of the decompressed
data.

        long GetBlock (long **buffer);

Returns the decompressed data into the 'buffer' and the buffer length (count
of samples) as a function value. Returns 0 if the end-of-file is reached.

The GetStat() function allows to get a decompression statistics.

        TTAStat GetStat();

This function returns the the structure which has a TTAStat type, described
above.

If an error occurs, the both of TTAEncoder and TTADecoder classes generates
exceptions of the TTAException type. The error code can be retrieved by
GetErrNo function of the exception value. The returned error code can be
converted into the text string by the GetErrStr() function, which accepts the
error value as a parameter.

============================================================================
                              WavFile class
============================================================================

The WavFile class provides a simple interface to work with a WAV format files.
The error code can be retrieved by GetErrNo function. The returned error code
can be converted into the text string by the GetErrStr() function, which
accepts the error value as a parameter.

        HANDLE Create(const char *filename);

Creates the WAV file with a name, specified by the 'filename' parameter.
Returns the file HANDLE or INVALID_HANDLE_VALUE on error.

        HANDLE Open(const char *filename);

Opens WAV file to read. Function accepts the file name as a parameter and
returns the file HANDLE or INVALID_HANDLE_VALUE on error.

        bool ReadHeaders();

This function is intended to get WAV file headers. Then, the read parameters
can be retrieved from class attributes 'wave_hdr' and 'subchunk_hdr'.
Function returns 'true' on success and 'false' on failure.

        bool Read(long *data, long byte_size, unsigned long *len);

The function reads 'len' samples of data, each size 'byte_size' bytes, from
the input file into the buffer 'data'. Function returns 'true' on success
and 'false' on failure.

        bool WriteHeaders();

This function is intended to write WAV file headers. The writing headers
will be retrieved from class attributes 'wave_hdr' and 'subchunk_hdr'.
Function returns 'true' on success and 'false' on failure.

        bool Write(long *data, long byte_size, long num_chan,
                   unsigned long *len);

The function writes 'len' samples of data, each size 'byte_size' bytes,
into the output file from the buffer 'data'. The 'num_chan' parameter
defines the number of audio channels. Function returns 'true' on success
and 'false' on failure. The actually wrote number of samples will be
returned back by the 'len' parameter.

        void Close();    // Closes the WAV file

        TTAError GetErrNo() const;    // Returns the error code.

The returned error code can be converted into the text string by the
GetErrStr() function, which accepts the error value as a parameter.

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

Copyright (c) 2004 Alexander Djourik. All rights reserved.
Copyright (c) 2004 Pavel Zhilin. All rights reserved.

For the latest in news and downloads, please visit the official True Audio
project site: http://tta.sourceforge.net

