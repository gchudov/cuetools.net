/*****************************************************************************************
Monkey's Audio MACDll.h (include for using MACDll.dll in your projects)
Copyright (C) 2000-2011 by Matthew T. Ashland   All Rights Reserved.

Overview:

Basically all this dll does is wrap MACLib.lib, so browse through MACLib.h for documentation
on how to use the interfaces.
*****************************************************************************************/

#pragma once

/*****************************************************************************************
Includes
*****************************************************************************************/
#include "All.h"
#include "MACLib.h"

/*****************************************************************************************
Helper functions
*****************************************************************************************/
extern "C"
{
    DLLEXPORT int __stdcall GetVersionNumber();
    DLLEXPORT const wchar_t *__stdcall GetVersionString();
}

typedef int (__stdcall * proc_GetVersionNumber)();

typedef void * APE_CIO_HANDLE;

typedef int (__stdcall *proc_APECIO_Read)(void* pUserData, void * pBuffer, unsigned int nBytesToRead, unsigned int * pBytesRead);
typedef int (__stdcall *proc_APECIO_Write)(void* pUserData, const void * pBuffer, unsigned int nBytesToWrite, unsigned int * pBytesWritten);
typedef APE::int64 (__stdcall *proc_APECIO_Seek)(void* pUserData, APE::int64 nDistance, unsigned int nMoveMode);
typedef int (__stdcall *proc_APECIO_GetPosition)(void* pUserData);
typedef unsigned int (__stdcall *proc_APECIO_GetSize)(void* pUserData);

extern "C"
{
    DLLEXPORT APE_CIO_HANDLE __stdcall c_APECIO_Create(void* pUserData,
        proc_APECIO_Read CIO_Read,
        proc_APECIO_Write CIO_Write,
        proc_APECIO_Seek CIO_Seek,
        proc_APECIO_GetPosition CIO_GetPosition,
        proc_APECIO_GetSize CIO_GetSize);
    DLLEXPORT void __stdcall c_APECIO_Destroy(APE_CIO_HANDLE);
}

/*****************************************************************************************
IAPECompress wrapper(s)
*****************************************************************************************/
typedef void * APE_COMPRESS_HANDLE;

typedef APE_COMPRESS_HANDLE (__stdcall * proc_APECompress_Create)(int *);
typedef void (__stdcall * proc_APECompress_Destroy)(APE_COMPRESS_HANDLE);
#ifndef EXCLUDE_CIO
typedef int (__stdcall * proc_APECompress_Start)(APE_COMPRESS_HANDLE, const char *, const APE::WAVEFORMATEX *, int, int, const void *, int);
typedef int (__stdcall * proc_APECompress_StartW)(APE_COMPRESS_HANDLE, const APE::str_utfn *, const APE::WAVEFORMATEX *, int, int, const void *, int);
#endif
typedef int (__stdcall * proc_APECompress_StartEx)(APE_COMPRESS_HANDLE, APE_CIO_HANDLE, const APE::WAVEFORMATEX *, int, int, const void *, int);
typedef int (__stdcall * proc_APECompress_AddData)(APE_COMPRESS_HANDLE, unsigned char *, int);
typedef APE::intn (__stdcall * proc_APECompress_GetBufferBytesAvailable)(APE_COMPRESS_HANDLE);
typedef unsigned char * (__stdcall * proc_APECompress_LockBuffer)(APE_COMPRESS_HANDLE, APE::intn *);
typedef int (__stdcall * proc_APECompress_UnlockBuffer)(APE_COMPRESS_HANDLE, int, BOOL);
typedef int (__stdcall * proc_APECompress_Finish)(APE_COMPRESS_HANDLE, unsigned char *, int, int);
typedef int (__stdcall * proc_APECompress_Kill)(APE_COMPRESS_HANDLE);

extern "C"
{
    DLLEXPORT APE_COMPRESS_HANDLE __stdcall c_APECompress_Create(int * pErrorCode = NULL);
    DLLEXPORT void __stdcall c_APECompress_Destroy(APE_COMPRESS_HANDLE hAPECompress);
#ifndef EXCLUDE_CIO
    DLLEXPORT int __stdcall c_APECompress_Start(APE_COMPRESS_HANDLE hAPECompress, const char * pOutputFilename, const APE::WAVEFORMATEX * pwfeInput, int nMaxAudioBytes = MAX_AUDIO_BYTES_UNKNOWN, int nCompressionLevel = COMPRESSION_LEVEL_NORMAL, const void * pHeaderData = NULL, int nHeaderBytes = CREATE_WAV_HEADER_ON_DECOMPRESSION);
    DLLEXPORT int __stdcall c_APECompress_StartW(APE_COMPRESS_HANDLE hAPECompress, const APE::str_utfn * pOutputFilename, const APE::WAVEFORMATEX * pwfeInput, int nMaxAudioBytes = MAX_AUDIO_BYTES_UNKNOWN, int nCompressionLevel = COMPRESSION_LEVEL_NORMAL, const void * pHeaderData = NULL, int nHeaderBytes = CREATE_WAV_HEADER_ON_DECOMPRESSION);
#endif
    DLLEXPORT int __stdcall c_APECompress_StartEx(APE_COMPRESS_HANDLE hAPECompress, APE_CIO_HANDLE hCIO, const APE::WAVEFORMATEX * pwfeInput, int nMaxAudioBytes = MAX_AUDIO_BYTES_UNKNOWN, int nCompressionLevel = COMPRESSION_LEVEL_NORMAL, const void * pHeaderData = NULL, int nHeaderBytes = CREATE_WAV_HEADER_ON_DECOMPRESSION);
    DLLEXPORT int __stdcall c_APECompress_AddData(APE_COMPRESS_HANDLE hAPECompress, unsigned char * pData, int nBytes);
    DLLEXPORT APE::intn __stdcall c_APECompress_GetBufferBytesAvailable(APE_COMPRESS_HANDLE hAPECompress);
    DLLEXPORT unsigned char * __stdcall c_APECompress_LockBuffer(APE_COMPRESS_HANDLE hAPECompress, APE::intn * pBytesAvailable);
    DLLEXPORT    int __stdcall c_APECompress_UnlockBuffer(APE_COMPRESS_HANDLE hAPECompress, int nBytesAdded, BOOL bProcess = true);
    DLLEXPORT    int __stdcall c_APECompress_Finish(APE_COMPRESS_HANDLE hAPECompress, unsigned char * pTerminatingData, int nTerminatingBytes, int nWAVTerminatingBytes);
    DLLEXPORT    int __stdcall c_APECompress_Kill(APE_COMPRESS_HANDLE hAPECompress);
}

/*****************************************************************************************
IAPEDecompress wrapper(s)
*****************************************************************************************/
typedef void * APE_DECOMPRESS_HANDLE;

#ifndef EXCLUDE_CIO
typedef APE_DECOMPRESS_HANDLE (__stdcall * proc_APEDecompress_Create)(const char *, int *);
typedef APE_DECOMPRESS_HANDLE (__stdcall * proc_APEDecompress_CreateW)(const APE::str_utfn *, int *);
#endif
typedef APE_DECOMPRESS_HANDLE (__stdcall * proc_APEDecompress_CreateEx)(APE_CIO_HANDLE, int *);
typedef void (__stdcall * proc_APEDecompress_Destroy)(APE_DECOMPRESS_HANDLE);
typedef int (__stdcall * proc_APEDecompress_GetData)(APE_DECOMPRESS_HANDLE, char *, APE::intn, APE::intn *);
typedef int (__stdcall * proc_APEDecompress_Seek)(APE_DECOMPRESS_HANDLE, int);
typedef APE::intn (__stdcall * proc_APEDecompress_GetInfo)(APE_DECOMPRESS_HANDLE, APE::APE_DECOMPRESS_FIELDS, int, int);

extern "C"
{
#ifndef EXCLUDE_CIO
    DLLEXPORT APE_DECOMPRESS_HANDLE __stdcall c_APEDecompress_Create(const APE::str_ansi * pFilename, int * pErrorCode = NULL);
    DLLEXPORT APE_DECOMPRESS_HANDLE __stdcall c_APEDecompress_CreateW(const APE::str_utfn * pFilename, int * pErrorCode = NULL);
#endif
    DLLEXPORT APE_DECOMPRESS_HANDLE __stdcall c_APEDecompress_CreateEx(APE_CIO_HANDLE hCIO, int * pErrorCode = NULL);
    DLLEXPORT void __stdcall c_APEDecompress_Destroy(APE_DECOMPRESS_HANDLE hAPEDecompress);
    DLLEXPORT int __stdcall c_APEDecompress_GetData(APE_DECOMPRESS_HANDLE hAPEDecompress, char * pBuffer, APE::intn nBlocks, APE::intn * pBlocksRetrieved);
    DLLEXPORT int __stdcall c_APEDecompress_Seek(APE_DECOMPRESS_HANDLE hAPEDecompress, int nBlockOffset);
    DLLEXPORT APE::intn __stdcall c_APEDecompress_GetInfo(APE_DECOMPRESS_HANDLE hAPEDecompress, APE::APE_DECOMPRESS_FIELDS Field, int nParam1 = 0, int nParam2 = 0);
}
