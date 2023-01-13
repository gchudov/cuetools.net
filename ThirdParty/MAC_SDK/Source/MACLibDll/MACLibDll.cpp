#include "MACLibDll.h"
#include "IO.h"

using namespace APE;

class CallbackCIO : public CIO
{
public:
    // construction / destruction
    CallbackCIO(
        void * pUserData,
        proc_APECIO_Read CIO_Read,
        proc_APECIO_Write CIO_Write,
        proc_APECIO_Seek CIO_Seek,
        proc_APECIO_GetPosition CIO_GetPosition,
        proc_APECIO_GetSize CIO_GetSize)
    {
        m_pUserData = pUserData;
        m_CIO_Read = CIO_Read;
        m_CIO_Write = CIO_Write;
        m_CIO_Seek = CIO_Seek;
        m_CIO_GetPosition = CIO_GetPosition;
        m_CIO_GetSize = CIO_GetSize;
    }
    ~CallbackCIO() {}

    // open / close
    int Open(const wchar_t * pName, bool bOpenReadOnly = false)
    {
        return -1;
    }
    int Close()
    {
        return -1;
    }

    // read / write
    int Read(void * pBuffer, unsigned int nBytesToRead, unsigned int * pBytesRead)
    {
        return m_CIO_Read(m_pUserData, pBuffer, nBytesToRead, pBytesRead);
    }
    int Write(const void * pBuffer, unsigned int nBytesToWrite, unsigned int * pBytesWritten)
    {
        return m_CIO_Write(m_pUserData, pBuffer, nBytesToWrite, pBytesWritten);
    }

    // seek
    APE::int64 PerformSeek()
    {
        return m_CIO_Seek(m_pUserData, m_nSeekPosition, m_nSeekMethod);
    }

    // other functions
    int SetEOF()
    {
        return -1;
    }

    // creation / destruction
    int Create(const wchar_t * pName)
    {
        return -1;
    }
    int Delete()
    {
        return -1;
    }

    // attributes
    int GetPosition()
    {
        return m_CIO_GetPosition(m_pUserData);
    }
    unsigned int GetSize()
    {
        return m_CIO_GetSize(m_pUserData);
    }
    int GetName(wchar_t * pBuffer)
    {
        return -1;
    }

private:
    void * m_pUserData;
    proc_APECIO_Read m_CIO_Read;
    proc_APECIO_Write m_CIO_Write;
    proc_APECIO_Seek m_CIO_Seek;
    proc_APECIO_GetPosition m_CIO_GetPosition;
    proc_APECIO_GetSize m_CIO_GetSize;
};

int __stdcall GetVersionNumber()
{
    return MAC_FILE_VERSION_NUMBER;
}

const wchar_t *__stdcall GetVersionString()
{
    return MAC_VERSION_STRING;
}

APE_CIO_HANDLE __stdcall c_APECIO_Create(void* pUserData,
        proc_APECIO_Read CIO_Read,
        proc_APECIO_Write CIO_Write,
        proc_APECIO_Seek CIO_Seek,
        proc_APECIO_GetPosition CIO_GetPosition,
        proc_APECIO_GetSize CIO_GetSize)
{
    return (APE_CIO_HANDLE) new CallbackCIO(pUserData, CIO_Read, CIO_Write, CIO_Seek, CIO_GetPosition, CIO_GetSize);
}

void __stdcall c_APECIO_Destroy(APE_CIO_HANDLE hCIO)
{
    CallbackCIO * pCIO = (CallbackCIO *) hCIO;
    if (pCIO)
        delete pCIO;
}

/*****************************************************************************************
CAPEDecompress wrapper(s)
*****************************************************************************************/
#ifndef EXCLUDE_CIO
APE_DECOMPRESS_HANDLE __stdcall c_APEDecompress_Create(const str_ansi * pFilename, int * pErrorCode)
{
    CSmartPtr<wchar_t> spFilename(CAPECharacterHelper::GetUTF16FromANSI(pFilename), TRUE);
    return (APE_DECOMPRESS_HANDLE) CreateIAPEDecompress(spFilename, pErrorCode);
}

APE_DECOMPRESS_HANDLE __stdcall c_APEDecompress_CreateW(const str_utfn * pFilename, int * pErrorCode)
{
    return (APE_DECOMPRESS_HANDLE) CreateIAPEDecompress(pFilename, pErrorCode);
}
#endif

APE_DECOMPRESS_HANDLE __stdcall c_APEDecompress_CreateEx(APE_CIO_HANDLE hCIO, int * pErrorCode)
{
    return (APE_DECOMPRESS_HANDLE) CreateIAPEDecompressEx((CallbackCIO *) hCIO, pErrorCode);
}

void __stdcall c_APEDecompress_Destroy(APE_DECOMPRESS_HANDLE hAPEDecompress)
{
    IAPEDecompress * pAPEDecompress = (IAPEDecompress *) hAPEDecompress;
    if (pAPEDecompress)
        delete pAPEDecompress;
}

int __stdcall c_APEDecompress_GetData(APE_DECOMPRESS_HANDLE hAPEDecompress, char * pBuffer, intn nBlocks, intn * pBlocksRetrieved)
{
    return ((IAPEDecompress *) hAPEDecompress)->GetData(pBuffer, nBlocks, pBlocksRetrieved);
}

int __stdcall c_APEDecompress_Seek(APE_DECOMPRESS_HANDLE hAPEDecompress, int nBlockOffset)
{
    return ((IAPEDecompress *) hAPEDecompress)->Seek(nBlockOffset);
}

intn __stdcall c_APEDecompress_GetInfo(APE_DECOMPRESS_HANDLE hAPEDecompress, APE_DECOMPRESS_FIELDS Field, int nParam1, int nParam2)
{
    return ((IAPEDecompress *) hAPEDecompress)->GetInfo(Field, nParam1, nParam2);
}

/*****************************************************************************************
CAPECompress wrapper(s)
*****************************************************************************************/
APE_COMPRESS_HANDLE __stdcall c_APECompress_Create(int * pErrorCode)
{
    return (APE_COMPRESS_HANDLE) CreateIAPECompress(pErrorCode);
}

void __stdcall c_APECompress_Destroy(APE_COMPRESS_HANDLE hAPECompress)
{
    IAPECompress * pAPECompress = (IAPECompress *) hAPECompress;
    if (pAPECompress)
        delete pAPECompress;
}

#ifndef EXCLUDE_CIO
int __stdcall c_APECompress_Start(APE_COMPRESS_HANDLE hAPECompress, const char * pOutputFilename, const APE::WAVEFORMATEX * pwfeInput, int nMaxAudioBytes, int nCompressionLevel, const void * pHeaderData, int nHeaderBytes)
{
    CSmartPtr<wchar_t> spOutputFilename(CAPECharacterHelper::GetUTF16FromANSI(pOutputFilename), TRUE);
    return ((IAPECompress *) hAPECompress)->Start(spOutputFilename, pwfeInput, nMaxAudioBytes, nCompressionLevel, pHeaderData, nHeaderBytes);
}

int __stdcall c_APECompress_StartW(APE_COMPRESS_HANDLE hAPECompress, const str_utfn * pOutputFilename, const APE::WAVEFORMATEX * pwfeInput, int nMaxAudioBytes, int nCompressionLevel, const void * pHeaderData, int nHeaderBytes)
{
    return ((IAPECompress *) hAPECompress)->Start(pOutputFilename, pwfeInput, nMaxAudioBytes, nCompressionLevel, pHeaderData, nHeaderBytes);
}
#endif

int __stdcall c_APECompress_StartEx(APE_COMPRESS_HANDLE hAPECompress, APE_CIO_HANDLE hCIO, const APE::WAVEFORMATEX * pwfeInput, int nMaxAudioBytes, int nCompressionLevel, const void * pHeaderData, int nHeaderBytes)
{
    return ((IAPECompress *) hAPECompress)->StartEx((CallbackCIO *) hCIO, pwfeInput, nMaxAudioBytes, nCompressionLevel, pHeaderData, nHeaderBytes);
}

int __stdcall c_APECompress_AddData(APE_COMPRESS_HANDLE hAPECompress, unsigned char * pData, int nBytes)
{
    return ((IAPECompress *) hAPECompress)->AddData(pData, nBytes);
}

intn __stdcall c_APECompress_GetBufferBytesAvailable(APE_COMPRESS_HANDLE hAPECompress)
{
    return ((IAPECompress *) hAPECompress)->GetBufferBytesAvailable();
}

unsigned char * __stdcall c_APECompress_LockBuffer(APE_COMPRESS_HANDLE hAPECompress, intn * pBytesAvailable)
{
    return ((IAPECompress *) hAPECompress)->LockBuffer(pBytesAvailable);
}

int __stdcall c_APECompress_UnlockBuffer(APE_COMPRESS_HANDLE hAPECompress, int nBytesAdded, BOOL bProcess)
{
    return ((IAPECompress *) hAPECompress)->UnlockBuffer(nBytesAdded, bProcess);
}

int __stdcall c_APECompress_Finish(APE_COMPRESS_HANDLE hAPECompress, unsigned char * pTerminatingData, int nTerminatingBytes, int nWAVTerminatingBytes)
{
    return ((IAPECompress *) hAPECompress)->Finish(pTerminatingData, nTerminatingBytes, nWAVTerminatingBytes);
}

int __stdcall c_APECompress_Kill(APE_COMPRESS_HANDLE hAPECompress)
{
    return ((IAPECompress *) hAPECompress)->Kill();
}
