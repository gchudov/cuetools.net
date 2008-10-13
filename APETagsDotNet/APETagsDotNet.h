// APETagsDotNet.h

#pragma once

using namespace System;
using namespace System::Runtime::InteropServices;

#include <string.h>

/*****************************************************************************************
The version of the APE tag
*****************************************************************************************/
#define CURRENT_APE_TAG_VERSION                 2000

/*****************************************************************************************
Footer (and header) flags
*****************************************************************************************/
#define APE_TAG_FLAG_CONTAINS_HEADER            (1 << 31)
#define APE_TAG_FLAG_CONTAINS_FOOTER            (1 << 30)
#define APE_TAG_FLAG_IS_HEADER                  (1 << 29)

#define APE_TAG_FLAGS_DEFAULT                   (APE_TAG_FLAG_CONTAINS_FOOTER)

/*****************************************************************************************
Tag field flags
*****************************************************************************************/
#define TAG_FIELD_FLAG_READ_ONLY                (1 << 0)

#define TAG_FIELD_FLAG_DATA_TYPE_MASK           (6)
#define TAG_FIELD_FLAG_DATA_TYPE_TEXT_UTF8      (0 << 1)
#define TAG_FIELD_FLAG_DATA_TYPE_BINARY         (1 << 1)
#define TAG_FIELD_FLAG_DATA_TYPE_EXTERNAL_INFO  (2 << 1)
#define TAG_FIELD_FLAG_DATA_TYPE_RESERVED       (3 << 1)

/*****************************************************************************************
The footer at the end of APE tagged files (can also optionally be at the front of the tag)
*****************************************************************************************/
#define APE_TAG_FOOTER_BYTES    32

namespace APETagsDotNet {

	ref class APE_TAG_FOOTER
	{
	protected:

		String^ m_cID;              // should equal 'APETAGEX'
		int m_nVersion;             // equals CURRENT_APE_TAG_VERSION
		int m_nSize;                // the complete size of the tag, including this footer (excludes header)
		int m_nFields;              // the number of fields in the tag
		int m_nFlags;               // the tag flags
		array<unsigned char>^ m_cReserved; // reserved for later use (must be zero)

	public:

		APE_TAG_FOOTER(int nFields, int nFieldBytes)
		{
			//TODO
			//m_cID = gcnew array<unsigned char> (8);
			//Marshal::Copy (m_cID, 0, (IntPtr) "APETAGEX", 8);
			m_cID = gcnew String ("APETAGEX");
			m_cReserved = gcnew array<unsigned char> (8);
			//TODO
			//memset(m_cReserved, 0, 8);
			m_nFields = nFields;
			m_nFlags = APE_TAG_FLAGS_DEFAULT;
			m_nSize = nFieldBytes + APE_TAG_FOOTER_BYTES;
			m_nVersion = CURRENT_APE_TAG_VERSION;
		}

		property Int32	TotalTagBytes	{ Int32 get() { return m_nSize + (HasHeader ? APE_TAG_FOOTER_BYTES : 0); } }
		property Int32	FieldBytes		{ Int32 get() { return m_nSize - APE_TAG_FOOTER_BYTES; } }
		property Int32	FieldsOffset	{ Int32 get() { return HasHeader ? APE_TAG_FOOTER_BYTES : 0; } }
		property Int32	NumberFields	{ Int32 get() { return m_nFields; } }
		property bool	HasHeader		{ bool get() { return (m_nFlags & APE_TAG_FLAG_CONTAINS_HEADER) != 0; } }
		property bool	IsHeader		{ bool get() { return (m_nFlags & APE_TAG_FLAG_IS_HEADER) != 0; } }
		property Int32	Version			{ Int32 get() { return m_nVersion; } }

		bool GetIsValid (bool bAllowHeader)
		{
			//TODO
			return //(strncmp(m_cID, "APETAGEX", 8) == 0) && 
				(m_nVersion <= CURRENT_APE_TAG_VERSION) &&
				(m_nFields <= 65536) &&
				(FieldBytes <= (1024 * 1024 * 16)) &&
				(bAllowHeader || !IsHeader);
		}
	};


	public ref class APETagField
	{
	public:
		// create a tag field (use nFieldBytes = -1 for null-terminated strings)
		APETagField (String^ fieldName, array<unsigned char>^ fieldValue, int fieldFlags) {
			_fieldName = fieldName;
			_fieldValue = fieldValue;
			_fieldFlags = fieldFlags;
		}
	    
		// destructor
		~APETagField() {
		}

		// gets the size of the entire field in bytes (name, value, and metadata)
		int GetFieldSize()
		{
			return _fieldName->Length + 1 + _fieldValue->Length + 4 + 4;
		}
	    
		// get the name of the field
		property String^ FieldName {
			String ^ get() { return _fieldName; }
		}

		// get the value of the field
		property array<unsigned char>^  FieldValue {
			array<unsigned char>^ get() { return _fieldValue; }
		}
	    
		// output the entire field to a buffer (GetFieldSize() bytes)
		int SaveField(char * pBuffer)
		{
			//TODO
			//*((int *) pBuffer) = m_nFieldValueBytes;
			//pBuffer += 4;
			//*((int *) pBuffer) = m_nFieldFlags;
			//pBuffer += 4;
		 //   
			//CSmartPtr<char> spFieldNameANSI((char *) GetANSIFromUTF16(m_spFieldNameUTF16), TRUE); 
			//strcpy(pBuffer, spFieldNameANSI);
			//pBuffer += strlen(spFieldNameANSI) + 1;

			//memcpy(pBuffer, m_spFieldValue, m_nFieldValueBytes);

			return GetFieldSize();
		}

		// checks to see if the field is read-only
		property bool	IsReadOnly		{ bool get() { return (_fieldFlags & TAG_FIELD_FLAG_READ_ONLY) != 0; } }
		property bool	IsUTF8Text		{ bool get() { return (_fieldFlags & TAG_FIELD_FLAG_DATA_TYPE_MASK) == TAG_FIELD_FLAG_DATA_TYPE_TEXT_UTF8; } }

		// set helpers (use with EXTREME caution)
		property Int32 FieldFlags {
			Int32 get() { return _fieldFlags; }
			void set(Int32 fieldFlags) { _fieldFlags = fieldFlags; }
		}

	private:    
		String^ _fieldName;
		array<unsigned char>^ _fieldValue;
		int _fieldFlags;
		int _fieldValueBytes;
	};

	public ref class APETagsDotNet
	{
	};
}
