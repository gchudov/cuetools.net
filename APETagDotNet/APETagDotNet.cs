using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using System.IO;

namespace APETagsDotNet
{
	public class LittleEndian
	{
		static public Int32 Read32 (byte[] buffer, int offset)
		{
			return buffer[offset] + (buffer[offset + 1] << 8) + (buffer[offset + 2] << 16) + (buffer[offset + 3] << 24);
		}
		static public void Write32(byte[] buffer, int offset, Int32 value)
		{
			buffer[offset++] = (byte) (value & 0xff); value >>= 8;
			buffer[offset++] = (byte) (value & 0xff); value >>= 8;
			buffer[offset++] = (byte) (value & 0xff); value >>= 8;
			buffer[offset++] = (byte) (value & 0xff); value >>= 8;
		}
	}

	public class APE_TAG_FOOTER
	{
		public APE_TAG_FOOTER(int nFields, int nFieldBytes)
		{		
			m_cID = new ASCIIEncoding().GetBytes ("APETAGEX");
			m_cReserved = new byte[8];
			m_nFields = nFields;
			m_nFlags = APETagDotNet.APE_TAG_FLAGS_DEFAULT;
			m_nSize = nFieldBytes + APETagDotNet.APE_TAG_FOOTER_BYTES;
			m_nVersion = APETagDotNet.CURRENT_APE_TAG_VERSION;
		}

		public APE_TAG_FOOTER(byte[] buffer)
		{
			m_cID = new byte[8];
			m_cReserved = new byte[8];

			Array.Copy(buffer, m_cID, 8);
			m_nVersion = LittleEndian.Read32(buffer, 8);
			m_nSize = LittleEndian.Read32(buffer, 12);
			m_nFields = LittleEndian.Read32(buffer, 16);
			m_nFlags = LittleEndian.Read32(buffer, 20);
			Array.Copy (buffer, 24, m_cReserved, 0, 8);
		}

		public int Save(byte[] spRawTag, int nLocation)
		{
			Array.Copy(m_cID, 0, spRawTag, nLocation, 8);
			LittleEndian.Write32(spRawTag, nLocation + 8, m_nVersion);
			LittleEndian.Write32(spRawTag, nLocation + 12, m_nSize);
			LittleEndian.Write32(spRawTag, nLocation + 16, m_nFields);
			LittleEndian.Write32(spRawTag, nLocation + 20, m_nFlags);
			Array.Copy(m_cReserved, 0, spRawTag, nLocation + 24, 8);
			return APETagDotNet.APE_TAG_FOOTER_BYTES;
		}

		public int TotalTagBytes { get { return m_nSize + (HasHeader ? APETagDotNet.APE_TAG_FOOTER_BYTES : 0); } }
		public int FieldBytes { get { return m_nSize - APETagDotNet.APE_TAG_FOOTER_BYTES; } }
		public int FieldsOffset { get { return HasHeader ? APETagDotNet.APE_TAG_FOOTER_BYTES : 0; } }
		public int NumberFields { get { return m_nFields; } }
		public bool HasHeader { get { return (m_nFlags & APETagDotNet.APE_TAG_FLAG_CONTAINS_HEADER) != 0; } }
		public bool IsHeader { get { return (m_nFlags & APETagDotNet.APE_TAG_FLAG_IS_HEADER) != 0; } }
		public int Version { get { return m_nVersion; } }
		public bool IsValid
		{
			get
			{
				return new ASCIIEncoding().GetString (m_cID) == "APETAGEX" &&
					(m_nVersion <= APETagDotNet.CURRENT_APE_TAG_VERSION) &&
					(m_nFields <= 65536) &&
					(FieldBytes <= (1024 * 1024 * 16));
			}
		}
	
		private byte[] m_cID;              // should equal 'APETAGEX'
		private int m_nVersion;             // equals CURRENT_APE_TAG_VERSION
		private int m_nSize;                // the complete size of the tag, including this footer (excludes header)
		private int m_nFields;              // the number of fields in the tag
		private int m_nFlags;               // the tag flags
		private byte[] m_cReserved; // reserved for later use (must be zero)
	};


	public class APETagField
	{
		// create a tag field (use nFieldBytes = -1 for null-terminated strings)
		public APETagField (string fieldName, byte[] fieldValue, int fieldFlags) {
			_fieldName = fieldName;
			_fieldValue = fieldValue;
			_fieldFlags = fieldFlags;
		}
		    
		// destructor
		~APETagField() {
		}

		// gets the size of the entire field in bytes (name, value, and metadata)
		public int GetFieldSize()
		{
			return _fieldName.Length + 1 + _fieldValue.Length + 4 + 4;
		}
	    
		// get the name of the field
		public string FieldName { get { return _fieldName; } }

		// get the value of the field
		public byte[]  FieldValue { get { return _fieldValue; } }
	    
		// output the entire field to a buffer (GetFieldSize() bytes)
		public int SaveField(byte[] pBuffer, int pos)
		{
			LittleEndian.Write32(pBuffer, pos, _fieldValue.Length);
			LittleEndian.Write32(pBuffer, pos + 4, _fieldFlags);
			Array.Copy (new ASCIIEncoding().GetBytes(_fieldName), pBuffer, pos + 8);
			pBuffer[pos + 8 + _fieldName.Length] = 0;
			Array.Copy(_fieldValue, pBuffer, pos + 8 + _fieldName.Length + 1);
			return GetFieldSize();
		}

		// checks to see if the field is read-only
		public bool IsReadOnly { get { return (_fieldFlags & APETagDotNet.TAG_FIELD_FLAG_READ_ONLY) != 0; } }
		public bool IsUTF8Text { get { return (_fieldFlags & APETagDotNet.TAG_FIELD_FLAG_DATA_TYPE_MASK) == APETagDotNet.TAG_FIELD_FLAG_DATA_TYPE_TEXT_UTF8; } }

		// set helpers (use with EXTREME caution)
		public int FieldFlags {
			get { return _fieldFlags; }
			set { _fieldFlags = value; }
		}

		private string _fieldName;
		private byte[] _fieldValue;
		private int _fieldFlags;
	};

	public class APETagDotNet
	{
		/*****************************************************************************************
		The version of the APE tag
		*****************************************************************************************/
		public const int CURRENT_APE_TAG_VERSION					= 2000;

		public const int ID3_TAG_BYTES = 128;

		/*****************************************************************************************
		Footer (and header) flags
		*****************************************************************************************/
		public const int APE_TAG_FLAG_CONTAINS_HEADER = (1 << 31);
		public const int APE_TAG_FLAG_CONTAINS_FOOTER = (1 << 30);
		public const int APE_TAG_FLAG_IS_HEADER = (1 << 29);

		public const int APE_TAG_FLAGS_DEFAULT = (APE_TAG_FLAG_CONTAINS_FOOTER);

		/*****************************************************************************************
		Tag field flags
		*****************************************************************************************/
		public const int TAG_FIELD_FLAG_READ_ONLY = (1 << 0);

		public const int TAG_FIELD_FLAG_DATA_TYPE_MASK = (6);
		public const int TAG_FIELD_FLAG_DATA_TYPE_TEXT_UTF8 = (0 << 1);
		public const int TAG_FIELD_FLAG_DATA_TYPE_BINARY = (1 << 1);
		public const int TAG_FIELD_FLAG_DATA_TYPE_EXTERNAL_INFO = (2 << 1);
		public const int TAG_FIELD_FLAG_DATA_TYPE_RESERVED = (3 << 1);

		/*****************************************************************************************
		The footer at the end of APE tagged files (can also optionally be at the front of the tag)
		*****************************************************************************************/
		public const int APE_TAG_FOOTER_BYTES = 32;


		// create an APE tags object
		// bAnalyze determines whether it will analyze immediately or on the first request
		// be careful with multiple threads / file pointer movement if you don't analyze immediately
		public APETagDotNet (string filename, bool analyze)
		{
			m_spIO = new FileStream (filename, FileMode.Open, FileAccess.Read, FileShare.Read);
			m_bAnalyzed = false;
			m_aryFields = new APETagField[0];
			m_nTagBytes = 0;
			m_bIgnoreReadOnly = false;
			if (analyze) Analyze ();
		}
	    
		// destructor
		~APETagDotNet () { ClearFields (); }

		// save the tag to the I/O source (bUseOldID3 forces it to save as an ID3v1.1 tag instead of an APE tag)
		int Save () 
		{
			if (!Remove(false))
				return -1;

			if (m_aryFields.Length == 0) { return 0; }

			int z = 0;

			// calculate the size of the whole tag
			int nFieldBytes = 0;
			for (z = 0; z < m_aryFields.Length; z++)
				nFieldBytes += m_aryFields[z].GetFieldSize();

			// sort the fields
			SortFields();

			// build the footer
			APE_TAG_FOOTER APETagFooter = new APE_TAG_FOOTER(m_aryFields.Length, nFieldBytes);

			// make a buffer for the tag
			int nTotalTagBytes = APETagFooter.TotalTagBytes;
			byte[] spRawTag = new byte[APETagFooter.TotalTagBytes];

			// save the fields
			int nLocation = 0;
			for (z = 0; z < m_aryFields.Length; z++)
				nLocation += m_aryFields[z].SaveField (spRawTag, nLocation);

			// add the footer to the buffer
			nLocation += APETagFooter.Save(spRawTag, nLocation);

			// dump the tag to the I/O source
			WriteBufferToEndOfIO (spRawTag);
			return 0;
		}
	    
		// removes any tags from the file (bUpdate determines whether is should re-analyze after removing the tag)
		bool Remove(bool bUpdate)
		{
			// variables
			int nBytesRead = 0;
			long nOriginalPosition = m_spIO.Position;

			bool bID3Removed = true;
			bool bAPETagRemoved = true;

			bool bFailedToRemove = false;

			while (bID3Removed || bAPETagRemoved)
			{
				bID3Removed = false;
				bAPETagRemoved = false;

				// ID3 tag
				if (m_spIO.Length > ID3_TAG_BYTES)
				{
					byte[] cTagHeader = new byte[3];
					m_spIO.Seek (-ID3_TAG_BYTES, SeekOrigin.End);
					nBytesRead = m_spIO.Read (cTagHeader, 0, 3);
					if (nBytesRead == 3)
					{
						if (cTagHeader[0]=='T' && cTagHeader[1]=='A' && cTagHeader[2]=='G')
						{
							try { m_spIO.SetLength(m_spIO.Length - ID3_TAG_BYTES); }
							catch { bFailedToRemove = true; }
							if (!bFailedToRemove)
								bID3Removed = true;
						}
					}
				}

				// APE Tag
				if (m_spIO.Length > APE_TAG_FOOTER_BYTES && bFailedToRemove == false)
				{
					APE_TAG_FOOTER APETagFooter;
					m_spIO.Seek(-APE_TAG_FOOTER_BYTES, SeekOrigin.End);
					byte [] buf = new byte[APE_TAG_FOOTER_BYTES];
					nBytesRead = m_spIO.Read(buf, 0, APE_TAG_FOOTER_BYTES);
					if (nBytesRead == APE_TAG_FOOTER_BYTES)
					{
						APETagFooter = new APE_TAG_FOOTER(buf);
						if (APETagFooter.IsValid)
						{
							try { m_spIO.SetLength(m_spIO.Length - APETagFooter.TotalTagBytes); }
							catch { bFailedToRemove = true; }
							if (!bFailedToRemove)
								bAPETagRemoved = true;
						}
					}
				}

			}
		    
			m_spIO.Seek(nOriginalPosition, SeekOrigin.Begin);

			if (bUpdate && bFailedToRemove == false)
			{
				m_bAnalyzed = false;
				Analyze();
			}
			return !bFailedToRemove;

		}

		// sets the value of a field (use nFieldBytes = -1 for null terminated strings)
		// note: using NULL or "" for a string type will remove the field
		void SetFieldString(string fieldName, string fieldValue)
		{
			// remove if empty
			if (fieldValue == "")
			{
				RemoveField(fieldName);
				return;
			}
			UTF8Encoding enc = new UTF8Encoding();
			// UTF-8 encode the value and call the binary SetField(...)
			SetFieldBinary (fieldName, enc.GetBytes (fieldValue), TAG_FIELD_FLAG_DATA_TYPE_TEXT_UTF8);
		}
		//int SetFieldString(string pFieldName, const char * pFieldValue, bool bAlreadyUTF8Encoded);
		void SetFieldBinary(string fieldName, byte[] fieldValue, int fieldFlags)
		{
			Analyze();

			// check to see if we're trying to remove the field (by setting it to NULL or an empty string)
			bool bRemoving = (fieldValue.Length == 0);

			// get the index
			int nFieldIndex = GetTagFieldIndex (fieldName);
			if (nFieldIndex != -1)
			{
				// existing field

				// fail if we're read-only (and not ignoring the read-only flag)
				if (!m_bIgnoreReadOnly && (m_aryFields[nFieldIndex].IsReadOnly))
					throw new Exception("read only");
		        
				// erase the existing field
				//SAFE_DELETE(m_aryFields[nFieldIndex])

				if (bRemoving)
				{
					RemoveField(nFieldIndex);
					return;
				}
			}
			else
			{
				if (bRemoving)
					return;
				nFieldIndex = m_aryFields.Length;
				Array.Resize (ref m_aryFields, nFieldIndex + 1);
			}
		    
			// create the field and add it to the field array
			m_aryFields[nFieldIndex] = new APETagField (fieldName, fieldValue, fieldFlags);
		}

		// gets the value of a field (returns -1 and an empty buffer if the field doesn't exist)
		byte[] GetFieldBinary(string pFieldName)
		{
			Analyze();
			APETagField pAPETagField = GetTagField (pFieldName);
			return (pAPETagField == null) ? null : pAPETagField.FieldValue;
		}
		public int Count { get { Analyze(); return m_aryFields.Length; } }
		public string GetFieldString(string pFieldName) { return GetFieldString(GetTagFieldIndex(pFieldName)); }
		public string GetFieldString(int Index)
		{
			Analyze();
			APETagField pAPETagField = GetTagField (Index);
			if (pAPETagField == null)
				return null;
			if (m_nAPETagVersion < 2000)
				return new ASCIIEncoding().GetString(pAPETagField.FieldValue);
			if (!pAPETagField.IsUTF8Text)
				return null;
			return new UTF8Encoding().GetString(pAPETagField.FieldValue);
		}

		public NameValueCollection GetStringTags(bool mapToFlac) 
		{
			Analyze();
			NameValueCollection tags = new NameValueCollection ();
			for (int i = 0; i < Count; i++)
			{
				string fieldName = m_aryFields[i].FieldName;
				string fieldValue = GetFieldString (i);
				if (fieldName != null && fieldValue != null)
				{
					if (mapToFlac)
					{
						if (fieldName.ToUpper() == "YEAR")
							fieldName = "DATE";
						if (fieldName.ToUpper() == "TRACK")
							fieldName = "TRACKNUMBER";
					}
					tags.Add (fieldName, fieldValue);
				}
			}
			return tags;
		}


		// remove a specific field
		void RemoveField(string pFieldName) 
		{
			RemoveField(GetTagFieldIndex(pFieldName));
		}

		void RemoveField(int nIndex)
		{
			for (int i = nIndex; i < m_aryFields.Length-1; i++)
				m_aryFields[i] = m_aryFields[i+1];
			Array.Resize (ref m_aryFields, m_aryFields.Length-1);
		}

		// clear all the fields
		void ClearFields() { m_aryFields = new APETagField[0]; }
	    
		// get the total tag bytes in the file from the last analyze
		// need to call Save() then Analyze() to update any changes
		int GetTagBytes() {
			Analyze();
			return m_nTagBytes;
		}

		// fills in an ID3_TAG using the current fields (useful for quickly converting the tag)
		//int CreateID3Tag(ID3_TAG * pID3Tag);

		// see whether the file has an ID3 or APE tag
		//bool GetHasID3Tag() { if (!m_bAnalyzed) { Analyze(); } return m_bHasID3Tag;    }
		bool GetHasAPETag() { Analyze(); return m_bHasAPETag;    }
		int GetAPETagVersion() { return GetHasAPETag() ? m_nAPETagVersion : -1;    }

		// gets a desired tag field (returns NULL if not found)
		// again, be careful, because this a pointer to the actual field in this class
		APETagField GetTagField(string pFieldName)
		{
			int nIndex = GetTagFieldIndex (pFieldName);
			return (nIndex != -1) ? m_aryFields[nIndex] : null;
		}
		public APETagField GetTagField(int nIndex) 
		{
			Analyze();
			if (nIndex >= 0 && nIndex < m_aryFields.Length)
				return m_aryFields[nIndex];
			return null;
		}

		// options
		void SetIgnoreReadOnly(bool bIgnoreReadOnly) { m_bIgnoreReadOnly = bIgnoreReadOnly; }

		// private functions
		private int Analyze()
		{
			if (m_bAnalyzed)
				return 0;
			// clean-up
			// ID3_TAG ID3Tag;
			ClearFields();
			m_nTagBytes = 0;

			m_bAnalyzed = true;

			// store the original location
			long nOriginalPosition = m_spIO.Position;	
		    
			// check for a tag
			int nBytesRead;
			//m_bHasID3Tag = false;
			m_bHasAPETag = false;
			m_nAPETagVersion = -1;
			
			//m_spIO->Seek(-ID3_TAG_BYTES, FILE_END);
			//nRetVal = m_spIO->Read((unsigned char *) &ID3Tag, sizeof(ID3_TAG), &nBytesRead);
		 //   
			//if ((nBytesRead == sizeof(ID3_TAG)) && (nRetVal == 0))
			//{
			//	if (ID3Tag.Header[0] == 'T' && ID3Tag.Header[1] == 'A' && ID3Tag.Header[2] == 'G') 
			//	{
			//		m_bHasID3Tag = true;
			//		m_nTagBytes += ID3_TAG_BYTES;
			//	}
			//}
		 //   
			//// set the fields
			//if (m_bHasID3Tag)
			//{
			//	SetFieldID3String(APE_TAG_FIELD_ARTIST, ID3Tag.Artist, 30);
			//	SetFieldID3String(APE_TAG_FIELD_ALBUM, ID3Tag.Album, 30);
			//	SetFieldID3String(APE_TAG_FIELD_TITLE, ID3Tag.Title, 30);
			//	SetFieldID3String(APE_TAG_FIELD_COMMENT, ID3Tag.Comment, 28);
			//	SetFieldID3String(APE_TAG_FIELD_YEAR, ID3Tag.Year, 4);
		 //       
			//	char cTemp[16]; sprintf(cTemp, "%d", ID3Tag.Track);
			//	SetFieldString(APE_TAG_FIELD_TRACK, cTemp, false);

			//	if ((ID3Tag.Genre == GENRE_UNDEFINED) || (ID3Tag.Genre >= GENRE_COUNT)) 
			//		SetFieldString(APE_TAG_FIELD_GENRE, APE_TAG_GENRE_UNDEFINED);
			//	else 
			//		SetFieldString(APE_TAG_FIELD_GENRE, g_ID3Genre[ID3Tag.Genre]);
			//}

			// try loading the APE tag
			//if (m_bHasID3Tag == false)
			{
				byte[] pBuffer = new byte[APE_TAG_FOOTER_BYTES];
				m_spIO.Seek (-APE_TAG_FOOTER_BYTES, SeekOrigin.End);
				nBytesRead = m_spIO.Read (pBuffer, 0, APE_TAG_FOOTER_BYTES);
				if (nBytesRead == APE_TAG_FOOTER_BYTES)
				{
					APE_TAG_FOOTER APETagFooter = new APE_TAG_FOOTER(pBuffer);
					if (APETagFooter.IsValid && !APETagFooter.IsHeader)
					{
						m_bHasAPETag = true;
						m_nAPETagVersion = APETagFooter.Version;

						int nRawFieldBytes = APETagFooter.FieldBytes;
						m_nTagBytes += APETagFooter.TotalTagBytes;
		                
						byte[] spRawTag = new byte[nRawFieldBytes];
						m_spIO.Seek(-(APETagFooter.TotalTagBytes - APETagFooter.FieldsOffset), SeekOrigin.End);
						nBytesRead = m_spIO.Read(spRawTag, 0, nRawFieldBytes);

						if (nRawFieldBytes == nBytesRead)
						{
							// parse out the raw fields
							int nLocation = 0;
							for (int z = 0; z < APETagFooter.NumberFields; z++)
							{
								if (!LoadField(spRawTag, ref nLocation))
								{
									// if LoadField(...) fails, it means that the tag is corrupt (accidently or intentionally)
									// we'll just bail out -- leaving the fields we've already set
									break;
								}
							}
						}
					}
				}
			}
			// restore the file pointer
			m_spIO.Seek(nOriginalPosition, SeekOrigin.Begin);
		    
			return 0;
		}
		private int GetTagFieldIndex(string pFieldName)
		{
			Analyze();
			for (int z = 0; z < m_aryFields.Length; z++)
				if (m_aryFields[z].FieldName.ToUpper() == pFieldName.ToUpper())
					return z;
			return -1;
		}
		private void WriteBufferToEndOfIO (byte[] pBuffer) 
		{
			Int64 nOriginalPosition = m_spIO.Position;		    
			m_spIO.Seek(0, SeekOrigin.End);
			m_spIO.Write(pBuffer, 0, pBuffer.Length);    
			m_spIO.Seek(nOriginalPosition, SeekOrigin.Begin);
		}
		private bool LoadField(byte[] pBuffer, ref int nLocation)
		{
			// size and flags
			int nFieldValueSize = LittleEndian.Read32(pBuffer, nLocation);
			int nFieldFlags = LittleEndian.Read32(pBuffer, nLocation+4); 
			nLocation += 8;
		    
			// safety check (so we can't get buffer overflow attacked)
			int nMaximumRead = pBuffer.Length - nLocation - nFieldValueSize;
			for (int z = 0; z < nMaximumRead; z++)
			{
				int nCharacter = pBuffer[nLocation + z];
				if (nCharacter == 0)
					break;
				if ((nCharacter < 0x20) || (nCharacter > 0x7E))
					return false;
			}

			// name
			ASCIIEncoding ascii = new ASCIIEncoding();
			int nNameCharacters = 0;
			for ( nNameCharacters = nLocation; nNameCharacters < pBuffer.Length && pBuffer[nNameCharacters] != 0; nNameCharacters++)
				;
			string spName = ascii.GetString (pBuffer, nLocation, nNameCharacters-nLocation);
			byte [] spFieldBuffer = new byte[nFieldValueSize];
			
			nLocation = nNameCharacters + 1;

			// value
			Array.Copy(pBuffer, nLocation, spFieldBuffer, 0, nFieldValueSize);
			nLocation += nFieldValueSize;

			// set
			SetFieldBinary(spName, spFieldBuffer, nFieldFlags);
			return true;
		}
		private void SortFields()
		{
			SortedList<byte[],string> list = new SortedList<byte[],string>();
		}
		// TODO private static int CompareFields(const void * pA, const void * pB);

		// helper set / get field functions
		//int SetFieldID3String(string pFieldName, const char * pFieldValue, int nBytes);
		//int GetFieldID3String(string pFieldName, char * pBuffer, int nBytes);

		// private data
		private FileStream m_spIO;
		private bool m_bAnalyzed;
		private int m_nTagBytes;
		private APETagField[] m_aryFields;
		private bool m_bHasAPETag;
		private int m_nAPETagVersion;
		//private bool m_bHasID3Tag;
		private bool m_bIgnoreReadOnly;
	};
}
