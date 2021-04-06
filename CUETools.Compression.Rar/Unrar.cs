using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Collections;
using CUETools.Compression;

/*  Author:  Michael A. McCloskey
 *  Company: Schematrix
 *  Version: 20040714
 *  
 *  Personal Comments:
 *  I created this unrar wrapper class for personal use 
 *  after running into a number of issues trying to use
 *  another COM unrar product via COM interop.  I hope it 
 *  proves as useful to you as it has to me and saves you
 *  some time in building your own products.
 */

namespace CUETools.Compression.Rar
{
	#region Event Delegate Definitions

	/// <summary>
	/// Represents the method that will handle data available events
	/// </summary>
	public delegate void DataAvailableHandler(object sender, DataAvailableEventArgs e);
	/// <summary>
	/// Represents the method that will handle missing archive volume events
	/// </summary>
	public delegate void MissingVolumeHandler(object sender, MissingVolumeEventArgs e);
	/// <summary>
	/// Represents the method that will handle new volume events
	/// </summary>
	public delegate void NewVolumeHandler(object sender, NewVolumeEventArgs e);
	/// <summary>
	/// Represents the method that will handle new file notifications
	/// </summary>
	public delegate void NewFileHandler(object sender, NewFileEventArgs e);

	#endregion
	
	/// <summary>
	/// Wrapper class for unrar DLL supplied by RARSoft.  
	/// Calls unrar DLL via platform invocation services (pinvoke).
	/// DLL is available at http://www.rarlab.com/rar/UnRARDLL.exe
	/// </summary>
	public class Unrar : IDisposable
	{
		#region Unrar DLL enumerations

		/// <summary>
		/// Mode in which archive is to be opened for processing.
		/// </summary>
		public enum OpenMode
		{
			/// <summary>
			/// Open archive for listing contents only
			/// </summary>
			List=0,
			/// <summary>
			/// Open archive for testing or extracting contents
			/// </summary>
			Extract=1
		}
		
		private enum RarError : uint
		{
			EndOfArchive=10,
			InsufficientMemory=11,
			BadData=12,
			BadArchive=13,
			UnknownFormat=14,
			OpenError=15,
			CreateError=16,
			CloseError=17,
			ReadError=18,
			WriteError=19,
			BufferTooSmall=20,
			UnknownError=21
		}

		private enum Operation : uint
		{
			Skip=0,
			Test=1,
			Extract=2
		}

		private enum VolumeMessage : uint
		{
			Ask=0,
			Notify=1
		}

		[Flags]
			private enum  ArchiveFlags : uint
		{
			Volume=0x1,										// Volume attribute (archive volume)
			CommentPresent=0x2,						// Archive comment present
			Lock=0x4,											// Archive lock attribute
			SolidArchive=0x8,							// Solid attribute (solid archive)
			NewNamingScheme=0x10,					// New volume naming scheme ('volname.partN.rar')
			AuthenticityPresent=0x20,			// Authenticity information present
			RecoveryRecordPresent=0x40,		// Recovery record present
			EncryptedHeaders=0x80,				// Block headers are encrypted
			FirstVolume=0x100							// 0x0100  - First volume (set only by RAR 3.0 and later)
		}

		private enum CallbackMessages : uint
		{
			VolumeChange=0,
			ProcessData=1,
			NeedPassword=2
		}

		#endregion

		#region Unrar DLL structure definitions

		[StructLayout(LayoutKind.Sequential, CharSet=CharSet.Ansi)]
			private struct RARHeaderData
		{
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=260)]
			public string ArcName;
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=260)]
			public string FileName;
			public uint Flags;
			public uint PackSize;
			public uint UnpSize;
			public uint HostOS;
			public uint FileCRC;
			public uint FileTime;
			public uint UnpVer;
			public uint Method;
			public uint FileAttr;
			[MarshalAs(UnmanagedType.LPStr)]
			public string CmtBuf;
			public uint CmtBufSize;
			public uint CmtSize;
			public uint CmtState;

			public void Initialize()
			{
				this.CmtBuf=new string((char)0, 65536);
				this.CmtBufSize=65536;
			}
		}

		[StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
			public struct RARHeaderDataEx
		{
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=512)]
			public string ArcName;
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=1024)]
			public string ArcNameW;
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=512)]
			public string FileName;
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=1024)]
			public string FileNameW;
			public uint Flags;
			public uint PackSize;
			public uint PackSizeHigh;
			public uint UnpSize;
			public uint UnpSizeHigh;
			public uint HostOS;
			public uint FileCRC;
			public uint FileTime;
			public uint UnpVer;
			public uint Method;
			public uint FileAttr;
			[MarshalAs(UnmanagedType.LPStr)]
			public string CmtBuf;
			public uint CmtBufSize;
			public uint CmtSize;
			public uint CmtState;
			[MarshalAs(UnmanagedType.ByValArray, SizeConst=1024)]
			public uint[] Reserved;

			public void Initialize()
			{
				this.CmtBuf=new string((char)0, 65536);
				this.CmtBufSize=65536;
			}
		}

		[StructLayout(LayoutKind.Sequential, CharSet=CharSet.Ansi)]
			public struct RAROpenArchiveData
		{
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst=260)]
			public string ArcName;
			public uint OpenMode;
			public uint OpenResult;
			[MarshalAs(UnmanagedType.LPStr)]
			public string CmtBuf;
			public uint CmtBufSize;
			public uint CmtSize;
			public uint CmtState;

			public void Initialize()
			{
				this.CmtBuf=new string((char)0,65536);
				this.CmtBufSize=65536;
			}
		}

		[StructLayout(LayoutKind.Sequential)]
			public struct RAROpenArchiveDataEx
		{
			[MarshalAs(UnmanagedType.LPStr)]
			public string ArcName;
			[MarshalAs(UnmanagedType.LPWStr)]
			public string ArcNameW;
			public uint OpenMode;
			public uint OpenResult;
			[MarshalAs(UnmanagedType.LPStr)]
			public string CmtBuf;
			public uint CmtBufSize;
			public uint CmtSize;
			public uint CmtState;
			public uint Flags;
			[MarshalAs(UnmanagedType.ByValArray, SizeConst=32)]
			public uint[] Reserved;

			public void Initialize()
			{
				this.CmtBuf=new string((char)0,65536);
				this.CmtBufSize=65536;
				this.Reserved=new uint[32];
			}
		}

		#endregion

		#region Unrar function declarations

		[DllImport("unrar.dll")]
		private static extern IntPtr RAROpenArchive(ref RAROpenArchiveData archiveData);

		[DllImport("UNRAR.DLL")]
		private static extern IntPtr RAROpenArchiveEx(ref RAROpenArchiveDataEx archiveData);

		[DllImport("unrar.dll")]
		private static extern int RARCloseArchive(IntPtr hArcData);

		[DllImport("unrar.dll")]
		private static extern int RARReadHeader(IntPtr hArcData, ref RARHeaderData headerData);

		[DllImport("unrar.dll")]
		private static extern int RARReadHeaderEx(IntPtr hArcData, ref RARHeaderDataEx headerData);

		[DllImport("unrar.dll")]
		private static extern int RARProcessFile(IntPtr hArcData, int operation,
			[MarshalAs(UnmanagedType.LPStr)] string destPath, 
			[MarshalAs(UnmanagedType.LPStr)] string destName );

		[DllImport("unrar.dll")]
		private static extern void RARSetCallback(IntPtr hArcData, UNRARCallback callback, int userData);

		[DllImport("unrar.dll")]
		private static extern void RARSetPassword(IntPtr hArcData,
			[MarshalAs(UnmanagedType.LPStr)] string password);

		// Unrar callback delegate signature
		private delegate int UNRARCallback(uint msg, int UserData, IntPtr p1, int p2);

		#endregion

		#region Public event declarations

		/// <summary>
		/// Event that is raised when a new chunk of data has been extracted
		/// </summary>
		public event DataAvailableHandler DataAvailable;
		/// <summary>
		/// Event that is raised to indicate extraction progress
		/// </summary>
		public event EventHandler<CompressionExtractionProgressEventArgs> ExtractionProgress;
		/// <summary>
		/// Event that is raised when a required archive volume is missing
		/// </summary>
		public event MissingVolumeHandler MissingVolume;
		/// <summary>
		/// Event that is raised when a new file is encountered during processing
		/// </summary>
		public event NewFileHandler NewFile;
		/// <summary>
		/// Event that is raised when a new archive volume is opened for processing
		/// </summary>
		public event NewVolumeHandler NewVolume;
		/// <summary>
		/// Event that is raised when a password is required before continuing
		/// </summary>
		public event EventHandler<CompressionPasswordRequiredEventArgs> PasswordRequired;

		#endregion

		#region Private fields

		private string archivePathName=string.Empty;
		private IntPtr archiveHandle=new IntPtr(0);
		private bool retrieveComment=true;
		private string password=string.Empty;
		private string comment=string.Empty;
		private ArchiveFlags archiveFlags=0;
		private RARHeaderDataEx header=new RARHeaderDataEx();
		private string destinationPath=string.Empty;
		private RARFileInfo currentFile=null;
		private UNRARCallback callback=null;

		#endregion

		#region Object lifetime procedures

		public Unrar() 
		{
			this.callback=new UNRARCallback(RARCallback);
		}

		public Unrar(string archivePathName) : this()
		{
			this.archivePathName=archivePathName;
		}

		~Unrar()
		{
			if(this.archiveHandle!=IntPtr.Zero)
			{
				Unrar.RARCloseArchive(this.archiveHandle);
				this.archiveHandle=IntPtr.Zero;
			}
		}

		public void Dispose()
		{
			if(this.archiveHandle!=IntPtr.Zero)
			{
				Unrar.RARCloseArchive(this.archiveHandle);
				this.archiveHandle=IntPtr.Zero;
			}
		}

		#endregion

		#region Public Properties

		/// <summary>
		/// Path and name of RAR archive to open
		/// </summary>
		public string ArchivePathName
		{
			get
			{
				return this.archivePathName;
			}
			set
			{
				this.archivePathName=value;
			}
		}

		/// <summary>
		/// Archive comment 
		/// </summary>
		public string Comment
		{
			get
			{
				return(this.comment);
			}
		}

		/// <summary>
		/// Current file being processed
		/// </summary>
		public RARFileInfo CurrentFile
		{
			get
			{
				return(this.currentFile);
			}
		}

		/// <summary>
		/// Default destination path for extraction
		/// </summary>
		public string DestinationPath
		{
			get
			{
				return this.destinationPath;
			}
			set
			{
				this.destinationPath=value;
			}
		}

		/// <summary>
		/// Password for opening encrypted archive
		/// </summary>
		public string Password
		{
			get
			{
				return(this.password);
			}
			set
			{
				this.password=value;
				if(this.archiveHandle!=IntPtr.Zero)
					RARSetPassword(this.archiveHandle, value);
			}
		}

		#endregion

		#region Public Methods

		/// <summary>
		/// Close the currently open archive
		/// </summary>
		/// <returns></returns>
		public void Close()
		{
			// Exit without exception if no archive is open
			if(this.archiveHandle==IntPtr.Zero)
				return;

			// Close archive
			int result=Unrar.RARCloseArchive(this.archiveHandle);

			// Check result
			if(result!=0)
			{
				ProcessFileError(result);
			}
			else
			{
				this.archiveHandle=IntPtr.Zero;
			}
		}

		/// <summary>
		/// Opens archive specified by the ArchivePathName property for testing or extraction
		/// </summary>
		public void Open()
		{
			if(this.ArchivePathName.Length==0)
				throw new IOException("Archive name has not been set.");
			this.Open(this.ArchivePathName, OpenMode.Extract);
		}

		/// <summary>
		/// Opens archive specified by the ArchivePathName property with a specified mode
		/// </summary>
		/// <param name="openMode">Mode in which archive should be opened</param>
		public void Open(OpenMode openMode)
		{
			if(this.ArchivePathName.Length==0)
				throw new IOException("Archive name has not been set.");
			this.Open(this.ArchivePathName, openMode);
		}

		/// <summary>
		/// Opens specified archive using the specified mode.  
		/// </summary>
		/// <param name="archivePathName">Path of archive to open</param>
		/// <param name="openMode">Mode in which to open archive</param>
		public void Open(string archivePathName, OpenMode openMode)
		{
			IntPtr handle=IntPtr.Zero;

			// Close any previously open archives
			if(this.archiveHandle!=IntPtr.Zero)
				this.Close();

			// Prepare extended open archive struct
			this.ArchivePathName=archivePathName;
			RAROpenArchiveDataEx openStruct=new RAROpenArchiveDataEx();
			openStruct.Initialize();
			openStruct.ArcName=this.archivePathName+"\0";
			openStruct.ArcNameW=this.archivePathName+"\0";
			openStruct.OpenMode=(uint)openMode;
			if(this.retrieveComment)
			{
				openStruct.CmtBuf=new string((char)0,65536);
				openStruct.CmtBufSize=65536;
			}
			else
			{
				openStruct.CmtBuf=null;
				openStruct.CmtBufSize=0;
			}

			// Open archive
			handle=Unrar.RAROpenArchiveEx(ref openStruct);

			// Check for success
			if(openStruct.OpenResult!=0)
			{
				switch((RarError)openStruct.OpenResult)
				{
					case RarError.InsufficientMemory:
						throw new OutOfMemoryException("Insufficient memory to perform operation.");

					case RarError.BadData:
						throw new IOException("Archive header broken");

					case RarError.BadArchive:
						throw new IOException("File is not a valid archive.");

					case RarError.OpenError:
						throw new IOException("File could not be opened.");
				}
			}

			// Save handle and flags
			this.archiveHandle=handle;
			this.archiveFlags=(ArchiveFlags)openStruct.Flags;

			// Set callback
			Unrar.RARSetCallback(this.archiveHandle, this.callback, this.GetHashCode());

			// If comment retrieved, save it
			if(openStruct.CmtState==1)
				this.comment=openStruct.CmtBuf.ToString();

			// If password supplied, set it
			if(this.password.Length!=0)
				Unrar.RARSetPassword(this.archiveHandle, this.password);

			// Fire NewVolume event for first volume
			this.OnNewVolume(this.archivePathName);
		}

		/// <summary>
		/// Reads the next archive header and populates CurrentFile property data
		/// </summary>
		/// <returns></returns>
		public bool ReadHeader()
		{
			// Throw exception if archive not open
			if(this.archiveHandle==IntPtr.Zero)
				throw new IOException("Archive is not open.");

			// Initialize header struct
			this.header=new RARHeaderDataEx();
			header.Initialize();

			// Read next entry
			currentFile=null;
			int result=Unrar.RARReadHeaderEx(this.archiveHandle, ref this.header);

			// Check for error or end of archive
			if((RarError)result==RarError.EndOfArchive)
				return false;
			else if((RarError)result==RarError.BadData)
				throw new IOException("Archive data is corrupt.");

			// Determine if new file
			if(((header.Flags & 0x01) != 0) && currentFile!=null)
				currentFile.ContinuedFromPrevious=true;
			else
			{
				// New file, prepare header
				currentFile=new RARFileInfo();
				currentFile.FileName=header.FileNameW.ToString();
				if((header.Flags & 0x02) != 0)
					currentFile.ContinuedOnNext=true;
				if(header.PackSizeHigh != 0)
					currentFile.PackedSize=(header.PackSizeHigh * 0x100000000) + header.PackSize;
				else
					currentFile.PackedSize=header.PackSize;
				if(header.UnpSizeHigh != 0)
					currentFile.UnpackedSize=(header.UnpSizeHigh * 0x100000000) + header.UnpSize;
				else
					currentFile.UnpackedSize=header.UnpSize;
				currentFile.HostOS=(int)header.HostOS;
				currentFile.FileCRC=header.FileCRC;
				currentFile.FileTime=FromMSDOSTime(header.FileTime);
				currentFile.VersionToUnpack=(int)header.UnpVer;
				currentFile.Method=(int)header.Method;
				currentFile.FileAttributes=(int)header.FileAttr;
				currentFile.BytesExtracted=0;
				if((header.Flags & 0x20) == 0x20)
					currentFile.IsDirectory=true;
				this.OnNewFile();
			}

			// Return success
			return true;
		}

		/// <summary>
		/// Returns array of file names contained in archive
		/// </summary>
		/// <returns></returns>
		public string[] ListFiles()
		{
			ArrayList fileNames=new ArrayList();
			while(this.ReadHeader())
			{
				if(!currentFile.IsDirectory)
					fileNames.Add(currentFile.FileName);
				this.Skip();
			}
			string[] files=new string[fileNames.Count];
			fileNames.CopyTo(files);
			return files;
		}

		/// <summary>
		/// Moves the current archive position to the next available header
		/// </summary>
		/// <returns></returns>
		public void Skip()
		{
			int result=Unrar.RARProcessFile(this.archiveHandle, (int)Operation.Skip, string.Empty, string.Empty);

			// Check result
			if(result!=0)
			{
				ProcessFileError(result);
			}
		}

		/// <summary>
		/// Tests the ability to extract the current file without saving extracted data to disk
		/// </summary>
		/// <returns></returns>
		public void Test()
		{
			int result=Unrar.RARProcessFile(this.archiveHandle, (int)Operation.Test, string.Empty, string.Empty);

			// Check result
			if(result!=0)
			{
				ProcessFileError(result);
			}
		}

		/// <summary>
		/// Extracts the current file to the default destination path
		/// </summary>
		/// <returns></returns>
		public void Extract()
		{
			this.Extract(this.destinationPath, string.Empty);
		}

		/// <summary>
		/// Extracts the current file to a specified destination path and filename
		/// </summary>
		/// <param name="destinationName">Path and name of extracted file</param>
		/// <returns></returns>
		public void Extract(string destinationName)
		{
			this.Extract(string.Empty, destinationName);
		}

		/// <summary>
		/// Extracts the current file to a specified directory without renaming file
		/// </summary>
		/// <param name="destinationPath"></param>
		/// <returns></returns>
		public void ExtractToDirectory(string destinationPath)
		{
			this.Extract(destinationPath, string.Empty);
		}

		#endregion

		#region Private Methods

		private void Extract(string destinationPath, string destinationName)
		{
			int result=Unrar.RARProcessFile(this.archiveHandle, (int)Operation.Extract, destinationPath, destinationName);

			// Check result
			if(result!=0)
			{
				ProcessFileError(result);
			}
		}

		private DateTime FromMSDOSTime(uint dosTime)
		{
			int day=0;
			int month=0;
			int year=0;
			int second=0;
			int hour=0;
			int minute=0;
			ushort hiWord;
			ushort loWord;
			hiWord=(ushort)((dosTime & 0xFFFF0000) >> 16);
			loWord=(ushort)(dosTime & 0xFFFF);
			year=((hiWord & 0xFE00) >> 9)+1980;
			month=(hiWord & 0x01E0) >> 5;
			day=hiWord & 0x1F;
			hour=(loWord & 0xF800) >> 11;
			minute=(loWord & 0x07E0) >> 5;
			second=(loWord & 0x1F) << 1;
			return new DateTime(year, month, day, hour, minute, second);
		}

		private void ProcessFileError(int result)
		{
			switch((RarError)result)
			{
				case RarError.UnknownFormat:
					throw new OutOfMemoryException("Unknown archive format.");

				case RarError.BadData:
					throw new IOException("File CRC Error");

				case RarError.BadArchive:
					throw new IOException("File is not a valid archive.");

				case RarError.OpenError:
					throw new IOException("File could not be opened.");

				case RarError.CreateError:
					throw new IOException("File could not be created.");

				case RarError.CloseError:
					throw new IOException("File close error.");

				case RarError.ReadError:
					throw new IOException("File read error.");

				case RarError.WriteError:
					throw new IOException("File write error.");
			}
		}

		private int RARCallback(uint msg, int UserData, IntPtr p1, int p2)
		{
			string volume=string.Empty;
			string newVolume=string.Empty;
			int result=-1;

			switch((CallbackMessages)msg)
			{
				case CallbackMessages.VolumeChange:
					volume=Marshal.PtrToStringAnsi(p1);
					if((VolumeMessage)p2==VolumeMessage.Notify)
						result=OnNewVolume(volume);
					else if((VolumeMessage)p2==VolumeMessage.Ask)
					{
						newVolume=OnMissingVolume(volume);
						if(newVolume.Length==0)
							result=-1;
						else
						{
							if(newVolume!=volume)
							{
								for(int i=0; i<newVolume.Length; i++)
								{
									Marshal.WriteByte(p1, i, (byte)newVolume[i]);
								}
								Marshal.WriteByte(p1, newVolume.Length, (byte)0);
							}
							result=1;
						}
					}
					break;

				case CallbackMessages.ProcessData:
					result=OnDataAvailable(p1, p2);
					break;

				case CallbackMessages.NeedPassword:
					result=OnPasswordRequired(p1, p2);
					break;
			}
			return result;
		}

		#endregion

		#region Protected Virtual (Overridable) Methods

		protected virtual void OnNewFile()
		{
			if(this.NewFile!=null)
			{
				NewFileEventArgs e = new NewFileEventArgs(this.currentFile);
				this.NewFile(this, e);
			}
		}

		protected virtual int OnPasswordRequired(IntPtr p1, int p2)
		{
			int result=-1;
			if(this.PasswordRequired!=null)
			{
				CompressionPasswordRequiredEventArgs e = new CompressionPasswordRequiredEventArgs();
				this.PasswordRequired(this, e);
				if(e.ContinueOperation && e.Password.Length>0)
				{
					for(int i=0; (i<e.Password.Length) && (i<p2); i++)
						Marshal.WriteByte(p1, i, (byte)e.Password[i]);
					Marshal.WriteByte(p1, e.Password.Length, (byte)0);
					result=1;
				}
			}
			else
			{
				throw new IOException("Password is required for extraction.");
			}
			return result;
		}

		protected virtual int OnDataAvailable(IntPtr p1, int p2)
		{
			int result=1;
			if(this.currentFile!=null)
				this.currentFile.BytesExtracted+=p2;
			if(this.DataAvailable!=null)
			{
				byte[] data=new byte[p2];
				Marshal.Copy(p1, data, 0, p2);
				DataAvailableEventArgs e=new DataAvailableEventArgs(data);
				this.DataAvailable(this, e);
				if(!e.ContinueOperation)
					result=-1;
			}
			if((this.ExtractionProgress!=null) && (this.currentFile!=null))
			{
				CompressionExtractionProgressEventArgs e = new CompressionExtractionProgressEventArgs();
				e.FileName=this.currentFile.FileName;
				e.FileSize=this.currentFile.UnpackedSize;
				e.BytesExtracted=this.currentFile.BytesExtracted;
				e.PercentComplete=this.currentFile.PercentComplete;
				this.ExtractionProgress(this, e);
				if(!e.ContinueOperation)
					result=-1;
			}
			return result;
		}

		protected virtual int OnNewVolume(string volume)
		{
			int result=1;
			if(this.NewVolume!=null)
			{
				NewVolumeEventArgs e=new NewVolumeEventArgs(volume);
				this.NewVolume(this, e);
				if(!e.ContinueOperation)
					result=-1;
			}
			return result;
		}

		protected virtual string OnMissingVolume(string volume)
		{
			string result=string.Empty;
			if(this.MissingVolume!=null)
			{
				MissingVolumeEventArgs e=new MissingVolumeEventArgs(volume);
				this.MissingVolume(this, e);
				if(e.ContinueOperation)
					result=e.VolumeName;
			}
			return result;
		}

		#endregion
	}

	#region Event Argument Classes

	public class NewVolumeEventArgs
	{
		public string VolumeName;
		public bool ContinueOperation=true;

		public NewVolumeEventArgs(string volumeName)
		{
			this.VolumeName=volumeName;
		}
	}

	public class MissingVolumeEventArgs
	{
		public string VolumeName;
		public bool ContinueOperation=false;

		public MissingVolumeEventArgs(string volumeName)
		{
			this.VolumeName=volumeName;
		}
	}

	public class DataAvailableEventArgs
	{
		public readonly byte[] Data;
		public bool ContinueOperation=true;

		public DataAvailableEventArgs(byte[] data)
		{
			this.Data=data;
		}
	}

	public class NewFileEventArgs
	{
		public RARFileInfo fileInfo;
		public NewFileEventArgs(RARFileInfo fileInfo)
		{
			this.fileInfo=fileInfo;
		}
	}

	public class RARFileInfo
	{
		public string FileName;
		public bool		ContinuedFromPrevious=false;
		public bool		ContinuedOnNext=false;
		public bool		IsDirectory=false;
		public long		PackedSize=0;
		public long		UnpackedSize=0;
		public int		HostOS=0;
		public long		FileCRC=0;
		public				DateTime FileTime;
		public int		VersionToUnpack=0;
		public int		Method=0;
		public int		FileAttributes=0;
		public long		BytesExtracted=0;

		public double PercentComplete
		{
			get
			{
				if(this.UnpackedSize!=0)
					return(((double)this.BytesExtracted/(double)this.UnpackedSize) * (double)100.0);
				else
					return (double)0;
			}
		}
	}

	#endregion
}
