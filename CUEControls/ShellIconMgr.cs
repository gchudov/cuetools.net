//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for Windows XP
// 
// Copyright (C) 2006 by Jack W. Griffin (butchg@comcast.net)
//
// This program is free software; you can redistribute it and/or modify 
// it under the terms of the GNU General Public License as published by 
// the Free Software Foundation; either version 2 of the License, or 
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
// for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program; if not, write to the 
//
// Free Software Foundation, Inc., 
// 59 Temple Place, Suite 330, 
// Boston, MA 02111-1307 USA
//
using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Windows.Forms;
using System.IO;

namespace CUEControls
{
	/// <summary>
	/// This class manages the association of files and the ICON used to
	/// represent the file.  This class calls into the shell function SHGetFileInfo to get
	/// the ICON assocaited with the file name given (or device name).
	/// </summary>
	public unsafe class ShellIconMgr
	{
		#region external functions
		[DllImport("Shell32.dll", CharSet = CharSet.Unicode)]
		private static extern IntPtr SHGetFileInfo(string pszPath, uint dwFileAttributes, ref SHFILEINFO psfi, uint cbFileInfo, uint uFlags);
		[DllImport("Shell32.dll", CharSet = CharSet.Unicode, EntryPoint="SHGetFileInfo")]
		private static extern IntPtr SHGetFileInfoPIDL(IntPtr pidl, uint dwFileAttributes, ref SHFILEINFO psfi, uint cbFileInfo, uint uFlags);
		/// <summary>
		/// Destroys a PIDL.
		/// </summary>
		[DllImport("shell32.dll")]
		private static extern void ILFree(IntPtr pidl);
		/// <summary>
		/// Retrieves the path of a folder as an PIDL.
		/// </summary>
		[DllImport("shell32.dll")]
		private static extern Int32 SHGetFolderLocation(
			IntPtr hwndOwner,		// Handle to the owner window.
			Int32 nFolder,			// A CSIDL value that identifies the folder to be located.
			IntPtr hToken,			// Token that can be used to represent a particular user.
			UInt32 dwReserved,		// Reserved.
			out IntPtr ppidl);		// Address of a pointer to an item identifier list structure 
									// specifying the folder's location relative to the root of the namespace 
									// (the desktop). 
		[DllImport("shell32.dll")]
		private static extern int SHGetFolderPath(IntPtr hwndOwner, int nFolder, IntPtr hToken,
		   uint dwFlags, [Out] StringBuilder pszPath);
		[DllImport("user32.dll")]
		private static extern bool DestroyIcon(IntPtr handle);
		#endregion

		#region structures for external functions
		[FlagsAttribute]
		private enum GetInfoFlags : uint
		{
			SHGFI_ICON = 0x000000100,               // get icon
			SHGFI_DISPLAYNAME = 0x000000200,        // get display name
			SHGFI_TYPENAME = 0x000000400,           // get type name
			SHGFI_ATTRIBUTES = 0x000000800,         // get attributes
			SHGFI_ICONLOCATION = 0x000001000,       // get icon location
			SHGFI_EXETYPE = 0x000002000,            // return exe type
			SHGFI_SYSICONINDEX = 0x000004000,       // get system icon index
			SHGFI_LINKOVERLAY = 0x000008000,        // put a link overlay on icon
			SHGFI_SELECTED = 0x000010000,           // show icon in selected state
			SHGFI_ATTR_SPECIFIED = 0x000020000,     // get only specified attributes
			SHGFI_LARGEICON = 0x000000000,          // get large icon
			SHGFI_SMALLICON = 0x000000001,          // get small icon
			SHGFI_OPENICON = 0x000000002,           // get open icon
			SHGFI_SHELLICONSIZE = 0x000000004,      // get shell size icon
			SHGFI_PIDL = 0x000000008,               // pszPath is a pidl
			SHGFI_USEFILEATTRIBUTES = 0x000000010,  // use passed dwFileAttribute
			SHGFI_ADDOVERLAYS = 0x000000020,        // apply the appropriate overlays
			SHGFI_OVERLAYINDEX = 0x000000040        // Get the index of the overlay
		} ;

		[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
		struct SHFILEINFO
		{
			public const int NAMESIZE = 80;
			public IntPtr hIcon;
			public int iIcon;
			public uint dwAttributes;
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
			public string szDisplayName;
			[MarshalAs(UnmanagedType.ByValTStr, SizeConst = 80)]
			public string szTypeName;
		};
		#endregion

		#region private variables
		private ImageList m_image_list;
		private IDictionary<int, int> m_index_map;
		private IDictionary<string, int> m_extension_map;
		#endregion

		#region constructor
		/// <summary>
		/// This creates a new shell icon manager.
		/// </summary>
		public ShellIconMgr()
		{
			m_image_list = new ImageList();

			m_index_map = new Dictionary<int, int>();
			m_extension_map = new Dictionary<string, int>();
			m_image_list.ImageSize = new Size(16, 16);
			m_image_list.ColorDepth = ColorDepth.Depth32Bit;
		}
		#endregion

		#region public properties
		/// <summary>
		/// Return the image list that contains the icons found to date.
		/// </summary>
		public ImageList ImageList
		{
			get
			{
				return m_image_list;
			}
		}
		#endregion

		#region public methods
		/// <summary>
		/// Get the icon index associated with a given filename
		/// </summary>
		/// <param name="filename">the filename of interest</param>
		/// <param name="open">if true, the file is "open", most useful for folders</param>
		/// <returns>the index into the image list for the icon associated with this file</returns>
		public int GetIconIndex(FileSystemInfo filename, bool open)
		{
			int iIcon;
			if (filename is FileInfo && m_extension_map.TryGetValue(filename.Extension.ToLower(), out iIcon)) 
				return iIcon;
			GetInfoFlags flags = GetInfoFlags.SHGFI_ICON | GetInfoFlags.SHGFI_SMALLICON | GetInfoFlags.SHGFI_USEFILEATTRIBUTES;
			if (open)
				flags |= GetInfoFlags.SHGFI_OPENICON;
			SHFILEINFO info = new SHFILEINFO();
			SHGetFileInfo(filename.FullName, (uint)filename.Attributes, ref info, (uint)Marshal.SizeOf(info), (uint)flags);
			iIcon = MapIcon(info.hIcon, info.iIcon);
			DestroyIcon(info.hIcon);
			return iIcon;
		}

		/// <summary>
		/// Get the icon index associated with a given filename
		/// </summary>
		/// <param name="filename">the filename of interest</param>
		/// <param name="open">if true, the file is "open", most useful for folders</param>
		/// <returns>the index into the image list for the icon associated with this file</returns>
		public int GetIconIndex(ExtraSpecialFolder folder, bool open)
		{
			GetInfoFlags flags = GetInfoFlags.SHGFI_ICON | GetInfoFlags.SHGFI_SMALLICON | GetInfoFlags.SHGFI_PIDL;
			if (open)
				flags |= GetInfoFlags.SHGFI_OPENICON;

			IntPtr ppidl;
			if (SHGetFolderLocation(IntPtr.Zero, (int)folder, IntPtr.Zero, 0, out ppidl) != 0)
				throw new Exception("SHGetFolderLocation failed");

			SHFILEINFO info = new SHFILEINFO();
			SHGetFileInfoPIDL(ppidl, 0, ref info, (uint)Marshal.SizeOf(info), (uint)flags);
			int iIcon = MapIcon(info.hIcon, info.iIcon);
			ILFree(ppidl);
			DestroyIcon(info.hIcon);
			return iIcon;
		}
		#endregion

		#region private methods

		private int MapIcon(IntPtr hIcon, int iIcon)
		{
			int index = 0;
			if (!m_index_map.TryGetValue(iIcon, out index))
			{
				m_image_list.Images.Add(Icon.FromHandle(hIcon));
				index = m_image_list.Images.Count - 1;
				m_index_map.Add(iIcon, index);
			}
			return index;
		}

		public void SetExtensionIcon(string extension, Image icon)
		{
			m_image_list.Images.Add(extension, icon);
			m_extension_map.Add(extension, m_image_list.Images.Count - 1);
		}

		public void SetExtensionIcon(string extension, Icon icon)
		{
			m_image_list.Images.Add(extension, icon);
			m_extension_map.Add(extension, m_image_list.Images.Count - 1);
		}

		public string GetFolderPath(ExtraSpecialFolder folder)
		{
			StringBuilder SB = new StringBuilder(1024);
			if (0 != SHGetFolderPath(IntPtr.Zero, (int)folder, IntPtr.Zero, 0x0000, SB))
				return null;
			return SB.ToString();
		}

		public void SetFolderPath(ExtraSpecialFolder folder, string path)
		{
			Guid rfid;
			IKnownFolderManager knownFolderManager = (IKnownFolderManager) new KnownFolderManager();
			knownFolderManager.FolderIdFromCsidl((int) folder, out rfid);
			if (path == null)
			{
				IKnownFolder knownFolderInterface;
				knownFolderManager.GetFolder(ref rfid, out knownFolderInterface);
				knownFolderInterface.GetPath(KnownFolderPathFlags.KF_FLAG_DEFAULT_PATH, out path);
			}
			//knownFolderInterface.SetPath(KnownFolderPathFlags.KF_FLAG_DONT_UNEXPAND, path);
			string ppszError;
			knownFolderManager.Redirect(ref rfid, IntPtr.Zero, 0, path, 0, ref rfid, out ppszError);
		}

		public string GetDisplayName(FileSystemInfo filename)
		{
			GetInfoFlags flags = GetInfoFlags.SHGFI_DISPLAYNAME;
			SHFILEINFO info = new SHFILEINFO();
			if (SHGetFileInfo(filename.FullName, (uint)filename.Attributes, ref info, (uint)Marshal.SizeOf(info), (uint)flags) == IntPtr.Zero)
				return filename.Name;
			return info.szDisplayName;
		}

		public string GetDisplayName(ExtraSpecialFolder folder)
		{
			GetInfoFlags flags = GetInfoFlags.SHGFI_PIDL | GetInfoFlags.SHGFI_DISPLAYNAME;
			SHFILEINFO info = new SHFILEINFO();
			IntPtr ppidl;
			if (SHGetFolderLocation(IntPtr.Zero, (int) folder, IntPtr.Zero, 0, out ppidl) != 0)
				throw new Exception("SHGetFolderLocation failed");
			if (SHGetFileInfoPIDL(ppidl, 0, ref info, (uint)Marshal.SizeOf(info), (uint)flags) == IntPtr.Zero)
			{
				ILFree(ppidl);
				throw new Exception("SHGetFileInfo failed");
			}
			ILFree(ppidl);
			return info.szDisplayName;
		}
		#endregion
	}

	// see ShlObj.h
	public enum ExtraSpecialFolder
	{
		// Summary:
		//     The logical Desktop rather than the physical file system location.
		Desktop = 0,
		//
		// Summary:
		//     The directory that contains the user's program groups.
		Programs = 2,
		//
		// Summary:
		//     The "My Documents" folder.
		//
		// Version 6.0. The virtual folder representing the My Documents
		// desktop item. This is equivalent to CSIDL_MYDOCUMENTS.
		// Previous to Version 6.0. The file system directory used to
		// physically store a user's common repository of documents.
		// A typical path is C:\Documents and Settings\username\My Documents.
		// This should be distinguished from the virtual My Documents folder
		// in the namespace. To access that virtual folder,
		// use SHGetFolderLocation, which returns the ITEMIDLIST for the
		// virtual location, or refer to the technique described in
		// Managing the File System.
		MyDocuments = 5,
		//
		// Summary:
		//     The directory that serves as a common repository for the user's favorite
		//     items.
		Favorites = 6,
		//
		// Summary:
		//     The directory that corresponds to the user's Startup program group.
		Startup = 7,
		//
		// Summary:
		//     The directory that contains the user's most recently used documents.
		Recent = 8,
		//
		// Summary:
		//     The directory that contains the Send To menu items.
		SendTo = 9,
		//
		// Summary:
		//     The directory that contains the Start menu items.
		StartMenu = 11,
		//
		// Summary:
		//     The "My Music" folder.
		MyMusic = 13,
		//
		// Summary:
		//     The directory used to physically store file objects on the desktop.
		DesktopDirectory = 16,
		//
		// Summary:
		//     The "My Computer" folder.
		MyComputer = 17,
		//
		// Summary:
		//     The directory that serves as a common repository for document templates.
		Templates = 21,
		//
		// Summary:
		//     The directory that serves as a common repository for application-specific
		//     data for the current roaming user.
		//
		// Version 4.71. The file system directory that serves as
		// a common repository for application-specific data.
		// A typical path is C:\Documents and Settings\username\Application Data.
		// This CSIDL is supported by the redistributable Shfolder.dll
		// for systems that do not have the Microsoft Internet Explorer 4.0
		// integrated Shell installed
		ApplicationData = 26,
		//
		// Summary:
		//     The directory that serves as a common repository for application-specific
		//     data that is used by the current, non-roaming user.
		//
		// Version 5.0. The file system directory that serves as a data
		// repository for local (nonroaming) applications. A typical path
		// is C:\Documents and Settings\username\Local Settings\Application Data.
		LocalApplicationData = 28,
		//
		// Summary:
		//     The directory that serves as a common repository for temporary Internet files.
		//
		// Version 4.72. The file system directory that serves as
		// a common repository for temporary Internet files. A typical
		// path is C:\Documents and Settings\username\Local Settings\Temporary Internet Files.
		InternetCache = 32,
		//
		// Summary:
		//     The directory that serves as a common repository for Internet cookies.
		//
		// The file system directory that serves as a common repository
		// for Internet cookies. A typical path is
		// C:\Documents and Settings\username\Cookies.
		Cookies = 33,
		//
		// Summary:
		//     The directory that serves as a common repository for Internet history items.
		History = 34,
		//
		// Summary:
		//     The directory that serves as a common repository for application-specific
		//     data that is used by all users.
		//
		// Version 5.0. The file system directory containing
		// application data for all users. A typical path is
		// C:\Documents and Settings\All Users\Application Data.
		CommonApplicationData = 35,

		// Summary:
		//     The Windows directory.
		//
		// Version 5.0. The Windows directory or SYSROOT.
		// This corresponds to the %windir% or %SYSTEMROOT% environment
		// variables. A typical path is C:\Windows.
		Windows = 0x0024,

		//
		// Summary:
		//     The System directory.
		//
		// Version 5.0. The Windows System folder. A typical
		// path is C:\Windows\System32.
		System = 37,

		//
		// Summary:
		//     The program files directory.
		//
		// Version 5.0. The Program Files folder. A typical
		// path is C:\Program Files.
		ProgramFiles = 38,
		//
		// Summary:
		//     The "My Pictures" folder.
		//
		// Version 5.0. The file system directory that serves as
		// a common repository for image files. A typical path is
		// C:\Documents and Settings\username\My Documents\My Pictures.
		MyPictures = 39,
		// User Profile
		Profile = 0x0028,
		//
		// Summary:
		//     The directory for components that are shared across applications.
		//
		// Version 5.0. A folder for components that are shared across
		// applications. A typical path is C:\Program Files\Common.
		// Valid only for Windows NT, Windows 2000, and Windows XP systems.
		// Not valid for Windows Millennium Edition (Windows Me).
		CommonProgramFiles = 43,

		// The file system directory that contains documents
		// that are common to all users. A typical paths is
		// C:\Documents and Settings\All Users\Documents.
		// Valid for Windows NT systems and Microsoft Windows 95 and
		// Windows 98 systems with Shfolder.dll installed.
		CommonDocuments = 0x002e,

		// Version 5.0. The file system directory containing
		// administrative tools for all users of the computer.
		CommonAdministrativeTools = 0x002f,

		// Version 5.0. The file system directory that is used
		// to store administrative tools for an individual user.
		// The Microsoft Management Console (MMC) will save customized
		// consoles to this directory, and it will roam with the user.
		AdministrativeTools = 0x0030,

		// Music common to all users
		CommonMusic = 0x0035

		// Version 5.0. Combine this CSIDL with any of the following CSIDLs
		// to force the creation of the associated folder.
		// CreateFlag = 0x8000
	}

	public enum SIGDN : uint
	{
		NORMALDISPLAY = 0,
		PARENTRELATIVEPARSING = 0x80018001,
		PARENTRELATIVEFORADDRESSBAR = 0x8001c001,
		DESKTOPABSOLUTEPARSING = 0x80028000,
		PARENTRELATIVEEDITING = 0x80031001,
		DESKTOPABSOLUTEEDITING = 0x8004c000,
		FILESYSPATH = 0x80058000,
		URL = 0x80068000
	}

	[ComImport]
	[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
	[Guid("43826d1e-e718-42ee-bc55-a1e261c37bfe")]
	public interface IShellItem
	{
		void BindToHandler(IntPtr pbc,
			[MarshalAs(UnmanagedType.LPStruct)]Guid bhid,
			[MarshalAs(UnmanagedType.LPStruct)]Guid riid,
			out IntPtr ppv);

		void GetParent(out IShellItem ppsi);

		void GetDisplayName(SIGDN sigdnName, out IntPtr ppszName);

		void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);

		void Compare(IShellItem psi, uint hint, out int piOrder);
	};

	[ComImport, Guid("3AA7AF7E-9B36-420c-A8E3-F77D4674A488"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
	interface IKnownFolder
	{
		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetId(out Guid pkfid);

		// Not yet supported - adding to fill slot in vtable
		void spacer1();
		////[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		//void GetCategory(out mbtagKF_CATEGORY pCategory);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetShellItem([In] uint dwFlags, ref Guid riid, out IShellItem ppv);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetPath([In] KnownFolderPathFlags dwFlags, [MarshalAs(UnmanagedType.LPWStr)] out string ppszPath);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void SetPath([In] KnownFolderPathFlags dwFlags, [In, MarshalAs(UnmanagedType.LPWStr)] string pszPath);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetLocation([In] uint dwFlags, [Out, ComAliasName("ShellObjects.wirePIDL")] IntPtr ppidl);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetFolderType(out Guid pftid);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetRedirectionCapabilities(out uint pCapabilities);

		////[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetFolderDefinition([Out, MarshalAs(UnmanagedType.Struct)] out KNOWNFOLDER_DEFINITION definition);
	}

	[Flags]
	enum KF_CATEGORY
	{
		KF_CATEGORY_VIRTUAL = 0x00000001,
		KF_CATEGORY_FIXED = 0x00000002,
		KF_CATEGORY_COMMON = 0x00000003,
		KF_CATEGORY_PERUSER = 0x00000004
	}

	[Flags]
	enum KF_DEFINITION_FLAGS
	{
		KFDF_PERSONALIZE = 0x00000001,
		KFDF_LOCAL_REDIRECT_ONLY = 0x00000002,
		KFDF_ROAMABLE = 0x00000004,
	}

	[Flags]
	enum KnownFolderRedirectFlags
	{
		/// <summary>
		/// Ensure that only the user has permission to access the redirected folder. 
		/// </summary>
		KF_REDIRECT_USER_EXCLUSIVE = 0x00000001,
		/// <summary>
		/// Copy the discretionary access control list (DACL) of the source folder to the target to maintain current access permissions.  
		/// </summary>
		KF_REDIRECT_COPY_SOURCE_DACL = 0x00000002,
		/// <summary>
		/// Sets the user as the owner of a newly created target folder unless the user is a member of the Administrator group, in which case Administrator is set as the owner. Must be called with KF_REDIRECT_SET_OWNER_EXPLICIT. 
		/// </summary>
		KF_REDIRECT_OWNER_USER = 0x00000004,
		/// <summary>
		/// Set the owner of a newly created target folder. If the user belongs to the Administrators group, Administrators is assigned as the owner. Must be called with KF_REDIRECT_OWNER_USER. 
		/// </summary>
		KF_REDIRECT_SET_OWNER_EXPLICIT = 0x00000008,
		/// <summary>
		/// Do not perform a redirection, simply check whether redirection has occurred. If so, IKnownFolderManager::Redirect returns S_OK; if not, or if some actions remain to be completed, it returns S_FALSE. 
		/// </summary>
		KF_REDIRECT_CHECK_ONLY = 0x00000010,
		/// <summary>
		/// Display user interface (UI) during the redirection.
		/// </summary>		
		KF_REDIRECT_WITH_UI = 0x00000020,
		/// <summary>
		/// Unpin the source folder.
		/// </summary>
		KF_REDIRECT_UNPIN = 0x00000040,
		/// <summary>
		/// Pin the target folder.
		/// </summary> 
		KF_REDIRECT_PIN = 0x00000080,
		/// <summary>
		/// Copy the existing contents—both files and subfolders—of the known folder to the redirected folder.
		/// </summary>
		KF_REDIRECT_COPY_CONTENTS = 0x00000200,
		/// <summary>
		/// Delete the contents of the source folder after they have been copied to the redirected folder. This flag is valid only if KF_REDIRECT_COPY_CONTENTS is set. 
		/// </summary>
		KF_REDIRECT_DEL_SOURCE_CONTENTS = 0x00000400,
		/// <summary>
		/// Reserved. Do not use.
		/// </summary>
		KF_REDIRECT_EXCLUDE_ALL_KNOWN_SUBFOLDERS = 0x00000800,
	}

	[Flags]
	enum KnownFolderPathFlags : uint
	{
		KF_FLAG_CREATE = 0x00008000,
		KF_FLAG_DONT_VERIFY = 0x00004000,
		KF_FLAG_DONT_UNEXPAND = 0x00002000,
		KF_FLAG_NO_ALIAS = 0x00001000,
		KF_FLAG_INIT = 0x00000800,
		KF_FLAG_DEFAULT_PATH = 0x00000400,
		KF_FLAG_NOT_PARENT_RELATIVE = 0x00000200,
		KF_FLAG_SIMPLE_IDLIST = 0x00000100,
		KF_FLAG_ALIAS_ONLY = 0x80000000
	};

	public enum KnownFolderFindMode : int
	{
		ExactMatch = 0,
		NearestParentMatch = ExactMatch + 1
	};

	[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto, Pack = 4)]
	struct KNOWNFOLDER_DEFINITION
	{
		public KF_CATEGORY category;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszName;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszCreator;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszDescription;

		public Guid fidParent;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszRelativePath;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszParsingName;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszToolTip;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszLocalizedName;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszIcon;

		[MarshalAs(UnmanagedType.LPWStr)]
		public string pszSecurity;

		public uint dwAttributes;
		public KF_DEFINITION_FLAGS kfdFlags;
		public Guid ftidType;
	}

	public class KnownFolderIdentifiers
	{
		public static Guid Computer = new Guid(0x0AC0837C, 0xBBF8, 0x452A, 0x85, 0x0D, 0x79, 0xD0, 0x8E, 0x66, 0x7C, 0xA7);
		public static Guid Conflict = new Guid(0x4bfefb45, 0x347d, 0x4006, 0xa5, 0xbe, 0xac, 0x0c, 0xb0, 0x56, 0x71, 0x92);
		public static Guid ControlPanel = new Guid(0x82A74AEB, 0xAEB4, 0x465C, 0xA0, 0x14, 0xD0, 0x97, 0xEE, 0x34, 0x6D, 0x63);
		public static Guid Desktop = new Guid(0xB4BFCC3A, 0xDB2C, 0x424C, 0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41);
		public static Guid Internet = new Guid(0x4D9F7874, 0x4E0C, 0x4904, 0x96, 0x7B, 0x40, 0xB0, 0xD2, 0x0C, 0x3E, 0x4B);
		public static Guid Network = new Guid(0xD20BEEC4, 0x5CA8, 0x4905, 0xAE, 0x3B, 0xBF, 0x25, 0x1E, 0xA0, 0x9B, 0x53);
		public static Guid Printers = new Guid(0x76FC4E2D, 0xD6AD, 0x4519, 0xA6, 0x63, 0x37, 0xBD, 0x56, 0x06, 0x81, 0x85);
		public static Guid SyncManager = new Guid(0x43668BF8, 0xC14E, 0x49B2, 0x97, 0xC9, 0x74, 0x77, 0x84, 0xD7, 0x84, 0xB7);
		public static Guid Connections = new Guid(0x6F0CD92B, 0x2E97, 0x45D1, 0x88, 0xFF, 0xB0, 0xD1, 0x86, 0xB8, 0xDE, 0xDD);
		public static Guid SyncSetup = new Guid(0xf214138, 0xb1d3, 0x4a90, 0xbb, 0xa9, 0x27, 0xcb, 0xc0, 0xc5, 0x38, 0x9a);
		public static Guid SyncResults = new Guid(0x289a9a43, 0xbe44, 0x4057, 0xa4, 0x1b, 0x58, 0x7a, 0x76, 0xd7, 0xe7, 0xf9);
		public static Guid RecycleBin = new Guid(0xB7534046, 0x3ECB, 0x4C18, 0xBE, 0x4E, 0x64, 0xCD, 0x4C, 0xB7, 0xD6, 0xAC);
		public static Guid Fonts = new Guid(0xFD228CB7, 0xAE11, 0x4AE3, 0x86, 0x4C, 0x16, 0xF3, 0x91, 0x0A, 0xB8, 0xFE);
		public static Guid Startup = new Guid(0xB97D20BB, 0xF46A, 0x4C97, 0xBA, 0x10, 0x5E, 0x36, 0x08, 0x43, 0x08, 0x54);
		public static Guid Programs = new Guid(0xA77F5D77, 0x2E2B, 0x44C3, 0xA6, 0xA2, 0xAB, 0xA6, 0x01, 0x05, 0x4A, 0x51);
		public static Guid StartMenu = new Guid(0x625B53C3, 0xAB48, 0x4EC1, 0xBA, 0x1F, 0xA1, 0xEF, 0x41, 0x46, 0xFC, 0x19);
		public static Guid Recent = new Guid(0xAE50C081, 0xEBD2, 0x438A, 0x86, 0x55, 0x8A, 0x09, 0x2E, 0x34, 0x98, 0x7A);
		public static Guid SendTo = new Guid(0x8983036C, 0x27C0, 0x404B, 0x8F, 0x08, 0x10, 0x2D, 0x10, 0xDC, 0xFD, 0x74);
		public static Guid Documents = new Guid(0xFDD39AD0, 0x238F, 0x46AF, 0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7);
		public static Guid Favorites = new Guid(0x1777F761, 0x68AD, 0x4D8A, 0x87, 0xBD, 0x30, 0xB7, 0x59, 0xFA, 0x33, 0xDD);
		public static Guid NetHood = new Guid(0xC5ABBF53, 0xE17F, 0x4121, 0x89, 0x00, 0x86, 0x62, 0x6F, 0xC2, 0xC9, 0x73);
		public static Guid PrintHood = new Guid(0x9274BD8D, 0xCFD1, 0x41C3, 0xB3, 0x5E, 0xB1, 0x3F, 0x55, 0xA7, 0x58, 0xF4);
		public static Guid Templates = new Guid(0xA63293E8, 0x664E, 0x48DB, 0xA0, 0x79, 0xDF, 0x75, 0x9E, 0x05, 0x09, 0xF7);
		public static Guid CommonStartup = new Guid(0x82A5EA35, 0xD9CD, 0x47C5, 0x96, 0x29, 0xE1, 0x5D, 0x2F, 0x71, 0x4E, 0x6E);
		public static Guid CommonPrograms = new Guid(0x0139D44E, 0x6AFE, 0x49F2, 0x86, 0x90, 0x3D, 0xAF, 0xCA, 0xE6, 0xFF, 0xB8);
		public static Guid CommonStartMenu = new Guid(0xA4115719, 0xD62E, 0x491D, 0xAA, 0x7C, 0xE7, 0x4B, 0x8B, 0xE3, 0xB0, 0x67);
		public static Guid PublicDesktop = new Guid(0xC4AA340D, 0xF20F, 0x4863, 0xAF, 0xEF, 0xF8, 0x7E, 0xF2, 0xE6, 0xBA, 0x25);
		public static Guid ProgramData = new Guid(0x62AB5D82, 0xFDC1, 0x4DC3, 0xA9, 0xDD, 0x07, 0x0D, 0x1D, 0x49, 0x5D, 0x97);
		public static Guid CommonTemplates = new Guid(0xB94237E7, 0x57AC, 0x4347, 0x91, 0x51, 0xB0, 0x8C, 0x6C, 0x32, 0xD1, 0xF7);
		public static Guid PublicDocuments = new Guid(0xED4824AF, 0xDCE4, 0x45A8, 0x81, 0xE2, 0xFC, 0x79, 0x65, 0x08, 0x36, 0x34);
		public static Guid RoamingAppData = new Guid(0x3EB685DB, 0x65F9, 0x4CF6, 0xA0, 0x3A, 0xE3, 0xEF, 0x65, 0x72, 0x9F, 0x3D);
		public static Guid LocalAppData = new Guid(0xF1B32785, 0x6FBA, 0x4FCF, 0x9D, 0x55, 0x7B, 0x8E, 0x7F, 0x15, 0x70, 0x91);
		public static Guid LocalAppDataLow = new Guid(0xA520A1A4, 0x1780, 0x4FF6, 0xBD, 0x18, 0x16, 0x73, 0x43, 0xC5, 0xAF, 0x16);
		public static Guid InternetCache = new Guid(0x352481E8, 0x33BE, 0x4251, 0xBA, 0x85, 0x60, 0x07, 0xCA, 0xED, 0xCF, 0x9D);
		public static Guid Cookies = new Guid(0x2B0F765D, 0xC0E9, 0x4171, 0x90, 0x8E, 0x08, 0xA6, 0x11, 0xB8, 0x4F, 0xF6);
		public static Guid History = new Guid(0xD9DC8A3B, 0xB784, 0x432E, 0xA7, 0x81, 0x5A, 0x11, 0x30, 0xA7, 0x59, 0x63);
		public static Guid System = new Guid(0x1AC14E77, 0x02E7, 0x4E5D, 0xB7, 0x44, 0x2E, 0xB1, 0xAE, 0x51, 0x98, 0xB7);
		public static Guid SystemX86 = new Guid(0xD65231B0, 0xB2F1, 0x4857, 0xA4, 0xCE, 0xA8, 0xE7, 0xC6, 0xEA, 0x7D, 0x27);
		public static Guid Windows = new Guid(0xF38BF404, 0x1D43, 0x42F2, 0x93, 0x05, 0x67, 0xDE, 0x0B, 0x28, 0xFC, 0x23);
		public static Guid Profile = new Guid(0x5E6C858F, 0x0E22, 0x4760, 0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73);
		public static Guid Pictures = new Guid(0x33E28130, 0x4E1E, 0x4676, 0x83, 0x5A, 0x98, 0x39, 0x5C, 0x3B, 0xC3, 0xBB);
		public static Guid ProgramFilesX86 = new Guid(0x7C5A40EF, 0xA0FB, 0x4BFC, 0x87, 0x4A, 0xC0, 0xF2, 0xE0, 0xB9, 0xFA, 0x8E);
		public static Guid ProgramFilesCommonX86 = new Guid(0xDE974D24, 0xD9C6, 0x4D3E, 0xBF, 0x91, 0xF4, 0x45, 0x51, 0x20, 0xB9, 0x17);
		public static Guid ProgramFilesX64 = new Guid(0x6d809377, 0x6af0, 0x444b, 0x89, 0x57, 0xa3, 0x77, 0x3f, 0x02, 0x20, 0x0e);
		public static Guid ProgramFilesCommonX64 = new Guid(0x6365d5a7, 0xf0d, 0x45e5, 0x87, 0xf6, 0xd, 0xa5, 0x6b, 0x6a, 0x4f, 0x7d);
		public static Guid ProgramFiles = new Guid(0x905e63b6, 0xc1bf, 0x494e, 0xb2, 0x9c, 0x65, 0xb7, 0x32, 0xd3, 0xd2, 0x1a);
		public static Guid ProgramFilesCommon = new Guid(0xF7F1ED05, 0x9F6D, 0x47A2, 0xAA, 0xAE, 0x29, 0xD3, 0x17, 0xC6, 0xF0, 0x66);
		public static Guid AdminTools = new Guid(0x724EF170, 0xA42D, 0x4FEF, 0x9F, 0x26, 0xB6, 0x0E, 0x84, 0x6F, 0xBA, 0x4F);
		public static Guid CommonAdminTools = new Guid(0xD0384E7D, 0xBAC3, 0x4797, 0x8F, 0x14, 0xCB, 0xA2, 0x29, 0xB3, 0x92, 0xB5);
		public static Guid Music = new Guid(0x4BD8D571, 0x6D19, 0x48D3, 0xBE, 0x97, 0x42, 0x22, 0x20, 0x08, 0x0E, 0x43);
		public static Guid Videos = new Guid(0x18989B1D, 0x99B5, 0x455B, 0x84, 0x1C, 0xAB, 0x7C, 0x74, 0xE4, 0xDD, 0xFC);
		public static Guid PublicPictures = new Guid(0xB6EBFB86, 0x6907, 0x413C, 0x9A, 0xF7, 0x4F, 0xC2, 0xAB, 0xF0, 0x7C, 0xC5);
		public static Guid PublicMusic = new Guid(0x3214FAB5, 0x9757, 0x4298, 0xBB, 0x61, 0x92, 0xA9, 0xDE, 0xAA, 0x44, 0xFF);
		public static Guid PublicVideos = new Guid(0x2400183A, 0x6185, 0x49FB, 0xA2, 0xD8, 0x4A, 0x39, 0x2A, 0x60, 0x2B, 0xA3);
		public static Guid ResourceDir = new Guid(0x8AD10C31, 0x2ADB, 0x4296, 0xA8, 0xF7, 0xE4, 0x70, 0x12, 0x32, 0xC9, 0x72);
		public static Guid LocalizedResourcesDir = new Guid(0x2A00375E, 0x224C, 0x49DE, 0xB8, 0xD1, 0x44, 0x0D, 0xF7, 0xEF, 0x3D, 0xDC);
		public static Guid CommonOEMLinks = new Guid(0xC1BAE2D0, 0x10DF, 0x4334, 0xBE, 0xDD, 0x7A, 0xA2, 0x0B, 0x22, 0x7A, 0x9D);
		public static Guid CDBurning = new Guid(0x9E52AB10, 0xF80D, 0x49DF, 0xAC, 0xB8, 0x43, 0x30, 0xF5, 0x68, 0x78, 0x55);
		public static Guid UserProfiles = new Guid(0x0762D272, 0xC50A, 0x4BB0, 0xA3, 0x82, 0x69, 0x7D, 0xCD, 0x72, 0x9B, 0x80);
		public static Guid Playlists = new Guid(0xDE92C1C7, 0x837F, 0x4F69, 0xA3, 0xBB, 0x86, 0xE6, 0x31, 0x20, 0x4A, 0x23);
		public static Guid SamplePlaylists = new Guid(0x15CA69B3, 0x30EE, 0x49C1, 0xAC, 0xE1, 0x6B, 0x5E, 0xC3, 0x72, 0xAF, 0xB5);
		public static Guid SampleMusic = new Guid(0xB250C668, 0xF57D, 0x4EE1, 0xA6, 0x3C, 0x29, 0x0E, 0xE7, 0xD1, 0xAA, 0x1F);
		public static Guid SamplePictures = new Guid(0xC4900540, 0x2379, 0x4C75, 0x84, 0x4B, 0x64, 0xE6, 0xFA, 0xF8, 0x71, 0x6B);
		public static Guid SampleVideos = new Guid(0x859EAD94, 0x2E85, 0x48AD, 0xA7, 0x1A, 0x09, 0x69, 0xCB, 0x56, 0xA6, 0xCD);
		public static Guid PhotoAlbums = new Guid(0x69D2CF90, 0xFC33, 0x4FB7, 0x9A, 0x0C, 0xEB, 0xB0, 0xF0, 0xFC, 0xB4, 0x3C);
		public static Guid Public = new Guid(0xDFDF76A2, 0xC82A, 0x4D63, 0x90, 0x6A, 0x56, 0x44, 0xAC, 0x45, 0x73, 0x85);
		public static Guid ChangeRemovePrograms = new Guid(0xdf7266ac, 0x9274, 0x4867, 0x8d, 0x55, 0x3b, 0xd6, 0x61, 0xde, 0x87, 0x2d);
		public static Guid AppUpdates = new Guid(0xa305ce99, 0xf527, 0x492b, 0x8b, 0x1a, 0x7e, 0x76, 0xfa, 0x98, 0xd6, 0xe4);
		public static Guid AddNewPrograms = new Guid(0xde61d971, 0x5ebc, 0x4f02, 0xa3, 0xa9, 0x6c, 0x82, 0x89, 0x5e, 0x5c, 0x04);
		public static Guid Downloads = new Guid(0x374de290, 0x123f, 0x4565, 0x91, 0x64, 0x39, 0xc4, 0x92, 0x5e, 0x46, 0x7b);
		public static Guid PublicDownloads = new Guid(0x3d644c9b, 0x1fb8, 0x4f30, 0x9b, 0x45, 0xf6, 0x70, 0x23, 0x5f, 0x79, 0xc0);
		public static Guid SavedSearches = new Guid(0x7d1d3a04, 0xdebb, 0x4115, 0x95, 0xcf, 0x2f, 0x29, 0xda, 0x29, 0x20, 0xda);
		public static Guid QuickLaunch = new Guid(0x52a4f021, 0x7b75, 0x48a9, 0x9f, 0x6b, 0x4b, 0x87, 0xa2, 0x10, 0xbc, 0x8f);
		public static Guid Contacts = new Guid(0x56784854, 0xc6cb, 0x462b, 0x81, 0x69, 0x88, 0xe3, 0x50, 0xac, 0xb8, 0x82);
		public static Guid SidebarParts = new Guid(0xa75d362e, 0x50fc, 0x4fb7, 0xac, 0x2c, 0xa8, 0xbe, 0xaa, 0x31, 0x44, 0x93);
		public static Guid SidebarDefaultParts = new Guid(0x7b396e54, 0x9ec5, 0x4300, 0xbe, 0xa, 0x24, 0x82, 0xeb, 0xae, 0x1a, 0x26);
		public static Guid TreeProperties = new Guid(0x5b3749ad, 0xb49f, 0x49c1, 0x83, 0xeb, 0x15, 0x37, 0x0f, 0xbd, 0x48, 0x82);
		public static Guid PublicGameTasks = new Guid(0xdebf2536, 0xe1a8, 0x4c59, 0xb6, 0xa2, 0x41, 0x45, 0x86, 0x47, 0x6a, 0xea);
		public static Guid GameTasks = new Guid(0x54fae61, 0x4dd8, 0x4787, 0x80, 0xb6, 0x9, 0x2, 0x20, 0xc4, 0xb7, 0x0);
		public static Guid SavedGames = new Guid(0x4c5c32ff, 0xbb9d, 0x43b0, 0xb5, 0xb4, 0x2d, 0x72, 0xe5, 0x4e, 0xaa, 0xa4);
		public static Guid Games = new Guid(0xcac52c1a, 0xb53d, 0x4edc, 0x92, 0xd7, 0x6b, 0x2e, 0x8a, 0xc1, 0x94, 0x34);
		public static Guid RecordedTV = new Guid(0xbd85e001, 0x112e, 0x431e, 0x98, 0x3b, 0x7b, 0x15, 0xac, 0x09, 0xff, 0xf1);
		public static Guid SearchMapi = new Guid(0x98ec0e18, 0x2098, 0x4d44, 0x86, 0x44, 0x66, 0x97, 0x93, 0x15, 0xa2, 0x81);
		public static Guid SearchCsc = new Guid(0xee32e446, 0x31ca, 0x4aba, 0x81, 0x4f, 0xa5, 0xeb, 0xd2, 0xfd, 0x6d, 0x5e);
		public static Guid Links = new Guid(0xbfb9d5e0, 0xc6a9, 0x404c, 0xb2, 0xb2, 0xae, 0x6d, 0xb6, 0xaf, 0x49, 0x68);
		public static Guid UsersFiles = new Guid(0xf3ce0f7c, 0x4901, 0x4acc, 0x86, 0x48, 0xd5, 0xd4, 0x4b, 0x04, 0xef, 0x8f);
		public static Guid SearchHome = new Guid(0x190337d1, 0xb8ca, 0x4121, 0xa6, 0x39, 0x6d, 0x47, 0x2d, 0x16, 0x97, 0x2a);
		public static Guid OriginalImages = new Guid(0x2C36C0AA, 0x5812, 0x4b87, 0xbf, 0xd0, 0x4c, 0xd0, 0xdf, 0xb1, 0x9b, 0x39);
	}
 
	[ComImport, Guid("8BE2D872-86AA-4d47-B776-32CCA40C7018" /*"44BEAAEC-24F4-4E90-B3F0-23D258FBB146"*/), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
	interface IKnownFolderManager
	{
		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void FolderIdFromCsidl([In] int nCsidl, out Guid pfid);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void FolderIdToCsidl([In] ref Guid rfid, out int pnCsidl);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetFolderIds([Out] IntPtr ppKFId, [In, Out] ref uint pCount);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetFolder([In] ref Guid rfid, [MarshalAs(UnmanagedType.Interface)] out IKnownFolder ppkf);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void GetFolderByName([In, MarshalAs(UnmanagedType.LPWStr)] string pszCanonicalName,
					 [MarshalAs(UnmanagedType.Interface)] out IKnownFolder ppkf);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void RegisterFolder([In] ref Guid rfid, [In] ref KNOWNFOLDER_DEFINITION pKFD);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void UnregisterFolder([In] ref Guid rfid);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void FindFolderFromPath([In, MarshalAs(UnmanagedType.LPWStr)] string pszPath,
					[In] KnownFolderFindMode mode,
					[MarshalAs(UnmanagedType.Interface)] out IKnownFolder ppkf);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void FindFolderFromIDList([In] IntPtr pidl, [MarshalAs(UnmanagedType.Interface)] out IKnownFolder ppkf);

		//[MethodImpl(MethodImplOptions.InternalCall, MethodCodeType = MethodCodeType.Runtime)]
		void Redirect([In] ref Guid rfid, [In] IntPtr hwnd, [In] KnownFolderRedirectFlags flags,
			  [In, MarshalAs(UnmanagedType.LPWStr)] string pszTargetPath, [In] uint cFolders,
		   [In] ref Guid pExclusion, [MarshalAs(UnmanagedType.LPWStr)] out string ppszError);
	}
	[ComImport, Guid("4df0c730-df9d-4ae3-9153-aa6b82e9795a")]

	internal class KnownFolderManager
	{

	}
}
