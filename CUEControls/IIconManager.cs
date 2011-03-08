using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Windows.Forms;
using System.IO;

namespace CUEControls
{
	public interface IIconManager
	{
		#region public properties
		/// <summary>
		/// Return the image list that contains the icons found to date.
		/// </summary>
		ImageList ImageList { get; }
		#endregion
		#region public methods
		int GetIconIndex(FileSystemInfo filename, bool open);

		/// <summary>
		/// Get the icon index associated with a given filename
		/// </summary>
		/// <param name="filename">the filename of interest</param>
		/// <param name="open">if true, the file is "open", most useful for folders</param>
		/// <returns>the index into the image list for the icon associated with this file</returns>
		int GetIconIndex(ExtraSpecialFolder folder, bool open);
		int GetIconIndex(string filename);
		void SetExtensionIcon(string extension, Image icon);
		void SetExtensionIcon(string extension, Icon icon);
		string GetFolderPath(ExtraSpecialFolder folder);
		void SetFolderPath(ExtraSpecialFolder folder, string path);
		string GetDisplayName(FileSystemInfo filename);
		string GetDisplayName(ExtraSpecialFolder folder);
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

	public unsafe class MonoIconMgr : IIconManager
	{
		#region private variables
		private ImageList m_image_list;
		private IDictionary<int, int> m_index_map;
		private IDictionary<string, int> m_extension_map;
		#endregion

		#region constructor
		/// <summary>
		/// This creates a new shell icon manager.
		/// </summary>
		public MonoIconMgr()
		{
			m_image_list = new ImageList();

			m_index_map = new Dictionary<int, int>();
			m_extension_map = new Dictionary<string, int>();
			m_image_list.ImageSize = new Size(16, 16);
			m_image_list.ColorDepth = ColorDepth.Depth32Bit;
			m_image_list.Images.Add(Properties.Resources.folder);
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
			//iIcon = MapIcon(System.Drawing.SystemIcons.Application, info.iIcon);
			//DestroyIcon(info.hIcon);
			return 0;
		}

		/// <summary>
		/// Get the icon index associated with a given filename
		/// </summary>
		/// <param name="filename">the filename of interest</param>
		/// <param name="open">if true, the file is "open", most useful for folders</param>
		/// <returns>the index into the image list for the icon associated with this file</returns>
		public int GetIconIndex(ExtraSpecialFolder folder, bool open)
		{
			return 0;
		}

		public int GetIconIndex(string filename)
		{
			int iIcon;
			if (m_extension_map.TryGetValue(Path.GetExtension(filename).ToLower(), out iIcon))
				return iIcon;
			if (m_extension_map.TryGetValue(".wav", out iIcon))
				return iIcon;
			return 0;
		}
		#endregion

		#region private methods

		//private int MapIcon(IntPtr hIcon, int iIcon)
		//{
		//    int index = 0;
		//    if (!m_index_map.TryGetValue(iIcon, out index))
		//    {
		//        m_image_list.Images.Add(Icon.FromHandle(hIcon));
		//        index = m_image_list.Images.Count - 1;
		//        m_index_map.Add(iIcon, index);
		//    }
		//    return index;
		//}

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
			switch (folder)
			{
				case ExtraSpecialFolder.MyComputer:
					return "/";
			}
			return Environment.GetFolderPath((Environment.SpecialFolder)folder);
		}

		public void SetFolderPath(ExtraSpecialFolder folder, string path)
		{
			throw new Exception("SetFolderPath not supported");
		}

		public string GetDisplayName(FileSystemInfo filename)
		{
			return filename.Name;
		}

		public string GetDisplayName(ExtraSpecialFolder folder)
		{
			switch (folder)
			{
				case ExtraSpecialFolder.MyComputer:
					return "/";
			}
			return Path.GetFileName(Environment.GetFolderPath((Environment.SpecialFolder) folder));
		}
		#endregion
	}
}
