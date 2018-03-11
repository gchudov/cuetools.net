using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Collections.ObjectModel;
using CUETools.CDImage;
using CUETools.Processor;

namespace CUERipper.WPF
{
	public class DriveInfo
	{
		public string Name { get; set; }
		public string Icon { get; set; }
	}

	/// <summary>
	/// Interaction logic for Window1.xaml
	/// </summary>
	public partial class Window1 : Window
	{
		private CUEControls.ShellIconMgr m_icon_mgr;

		public Window1()
		{
			InitializeComponent();
			m_icon_mgr = new CUEControls.ShellIconMgr();
		}

		ObservableCollection<CUEMetadataEntry> _Releases = new ObservableCollection<CUEMetadataEntry>();
		ObservableCollection<DriveInfo> _Drives = new ObservableCollection<DriveInfo>();
		public ObservableCollection<DriveInfo> Drives { get { return _Drives; } }
		public ObservableCollection<CUEMetadataEntry> Releases { get { return _Releases; } }

		public static DependencyProperty SelectedDriveProperty = DependencyProperty.Register("SelectedDrive", typeof(DriveInfo), typeof(Window1));
		public DriveInfo SelectedDrive
		{
			get { return ((DriveInfo)(base.GetValue(SelectedDriveProperty))); }
			set { base.SetValue(SelectedDriveProperty, value); }
		}

		public static DependencyProperty SelectedReleaseProperty = DependencyProperty.Register("SelectedRelease", typeof(CUEMetadataEntry), typeof(Window1));
		public CUEMetadataEntry SelectedRelease
		{
			get { return ((CUEMetadataEntry)(base.GetValue(SelectedReleaseProperty))); }
			set { base.SetValue(SelectedReleaseProperty, value); }
		}

		private void Window_Loaded(object sender, RoutedEventArgs e)
		{
			Drives.Add(new DriveInfo { Name = "aa", Icon = "/CUERipper.WPF;component/musicbrainz.ico" });
			Drives.Add(new DriveInfo { Name = "cc", Icon = "/CUERipper.WPF;component/freedb16.png" });
			Drives.Add(new DriveInfo { Name = "ee", Icon = "ff" });
			SelectedDrive = Drives[0];

			CDImageLayout toc = new CDImageLayout(2, 2, 1, "0 10000 20000");
			Releases.Add(new CUEMetadataEntry(toc, "/CUERipper.WPF;component/musicbrainz.ico"));
			Releases[0].metadata.Artist = "Mike Oldfield";
			Releases[0].metadata.Title = "Amarok";
			Releases[0].metadata.Tracks[0].Artist = "Mike Oldfield";
			Releases[0].metadata.Tracks[0].Title = "Amarok 01";
			Releases[0].metadata.Tracks[1].Artist = "Mike Oldfield";
			Releases[0].metadata.Tracks[1].Title = "Amarok 02";
			Releases.Add(new CUEMetadataEntry(toc, "/CUERipper.WPF;component/freedb16.png"));
			SelectedRelease = Releases[0];
		}
	}
}
