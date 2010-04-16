using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using System.Diagnostics;
using CUEControls;
using CUETools.Codecs;
using CUETools.DSP.Mixer;
using CUETools.Processor;

namespace CUEPlayer
{
	public partial class frmCUEPlayer : Form
	{
		private ShellIconMgr _icon_mgr;
		private CUEConfig _config;
		DataSet1TableAdapters.PlaylistTableAdapter adapterPlayList = new DataSet1TableAdapters.PlaylistTableAdapter();
		private DataSet1 dataSet = new DataSet1();
		private Thread mixThread;
		private MixingSource _mixer;

		internal Playlist wndPlaylist
		{
			get
			{
				return playlist;
			}
		}

		internal DataSet1 DataSet
		{
			get
			{
				return dataSet;
			}
		}

		internal CUEConfig Config
		{
			get
			{
				return _config;
			}
		}

		internal ShellIconMgr IconMgr
		{
			get
			{
				return _icon_mgr;
			}
		}

		public MixingSource Mixer
		{
			get
			{
				return _mixer;
			}
		}

		public frmCUEPlayer()
		{
			InitializeComponent();
			_icon_mgr = new ShellIconMgr();
			_config = new CUEConfig();
			_config.separateDecodingThread = false;
		}

		internal Deck deckA = new Deck(0, "A");
		internal Deck deckB = new Deck(1, "B");
		internal Output outputA = new Output();
		internal Browser browser = new Browser();
		internal Playlist playlist = new Playlist();

		private void frmCUEPlayer_Load(object sender, EventArgs e)
		{
			if (Properties.Settings.Default.AppSettings == null)
			{
				Properties.Settings.Default.AppSettings = new CUEPlayerSettings();
				Properties.Settings.Default.AppSettings.IcecastServers.Add(new CUETools.Codecs.Icecast.IcecastSettingsData());
			}

			//System.Data.SqlServerCe.SqlCeDataAdapter ad = new System.Data.SqlServerCe.SqlCeDataAdapter();
			//ad.SelectCommand = new System.Data.SqlServerCe.SqlCeCommand("SELECT * FROM Playlist WHERE track=1", adapterPlayList.Connection);
			//ad.Fill(dataSet.Playlist);
			adapterPlayList.Fill(dataSet.Playlist);

			_mixer = new MixingSource(new AudioPCMConfig(32, 2, 44100), 100, 2);

			outputA.Init(this);
			browser.Init(this);
			playlist.Init(this);
			deckB.Init(this, null);
			deckA.Init(this, deckB);
			Icecast icecast = new Icecast();
			icecast.Init(this);
			//LayoutMdi(MdiLayout.TileHorizontal);

			browser.Location = new Point(0, 0);
			browser.Height = ClientRectangle.Height - 5 - menuStrip1.Height;
			playlist.Location = new Point(browser.Location.X + browser.Width, 0);
			playlist.Height = ClientRectangle.Height - 5 - menuStrip1.Height;
			deckA.Location = new Point(playlist.Location.X + playlist.Width, 0);
			deckB.Location = new Point(playlist.Location.X + playlist.Width, deckA.Height);
			outputA.Location = new Point(deckA.Location.X + deckA.Width, 0);
			icecast.Location = new Point(deckA.Location.X + deckA.Width, outputA.Height);

			mixThread = new Thread(MixThread);
			mixThread.Priority = ThreadPriority.AboveNormal;
			mixThread.IsBackground = true;
			mixThread.Name = "Mixer";
			mixThread.Start();
		}

		private void frmCUEPlayer_FormClosing(object sender, FormClosingEventArgs e)
		{
			try
			{
				int rowsAffected = adapterPlayList.Update(dataSet.Playlist);
			}
			catch (Exception ex)
			{
				System.Diagnostics.Trace.WriteLine(ex.Message);
			}
		}

		private void MixThread()
		{
			AudioBuffer result = new AudioBuffer(_mixer.PCM, _mixer.BufferSize);
			while (true)
			{
				_mixer.Read(result, -1);
			}
		}

		public event EventHandler<UpdateMetadataEvent> updateMetadata;

		public void UpdateMetadata(string artist, string title)
		{
			UpdateMetadataEvent e = new UpdateMetadataEvent();
			e.artist = artist;
			e.title = title;
			if (updateMetadata != null)
				updateMetadata(this, e);
		}

		private void icecastToolStripMenuItem_Click(object sender, EventArgs e)
		{
			Icecast icecast = new Icecast();
			icecast.Init(this);
		}
	}

	public class UpdateMetadataEvent: EventArgs
	{
		public string artist;
		public string title;
	}

	public class CUEPlayerSettings
	{
		private BindingList<CUETools.Codecs.Icecast.IcecastSettingsData> icecastServers;

		public CUEPlayerSettings()
		{
			icecastServers = new BindingList<CUETools.Codecs.Icecast.IcecastSettingsData>();
		}

		public BindingList<CUETools.Codecs.Icecast.IcecastSettingsData> IcecastServers
		{
			get
			{
				return icecastServers;
			}
			set
			{
				icecastServers = value;
			}
		}
	}
}
