using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
//using System.Windows.Shapes;
using CUETools.Codecs.BDLPCM;
using CUETools.CDImage;
using CUETools.CTDB;
//using Microsoft.Win32;

namespace CUETools.eac3ui
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            textBoxSource.Text = Properties.Settings.Default.SourceFolder;
            textBoxDestination.Text = Properties.Settings.Default.DestinationFolder;
        }

        private void ButtonBrowseSource_Click(object sender, RoutedEventArgs e)
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                dialog.SelectedPath = textBoxSource.Text;
                dialog.ShowNewFolderButton = false;
                System.Windows.Forms.DialogResult result = dialog.ShowDialog();
                if (result == System.Windows.Forms.DialogResult.OK)
                    textBoxSource.Text = dialog.SelectedPath;
            }
            //OpenFileDialog openFileDialog = new OpenFileDialog();
            //if (openFileDialog.ShowDialog() == true)
            //    textBoxSource.Text = openFileDialog.FileName;
        }

        private void buttonBrowseDestination_Click(object sender, RoutedEventArgs e)
        {
            using (var dialog = new System.Windows.Forms.FolderBrowserDialog())
            {
                dialog.SelectedPath = textBoxDestination.Text;
                System.Windows.Forms.DialogResult result = dialog.ShowDialog();
                if (result == System.Windows.Forms.DialogResult.OK)
                    textBoxDestination.Text = dialog.SelectedPath;
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            Properties.Settings.Default.SourceFolder = textBoxSource.Text;
            Properties.Settings.Default.DestinationFolder = textBoxDestination.Text;
            Properties.Settings.Default.Save();
        }

        private void textBoxSource_TextChanged(object sender, TextChangedEventArgs e)
        {
            var titleSets = new List<MPLSReader>();
            IEnumerable<string> playlists = null;
            try
            {
                playlists = Directory.EnumerateFiles(Path.Combine(Path.Combine(textBoxSource.Text, "BDMV"), "PLAYLIST"));
            }
            catch
            {
            }
            if (playlists != null)
                foreach (var playlist in playlists)
                {
                    titleSets.Add(new MPLSReader(playlist, null));
                }
            cmbTitleSet.ItemsSource = titleSets;
            cmbTitleSet.SelectedIndex = 0;
        }

        private void cmbTitleSet_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            cmbMetadata.ItemsSource = new List<CTDBResponseMeta>();

            var audios = new List<MPLSStream>();
            if (e.AddedItems.Count == 1)
            {
                MPLSReader rdr = e.AddedItems[0] as MPLSReader;
                rdr.MPLSHeader.play_item.ForEach(i => i.audio.ForEach(v => { if (!audios.Exists(v1 => v1.pid == v.pid)) audios.Add(v); }));

                var chapters = rdr.Chapters;
                string strtoc = "";
                for (int i = 0; i < chapters.Count; i++)
                    strtoc += string.Format(" {0}", chapters[i] / 600);
                strtoc = strtoc.Substring(1);
                CDImageLayout toc = new CDImageLayout(strtoc);
                var ctdb = new CUEToolsDB(toc, null);
                //Console.Error.WriteLine("Contacting CTDB...");
                ctdb.ContactDB(null, "CUETools.eac3to 2.1.7", "", false, true, CTDBMetadataSearch.Extensive);
                cmbMetadata.ItemsSource = ctdb.Metadata;
                cmbMetadata.SelectedIndex = 0;
            }
            cmbAudioTrack.ItemsSource = audios;
            cmbAudioTrack.SelectedIndex = 0;
        }

        private void cmbAudioTrack_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (e.AddedItems.Count == 1)
            {
                MPLSStream stream = (MPLSStream)(e.AddedItems[0]);
            }
        }
    }
}
