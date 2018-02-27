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
using System.ComponentModel;
using Krystalware.UploadHelper;
using CUETools.Codecs;
using CUETools.Codecs.FLAKE;
using CUETools.Processor;
using System.Collections.ObjectModel;
//using Microsoft.Win32;

namespace BluTools
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            metaresults = new ObservableCollection<CUEMetadataEntry>();
            filterShort = 900;
            filterRepeats = 2;
            filterDups = true;
            InitializeComponent();
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
                playlists = Directory.EnumerateFiles(Path.Combine(Path.Combine(textBoxSource.Text, "BDMV"), "PLAYLIST")).OrderBy(f => f);
            }
            catch
            {
            }
            if (playlists != null)
                foreach (var playlist in playlists)
                {
                    var title = new MPLSReader(playlist, null);
                    if (filterDups)
                    {
                        if (titleSets.Exists(title2 =>
                            title.MPLSHeader.play_item.Count == title2.MPLSHeader.play_item.Count &&
                            Enumerable.Range(0, title.MPLSHeader.play_item.Count).ToList().TrueForAll(i =>
                                {
                                    var i1 = title.MPLSHeader.play_item[i];
                                    var i2 = title2.MPLSHeader.play_item[i];
                                    return i1.clip_id == i2.clip_id && i1.in_time == i2.in_time && i1.out_time == i2.out_time;
                                })
                            ))
                            continue;
                    }

                    if (filterRepeats > 1)
                    {
                        if (title.MPLSHeader.play_item.GroupBy(clip => clip.clip_id).Select(grp => grp.Count()).Max() > filterRepeats)
                            continue;
                    }

                    if (filterShort > 0)
                    {
                        if (title.Duration.TotalSeconds < filterShort)
                            continue;
                    }

                    titleSets.Add(title);
                }
            cmbTitleSet.ItemsSource = titleSets;
            cmbTitleSet.SelectedIndex = 0;
        }

        private void cmbTitleSet_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            cmbMetadata.ItemsSource = new List<CTDBResponseMeta>();
            ctdb = null;

            var audios = new List<MPLSStream>();
            if (e.AddedItems.Count == 1)
            {
                MPLSReader rdr = e.AddedItems[0] as MPLSReader;
                rdr.MPLSHeader.play_item.ForEach(i => i.audio.ForEach(v => { if (!audios.Exists(v1 => v1.pid == v.pid)) audios.Add(v); }));

                var chapters = rdr.Chapters;
                if (chapters.Count > 2)
                {
                    string strtoc = "";
                    for (int i = 0; i < chapters.Count; i++)
                        strtoc += string.Format(" {0}", chapters[i] / 600);
                    strtoc = strtoc.Substring(1);
                    CDImageLayout toc = new CDImageLayout(strtoc);
                    ctdb = new CUEToolsDB(toc, null);
                    workerCtdb = new BackgroundWorker();
                    workerCtdb.DoWork += workerCtdb_DoWork;
                    workerCtdb.RunWorkerAsync();
                }
            }
            cmbAudioTrack.ItemsSource = audios;
            cmbAudioTrack.SelectedIndex = 0;
        }

        void workerCtdb_DoWork(object sender, DoWorkEventArgs e)
        {
            //Console.Error.WriteLine("Contacting CTDB...");
            this.Dispatcher.Invoke(() =>
            {
                pbStatus.Visibility = Visibility.Visible;
                pbStatus.IsIndeterminate = true;
            });
            //ctdb.UploadHelper.onProgress += worker_ctdbProgress;
            ctdb.ContactDB(null, "CUETools.eac3to 2.1.7", "", false, true, CTDBMetadataSearch.Extensive);
            this.Dispatcher.Invoke(() =>
            {
                //metaresults.RaiseListChangedEvents = false; 
                cmbMetadata.ItemsSource = null;
                metaresults.Clear();
                foreach (var m in ctdb.Metadata)
                {
                    var entry = new CUEMetadataEntry(ctdb.TOC, m.source);
                    entry.metadata.FillFromCtdb(m, entry.TOC.FirstAudio - 1);
                    metaresults.Add(entry);
                }
                //metaresults.RaiseListChangedEvents = true; 
                cmbMetadata.ItemsSource = metaresults;
                pbStatus.Visibility = Visibility.Collapsed;
                pbStatus.IsIndeterminate = false;
                cmbMetadata.SelectedIndex = 0;
            });
        }

        //private void worker_ctdbProgress(object sender, UploadProgressEventArgs args)
        //{
        //    this.Dispatcher.Invoke(() =>
        //    {
        //        pbStatus.Value = (int)args.percent;
        //    });
        //}

        private void cmbAudioTrack_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (e.AddedItems.Count == 1)
            {
                MPLSStream stream = (MPLSStream)(e.AddedItems[0]);
            }
        }

        BackgroundWorker workerCtdb;
        CUEToolsDB ctdb;

        BackgroundWorker workerExtract;
        CUEMetadataEntry metaresult;
        ObservableCollection<CUEMetadataEntry> metaresults;
        MPLSReader chosenReader;
        ushort pid;
        string outputFolderPath;
        string outputAudioPath;
        string outputCuePath;

        bool filterDups;
        int filterShort;
        int filterRepeats;

        private void buttonExtract_Click(object sender, RoutedEventArgs e)
        {
            if (cmbTitleSet.SelectedItem == null) return;
            pid = ((MPLSStream)cmbAudioTrack.SelectedItem).pid;
            chosenReader = cmbTitleSet.SelectedItem as MPLSReader;
            metaresult = cmbMetadata.SelectedItem as CUEMetadataEntry;
            outputFolderPath = Path.Combine(textBoxDestination.Text, metaresult != null ? 
                metaresult.metadata.Artist + " - " + metaresult.metadata.Year + " - " + metaresult.metadata.Title :
                Path.GetFileName(textBoxSource.Text) + "." + chosenReader.FileName + "." + pid.ToString());
            outputAudioPath = Path.Combine(outputFolderPath, metaresult != null ? metaresult.metadata.Artist + " - " + metaresult.metadata.Year + " - " + metaresult.metadata.Title + ".flac" : "image.flac");
            outputCuePath = Path.ChangeExtension(outputAudioPath, "cue");

            pbStatus.Visibility = Visibility.Visible;
            pbStatus.Value = 0.0;
            //pbStatus.IsIndeterminate = true;
            stackParams.IsEnabled = false;
            buttonExtract.IsEnabled = false;
            buttonExtract.Visibility = Visibility.Hidden;
            buttonStop.Visibility = Visibility.Visible;
            buttonStop.IsEnabled = true;

            workerExtract = new BackgroundWorker();
            workerExtract.WorkerSupportsCancellation = true;
            workerExtract.DoWork += workerExtract_DoWork;
            workerExtract.RunWorkerAsync();
        }

        void workerExtract_DoWork(object sender, DoWorkEventArgs e)
        {
            MPLSReader reader = null;
            try
            {
                reader = new MPLSReader(chosenReader.Path, null, pid);
                Directory.CreateDirectory(outputFolderPath);
                if (File.Exists(outputCuePath)) throw new Exception(string.Format("File \"{0}\" already exists", outputCuePath));
                if (File.Exists(outputAudioPath)) throw new Exception(string.Format("File \"{0}\" already exists", outputAudioPath));
                AudioBuffer buff = new AudioBuffer(reader, 0x10000);
                FlakeWriterSettings settings = new FlakeWriterSettings()
                {
                    PCM = reader.PCM,
                    Padding = 16536,
                    EncoderMode = "5"
                };
                if (ctdb != null)
                {
                    using (StreamWriter cueWriter = new StreamWriter(outputCuePath, false, Encoding.UTF8))
                    {
                        cueWriter.WriteLine("REM COMMENT \"{0}\"", "Created by CUETools.eac3to");
                        if (metaresult != null && metaresult.metadata.Year != "")
                            cueWriter.WriteLine("REM DATE {0}", metaresult.metadata.Year);
                        else
                            cueWriter.WriteLine("REM DATE XXXX");
                        if (metaresult != null)
                        {
                            cueWriter.WriteLine("PERFORMER \"{0}\"", metaresult.metadata.Artist);
                            cueWriter.WriteLine("TITLE \"{0}\"", metaresult.metadata.Title);
                        }
                        else
                        {
                            cueWriter.WriteLine("PERFORMER \"\"");
                            cueWriter.WriteLine("TITLE \"\"");
                        }
                        cueWriter.WriteLine("FILE \"{0}\" WAVE", Path.GetFileName(outputAudioPath));
                        var toc = ctdb.TOC;
                        for (int track = 1; track <= toc.TrackCount; track++)
                            if (toc[track].IsAudio)
                            {
                                cueWriter.WriteLine("  TRACK {0:00} AUDIO", toc[track].Number);
                                if (metaresult != null && metaresult.metadata.Tracks.Count >= toc[track].Number)
                                {
                                    cueWriter.WriteLine("    TITLE \"{0}\"", metaresult.metadata.Tracks[(int)toc[track].Number - 1].Title);
                                    if (metaresult.metadata.Tracks[(int)toc[track].Number - 1].Artist != "")
                                        cueWriter.WriteLine("    PERFORMER \"{0}\"", metaresult.metadata.Tracks[(int)toc[track].Number - 1].Artist);
                                }
                                else
                                {
                                    cueWriter.WriteLine("    TITLE \"\"");
                                }
                                if (toc[track].ISRC != null)
                                    cueWriter.WriteLine("    ISRC {0}", toc[track].ISRC);
                                for (int index = toc[track].Pregap > 0 ? 0 : 1; index <= toc[track].LastIndex; index++)
                                    cueWriter.WriteLine("    INDEX {0:00} {1}", index, toc[track][index].MSF);
                            }
                    }
                }
                var start = DateTime.Now;
                TimeSpan lastPrint = TimeSpan.FromMilliseconds(0);
                var writer = new FlakeWriter(outputAudioPath, settings);
                try
                {
                    while (reader.Read(buff, -1) != 0)
                    {
                        writer.Write(buff);
                        TimeSpan elapsed = DateTime.Now - start;
                        if ((elapsed - lastPrint).TotalMilliseconds > 60)
                        {
                            long length = Math.Max((long)(reader.Duration.TotalSeconds * reader.PCM.SampleRate), Math.Max(reader.Position, 1));
                            this.Dispatcher.Invoke(() =>
                            {
                                pbStatus.Value = 100.0 * reader.Position / length;
                            });
                            lastPrint = elapsed;
                        }
                        if (workerExtract.CancellationPending)
                        {
                            throw new Exception("aborted");
                        }
                    }
                }
                catch (Exception ex)
                {
                    writer.Delete();
                    try { File.Delete(outputCuePath); } catch (Exception) { }
                    throw ex;
                }
                writer.Close();
            }
            catch (Exception ex)
            {
                this.Dispatcher.Invoke(() =>
                {
                    MessageBox.Show(this, ex.Message, "Extraction failed");
                });
            }
            finally
            {
                if (reader != null) reader.Close();
            }

            this.Dispatcher.Invoke(() =>
            {
                pbStatus.Visibility = Visibility.Collapsed;
                //pbStatus.IsIndeterminate = false;
                stackParams.IsEnabled = true;
                buttonExtract.IsEnabled = true;
                buttonExtract.Visibility = Visibility.Visible;
                buttonStop.Visibility = Visibility.Hidden;
                buttonStop.IsEnabled = false;
            });
        }

        private void buttonStop_Click(object sender, RoutedEventArgs e)
        {
            workerExtract.CancelAsync();
            buttonStop.Visibility = Visibility.Hidden;
            buttonStop.IsEnabled = false;
        }

        private void Window_Initialized(object sender, EventArgs e)
        {
            textBoxSource.Text = Properties.Settings.Default.SourceFolder;
            textBoxDestination.Text = Properties.Settings.Default.DestinationFolder;
            cmbMetadata.ItemsSource = metaresults;
        }
    }
    public class CodingTypeToIcon : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            // TODO: ResourceDictionary?
            var image = Application.Current.MainWindow.Resources["coding_type_" + value.ToString()] as Image;
            return image == null ? null : image.Source;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class FormatTypeToIcon : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            var image = Application.Current.MainWindow.Resources["format_type_" + value.ToString()] as Image;
            return image == null ? null : image.Source;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class MetadataSourceToIcon : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            var image = Application.Current.MainWindow.Resources[value.ToString()] as Image;
            return image == null ? null : image.Source;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }
}
