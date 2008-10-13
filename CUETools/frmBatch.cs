using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;

namespace JDP
{
	public partial class frmBatch : Form
	{
		public frmBatch()
		{
			InitializeComponent();
			_config = new CUEConfig();
			_cueStyle = CUEStyle.SingleFile;
			_audioFormat = OutputAudioFormat.WAV;
			_accurateRip = true;
			_accurateOffset = false;
		}

		public string InputPath
		{
			get { return pathIn; }
			set { pathIn = value; }
		}
		public bool AccurateRip
		{
			get { return _accurateRip; }
			set { _accurateRip = value; }
		}
		public bool AccurateOffset
		{
			get { return _accurateOffset; }
			set { _accurateOffset = value; }
		}

		Thread _workThread;
		CUESheet _workClass;
		CUEConfig _config;
		CUEStyle _cueStyle;
		OutputAudioFormat _audioFormat;
		string pathIn;
		string pathOut;
		bool _accurateRip;
		bool _accurateOffset;
		DateTime _startedAt;

		public void SetStatus(string status, uint percentTrack, double percentDisk)
		{
			this.BeginInvoke((MethodInvoker)delegate()
			{
				if (percentDisk == 0)
				{
					_startedAt = DateTime.Now;
				}
				else if (percentDisk > 0.02)
				{
					TimeSpan span = DateTime.Now - _startedAt;
					TimeSpan eta = new TimeSpan ((long) (span.Ticks/percentDisk));
					Text = String.Format("{0}, ETA {1}:{2:00}.", status, (int)eta.TotalMinutes, eta.Seconds);
				} else
					Text = status;
				progressBar1.Value = (int)percentTrack;
				progressBar2.Value = (int)(percentDisk*100);
			});
		}

		private void WriteAudioFilesThread(object o)
		{
			CUESheet cueSheet = (CUESheet)o;

			try
			{
				cueSheet.WriteAudioFiles(Path.GetDirectoryName(pathOut), _cueStyle, new SetStatus(this.SetStatus));
				this.Invoke((MethodInvoker)delegate()
				{
					//if (_batchPaths.Count == 0)
					{
						//TimeSpan span = DateTime.Now - _startedAt;
						Text = "Done.";
						progressBar1.Value = 0;
						progressBar2.Value = 0;
						if (cueSheet.AccurateRip)
						{
							StringWriter sw = new StringWriter();
							cueSheet.GenerateAccurateRipLog(sw);
							textBox1.Text = sw.ToString();
							sw.Close();
							textBox1.Show();
						}
					}
				});
			}
			catch (StopException)
			{
				////_batchPaths.Clear();
				//this.Invoke((MethodInvoker)delegate()
				//{
				//    MessageBox.Show("Conversion was stopped.", "Stopped", MessageBoxButtons.OK,
				//        MessageBoxIcon.Exclamation);
				//    Close();
				//});

				this.Invoke((MethodInvoker)delegate()
				{
					Text = "Aborted.";
					progressBar1.Value = 0;
					progressBar2.Value = 0;
				});
			}
			catch (Exception ex)
			{
				this.Invoke((MethodInvoker)delegate()
				{
					//if (_batchPaths.Count == 0) SetupControls(false);
					//if (!ShowErrorMessage(ex))
					//{
					//    _batchPaths.Clear();
					//    SetupControls(false);
					//}
					Text = "Error: " + ex.Message;
				});
			}

			//if (_batchPaths.Count != 0)
			//{
			//    _batchPaths.RemoveAt(0);
			//    this.BeginInvoke((MethodInvoker)delegate()
			//    {
			//        if (_batchPaths.Count == 0)
			//        {
			//            SetupControls(false);
			//            ShowBatchDoneMessage();
			//        }
			//        else
			//        {
			//            StartConvert();
			//        }
			//    });
			//}
		}

		public void StartConvert()
		{
			CUESheet cueSheet;

			try
			{
				_startedAt = DateTime.Now;

				_workThread = null;
				//if (_batchPaths.Count != 0)
				//{
				//    txtInputPath.Text = _batchPaths[0];
				//}

				if (!File.Exists(pathIn))
				{
					throw new Exception("Input CUE Sheet not found.");
				}

				bool outputAudio = _accurateOffset || !_accurateRip;
				cueSheet = new CUESheet(pathIn, _config);
				if (outputAudio)
				{
					bool pathFound = false;
					for (int i = 0; i < 20; i++)
					{
						string outDir = Path.Combine(Path.GetDirectoryName (pathIn), "CUEToolsOutput" + (i > 0? String.Format("({0})",i) : ""));
						if (!Directory.Exists(outDir))
						{
							Directory.CreateDirectory(outDir);
							pathOut = Path.Combine(outDir, Path.GetFileNameWithoutExtension(pathIn) + ".cue");
							pathFound = true;
							break;
						}
					}
					if (!pathFound)
					{
						Text = "Could not create a folder";
						return;
					}
				} else
					pathOut = pathIn;
				cueSheet.GenerateFilenames(_audioFormat, pathOut);
				if (outputAudio)
				{
					if (_cueStyle == CUEStyle.SingleFileWithCUE)
						cueSheet.SingleFilename = Path.ChangeExtension(Path.GetFileName (pathOut), General.FormatExtension (_audioFormat));
				}

				cueSheet.UsePregapForFirstTrackInSingleFile = false;
				cueSheet.AccurateRip = _accurateRip;
				cueSheet.AccurateOffset = _accurateOffset;

				_workThread = new Thread(WriteAudioFilesThread);
				_workClass = cueSheet;
				_workThread.Start(cueSheet);
			}
			catch (Exception ex)
			{
				Text = "Error: " + ex.Message;
				//if (!ShowErrorMessage(ex))
				//{
				//     _batchPaths.Clear();
				//}
				//Close();
			}
		}

		private void frmBatch_Load(object sender, EventArgs e)
		{
			//_batchPaths = new List<string>();
			textBox1.Hide();
			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt");
			string val;
			_config.Load(sr);

			try
			{
				val = sr.Load("CUEStyle");
				_cueStyle = (val != null) ? (CUEStyle)Int32.Parse(val) : CUEStyle.SingleFile;
				val = sr.Load("OutputAudioFormat");
				_audioFormat = (val != null) ? (OutputAudioFormat)Int32.Parse(val) : OutputAudioFormat.WAV;
			}
			catch { };

			StartConvert();
		}

		private void frmBatch_FormClosing(Object sender, FormClosingEventArgs e) 
		{
			if ((_workThread != null) && (_workThread.IsAlive))
			{
				_workClass.Stop();
				e.Cancel = true;
			}
		}
	}
}