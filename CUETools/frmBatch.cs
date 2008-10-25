using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;
using CUEToolsLib;

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
			_batchPaths = new List<string>();
		}

		public void AddInputPath(string path)
		{
			_batchPaths.Add(path);
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
		List<string> _batchPaths;

		public string ShortenString(string input, int max)
		{
			if (input.Length < max) 
				return input;
			return "..." + input.Substring(input.Length - max);
		}

		public void SetStatus(string status, uint percentTrack, double percentDisk, string input, string output)
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
				string inputSuffix = output != null ? "=>" : "";
				if (input == null)
					txtInputFile.Text = inputSuffix;
				else 
					txtInputFile.Text = ShortenString(input, 120) + " " + inputSuffix;
				if (output == null)
					txtOutputFile.Text = "";
				else
					txtOutputFile.Text = ShortenString(output, 120);
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
					if (_batchPaths.Count == 0)
						Text = "Done.";

					//TimeSpan span = DateTime.Now - _startedAt;
					progressBar1.Value = 0;
					progressBar2.Value = 0;
					if (cueSheet.AccurateRip)
					{
						StringWriter sw = new StringWriter();
						cueSheet.GenerateAccurateRipLog(sw);
						textBox1.Text += sw.ToString();
						sw.Close();
						textBox1.Show();
					}
					textBox1.Text += "----------------------------------------------------------\r\n";
					textBox1.Select(0, 0);
				});
			}
			catch (StopException)
			{
				_batchPaths.Clear();
				this.Invoke((MethodInvoker)delegate()
				{
					Text = "Aborted.";
					textBox1.Text += "Aborted.";
					progressBar1.Value = 0;
					progressBar2.Value = 0;
				});
			}
			catch (Exception ex)
			{
				this.Invoke((MethodInvoker)delegate()
				{
					Text = "Error: " + ex.Message;
					textBox1.Show();
					textBox1.Text += "Error: " + ex.Message + "\r\n";
					textBox1.Text += "----------------------------------------------------------\r\n";
					textBox1.Select(0, 0);
				});
			}

			if (_batchPaths.Count != 0)
			{
				_batchPaths.RemoveAt(0);
				this.BeginInvoke((MethodInvoker)delegate()
				{
					if (_batchPaths.Count == 0)
					{
						Text = "All done.";
					}
					else
					{
						StartConvert();
					}
				});
			}
		}

		public void StartConvert()
		{
			CUESheet cueSheet;

			try
			{
				_startedAt = DateTime.Now;

				_workThread = null;
				if (_batchPaths.Count != 0)
				    pathIn = _batchPaths[0];

				pathIn = Path.GetFullPath(pathIn);

				textBox1.Text += "Processing " + pathIn + ":\r\n";
				textBox1.Select (0,0);

				if (!File.Exists(pathIn))
					throw new Exception("Input CUE Sheet not found.");

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
						throw new Exception("Could not create a folder.");
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
				_workThread.Priority = ThreadPriority.BelowNormal;
				_workThread.IsBackground = true;
				_workThread.Start(cueSheet);
			}
			catch (Exception ex)
			{
				Text = "Error: " + ex.Message;
				textBox1.Show();
				textBox1.Text += "Error: " + ex.Message + "\r\n";
				textBox1.Text += "----------------------------------------------------------\r\n";
				textBox1.Select(0, 0);
			}
			if ((_workThread == null) && (_batchPaths.Count != 0))
			{
				_batchPaths.RemoveAt(0);
				if (_batchPaths.Count == 0)
				{
					Text = "All done.";
				}
				else
					StartConvert();
			}
		}

		private void frmBatch_Load(object sender, EventArgs e)
		{
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

			if (_accurateOffset || !_accurateRip)
				txtOutputFile.Show();

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