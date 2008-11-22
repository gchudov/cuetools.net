using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;
using System.Diagnostics;
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
		bool _reducePriority;
		bool _lossyWAV;
		DateTime _startedAt;
		List<string> _batchPaths;

		public static string ShortenString(string input, int max)
		{
			if (input.Length < max) 
				return input;
			return "..." + input.Substring(input.Length - max);
		}

		public void SetStatus(object sender, CUEToolsProgressEventArgs e)
		{
			this.BeginInvoke((MethodInvoker)delegate()
			{
				if (e.percentDisk == 0)
				{
					_startedAt = DateTime.Now;
					Text = e.status;
				}
				else if (e.percentDisk > 0.02)
				{
					TimeSpan span = DateTime.Now - _startedAt;
					TimeSpan eta = new TimeSpan ((long) (span.Ticks/e.percentDisk));
					Text = String.Format("{0}, ETA {1}:{2:00}.", e.status, (int)eta.TotalMinutes, eta.Seconds);
				} else
					Text = e.status;
				progressBar1.Value = (int)e.percentTrack;
				progressBar2.Value = (int)(e.percentDisk*100);
				string inputSuffix = e.output != null ? "=>" : "";
				if (e.input == null)
					txtInputFile.Text = inputSuffix;
				else 
					txtInputFile.Text = ShortenString(e.input, 120) + " " + inputSuffix;
				if (e.output == null)
					txtOutputFile.Text = "";
				else
					txtOutputFile.Text = ShortenString(e.output, 120);
			});
		}

		private void PasswordRequired(object sender, ArchivePasswordRequiredEventArgs e)
		{
			this.Invoke((MethodInvoker)delegate()
			{
				frmPassword dlg = new frmPassword();
				if (dlg.ShowDialog(this) == DialogResult.OK)
				{
					e.Password = dlg.txtPassword.Text;
					e.ContinueOperation = true;
				}
				else
					e.ContinueOperation = false;
			});
		}

		private void WriteAudioFilesThread(object o)
		{
			CUESheet cueSheet = (CUESheet)o;

			try
			{
				_startedAt = DateTime.Now;
				if (_batchPaths.Count != 0)
					pathIn = _batchPaths[0];

				pathIn = Path.GetFullPath(pathIn);

				textBox1.Text += "Processing " + pathIn + ":\r\n";
				textBox1.Select(0, 0);

				string cueName;
				if (!File.Exists(pathIn))
				{
					if (!Directory.Exists(pathIn))
						throw new Exception("Input CUE Sheet not found.");
					if (!pathIn.EndsWith(new string(Path.DirectorySeparatorChar, 1)))
						pathIn = pathIn + Path.DirectorySeparatorChar;
					cueName = Path.GetFileNameWithoutExtension(Path.GetDirectoryName(pathIn)) + ".cue";
				}
				else
					cueName = Path.GetFileNameWithoutExtension(pathIn) + ".cue";

				bool outputAudio = _accurateOffset || !_accurateRip;
				cueSheet.Open(pathIn, _lossyWAV);
				if (outputAudio)
				{
					bool pathFound = false;
					for (int i = 0; i < 20; i++)
					{
						string outDir = Path.Combine(Path.GetDirectoryName(pathIn), "CUEToolsOutput" + (i > 0 ? String.Format("({0})", i) : ""));
						if (!Directory.Exists(outDir))
						{
							Directory.CreateDirectory(outDir);
							pathOut = Path.Combine(outDir, cueName);
							pathFound = true;
							break;
						}
					}
					if (!pathFound)
						throw new Exception("Could not create a folder.");
				}
				else
					pathOut = Path.Combine(Path.GetDirectoryName(pathIn), cueName);
				cueSheet.GenerateFilenames(_audioFormat, pathOut);
				if (outputAudio)
				{
					if (_cueStyle == CUEStyle.SingleFileWithCUE)
						cueSheet.SingleFilename = Path.ChangeExtension(Path.GetFileName(pathOut), General.FormatExtension(_audioFormat));
				}

				cueSheet.UsePregapForFirstTrackInSingleFile = false;
				cueSheet.AccurateRip = _accurateRip;
				cueSheet.AccurateOffset = _accurateOffset;
				cueSheet.WriteAudioFiles(Path.GetDirectoryName(pathOut), _cueStyle);
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
#if !DEBUG
			catch (Exception ex)
			{
				this.Invoke((MethodInvoker)delegate()
				{
					Text = "Error: " + ex.Message;
					textBox1.Show();
					textBox1.Text += "Error";
					for (Exception e = ex; e != null; e = e.InnerException)
						textBox1.Text += ": " + e.Message;
					textBox1.Text += "\r\n----------------------------------------------------------\r\n";
					textBox1.Select(0, 0);
				});
			}
#endif

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
			try
			{
				CUESheet cueSheet = new CUESheet(_config);
				cueSheet.PasswordRequired += new ArchivePasswordRequiredHandler(PasswordRequired);
				cueSheet.CUEToolsProgress += new CUEToolsProgressHandler(SetStatus);

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
		}

		private void frmBatch_Load(object sender, EventArgs e)
		{
			textBox1.Hide();
			SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt");

			_config.Load(sr);
			_reducePriority = sr.LoadBoolean("ReducePriority") ?? true;
			_cueStyle = (CUEStyle?)sr.LoadInt32("CUEStyle", null, null) ?? CUEStyle.SingleFileWithCUE;
			_audioFormat = (OutputAudioFormat?)sr.LoadInt32("OutputAudioFormat", null, null) ?? OutputAudioFormat.WAV;
			_lossyWAV = sr.LoadBoolean("LossyWav") ?? false;
			
			if (_reducePriority)
				Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.Idle;

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