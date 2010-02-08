using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;
using System.Diagnostics;
using CUETools.Compression;
using CUETools.Processor;

namespace JDP
{
	public partial class frmBatch : Form
	{
		public frmBatch()
		{
			InitializeComponent();
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

		public string Profile
		{
			get { return _profileName; }
			set { _profileName = value; }
		}

		CUEToolsProfile _profile = null;
		string _profileName = "verify";
		Thread _workThread;
		CUESheet _workClass;
		string pathIn;
		string pathOut;
		bool _reducePriority;
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
				progressBar1.Value = Math.Max(0,Math.Min(100,(int)(e.percentTrck*100)));
				progressBar2.Value = Math.Max(0,Math.Min(100,(int)(e.percentDisk*100)));
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

		private void PasswordRequired(object sender, CompressionPasswordRequiredEventArgs e)
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
			CUEToolsScript script = _profile._script == null || !_profile._config.scripts.ContainsKey(_profile._script) ? null : _profile._config.scripts[_profile._script];

			try
			{
				_startedAt = DateTime.Now;
				if (_batchPaths.Count != 0)
					pathIn = _batchPaths[0];

				pathIn = Path.GetFullPath(pathIn);

				this.Invoke((MethodInvoker)delegate()
				{
					textBox1.Text += "Processing " + pathIn + ":\r\n";
					textBox1.Select(0, 0);
				});

				if (!File.Exists(pathIn) && !Directory.Exists(pathIn))
					throw new Exception("Input CUE Sheet not found.");

				if (_profile._action == CUEAction.CorrectFilenames)
					throw new Exception("CorrectFilenames action not yet supported in commandline mode.");
				if (_profile._action == CUEAction.CreateDummyCUE)
					throw new Exception("CreateDummyCUE action not yet supported in commandline mode.");

				bool useAR = _profile._action == CUEAction.Verify || _profile._useAccurateRip;

				cueSheet.Action = _profile._action;
				cueSheet.OutputStyle = _profile._CUEStyle;
				cueSheet.WriteOffset = _profile._writeOffset;
				cueSheet.Open(pathIn);
				if (useAR)
					cueSheet.UseAccurateRip();

				pathOut = CUESheet.GenerateUniqueOutputPath(_profile._config, 
					_profile._outputTemplate, 
					_profile._CUEStyle == CUEStyle.SingleFileWithCUE ? "." + _profile._outputAudioFormat : ".cue", 
					_profile._action, 
					new NameValueCollection(),
					pathIn, 
					cueSheet);
				if (pathOut == "" || (_profile._action != CUEAction.Verify && File.Exists(pathOut)))
					throw new Exception("Could not generate output path.");

				cueSheet.GenerateFilenames(_profile._outputAudioType, _profile._outputAudioFormat, pathOut);
				cueSheet.UsePregapForFirstTrackInSingleFile = false;

				if (script == null)
					cueSheet.Go();
				else
					/* status = */ cueSheet.ExecuteScript(script);
				this.Invoke((MethodInvoker)delegate()
				{
					if (_batchPaths.Count == 0)
						Text = "Done.";

					//TimeSpan span = DateTime.Now - _startedAt;
					progressBar1.Value = 0;
					progressBar2.Value = 0;
					if (cueSheet.IsCD)
					{
						textBox1.Text += cueSheet.LOGContents;
						textBox1.Show();
					}
					else if (useAR)
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
				CUESheet cueSheet = new CUESheet(_profile._config);
				cueSheet.PasswordRequired += new EventHandler<CompressionPasswordRequiredEventArgs>(PasswordRequired);
				cueSheet.CUEToolsProgress += new EventHandler<CUEToolsProgressEventArgs>(SetStatus);

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
			//SettingsReader sr = new SettingsReader("CUE Tools", "settings.txt", Application.ExecutablePath);
			//_profile.Load(sr);
			//_reducePriority = sr.LoadBoolean("ReducePriority") ?? true;
			_reducePriority = true;

			_profile = new CUEToolsProfile(_profileName);
			SettingsReader sr = new SettingsReader("CUE Tools", string.Format("profile-{0}.txt", _profileName), Application.ExecutablePath);
			_profile.Load(sr);
			
			if (_reducePriority)
				Process.GetCurrentProcess().PriorityClass = System.Diagnostics.ProcessPriorityClass.Idle;

			if (_profile._action != CUEAction.Verify)
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